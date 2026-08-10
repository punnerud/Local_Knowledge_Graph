"""Read embeddings from inside a model, not just from an embedding endpoint.

Ollama and every other embedding API hand back one pooled vector from the top of
the stack. This module taps a chosen point *inside* a transformer instead, so the
same application can ask what the reasoning steps look like at layer 4, at layer
16, and at the output -- including for models that expose no embedding endpoint at
all, because a forward pass is all that is required.

Three things here are load-bearing and are the usual sources of silently wrong
layer comparisons:

* **Hooks, not ``output_hidden_states=True``.** The flag materialises
  ``(n_layers + 1, batch, tokens, hidden)`` before you pool it, which is several
  gigabytes on an 8B model at a realistic batch and context. A hook that pools
  inside itself and drops the tensor never allocates that.
* **The last layer is normalised and the others are not.** In Hugging Face decoder
  models the per-layer hidden states are the raw residual stream, while the final
  entry has already been through the model's final norm. Comparing them without
  applying that norm yourself compares vectors from two different spaces, and the
  result looks plausible rather than broken.
* **Layer addresses are not ``model.model.layers``.** That path is a Llama detail.
  The block stack is found structurally instead, which covers Llama, Qwen, Mistral,
  Gemma, Phi, GPT-2, GPT-NeoX, Falcon, BERT, ViT and CLIP without a per-architecture
  table.

Requires the optional extras:  pip install torch transformers
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np

POOLINGS = ("last", "mean", "cls")


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading internal layers needs PyTorch. Install it with:\n"
            "    pip install torch transformers"
        ) from exc
    return torch


def find_block_stack(model) -> tuple[str, Any]:
    """Locate the repeated transformer blocks, structurally.

    Returns ``(name, module_list)`` for the longest ``nn.ModuleList`` whose children
    are all instances of one class. That is what a transformer block stack looks
    like in every architecture worth supporting, and it needs no per-model table.
    """
    torch = _require_torch()

    best: tuple[str, Any] | None = None
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) < 2:
            continue
        kinds = {type(child).__name__ for child in module}
        if len(kinds) != 1:
            continue
        if best is None or len(module) > len(best[1]):
            best = (name, module)

    if best is None:
        raise ValueError(
            "Could not find a repeated block stack in this model. Address a module "
            "explicitly instead, for example layer='encoder.layer.6'."
        )
    return best


def find_final_norm(model):
    """The normalisation applied after the last block, if the architecture has one.

    Needed to bring intermediate layers into the same space as the final one. Absent
    on some architectures, in which case cross-layer comparison is still possible but
    the scales differ and that is worth knowing.
    """
    for path in ("model.norm", "transformer.ln_f", "model.final_layernorm",
                 "gpt_neox.final_layer_norm", "encoder.final_layer_norm", "norm", "ln_f"):
        try:
            return model.get_submodule(path)
        except AttributeError:
            continue
    return None


def resolve_layer(model, address: str):
    """Turn a layer address into a module.

    Accepts ``blocks.N`` and ``blocks.-N`` against the detected block stack, an
    optional sub-path such as ``blocks.-1.mlp``, or any dotted module path.
    """
    if not address.startswith("blocks"):
        return model.get_submodule(address)

    stack_name, stack = find_block_stack(model)
    parts = address.split(".")
    if len(parts) < 2:
        raise ValueError(f"Layer address {address!r} needs an index, e.g. 'blocks.-1'")

    try:
        index = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Layer address {address!r} has a non-numeric index") from exc

    if not -len(stack) <= index < len(stack):
        raise IndexError(
            f"Layer {index} is out of range: this model has {len(stack)} blocks "
            f"({stack_name}.0 .. {stack_name}.{len(stack) - 1})"
        )

    full = f"{stack_name}.{index % len(stack)}"
    if len(parts) > 2:
        full = full + "." + ".".join(parts[2:])
    return model.get_submodule(full)


def describe_layers(model) -> dict:
    """What can be addressed on this model. Useful when picking a layer."""
    stack_name, stack = find_block_stack(model)
    return {
        "block_stack": stack_name,
        "n_blocks": len(stack),
        "block_type": type(stack[0]).__name__,
        "addresses": [f"blocks.{i}" for i in range(len(stack))],
        "sub_modules": [name for name, _ in stack[0].named_children()],
        "has_final_norm": find_final_norm(model) is not None,
    }


def pool(hidden, attention_mask, how: str):
    """Reduce ``(batch, tokens, hidden)`` to ``(batch, hidden)``.

    ``last`` is the architecturally correct choice for a decoder-only model: under a
    causal mask only the final position has attended to the whole sequence. ``mean``
    often scores better on retrieval benchmarks anyway, so both are offered rather
    than one being assumed.
    """
    torch = _require_torch()

    if how == "cls":
        return hidden[:, 0]
    if how == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    if how == "last":
        # Works for both padding sides. Getting this wrong pools the padding.
        left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padded:
            return hidden[:, -1]
        index = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), index]
    raise ValueError(f"Unknown pooling {how!r}, expected one of {POOLINGS}")


class HiddenStateEmbedding:
    """An ``EmbeddingBackend`` that reads a chosen layer inside a local model.

    >>> probe = HiddenStateEmbedding("HuggingFaceTB/SmolLM2-135M", layer="blocks.8")
    >>> probe.embed(["some text"]).shape
    (1, 576)
    """

    def __init__(
        self,
        model_name: str,
        *,
        layer: str = "blocks.-1",
        pooling: str = "last",
        apply_final_norm: bool = True,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        torch = _require_torch()
        from transformers import AutoModel, AutoTokenizer

        if pooling not in POOLINGS:
            raise ValueError(f"Unknown pooling {pooling!r}, expected one of {POOLINGS}")

        self.model_name = model_name
        self.layer = layer
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

        self._module = resolve_layer(self.model, layer)
        self._final_norm = find_final_norm(self.model) if apply_final_norm else None
        self.applies_final_norm = self._final_norm is not None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[1])
        return self._dim

    def describe(self) -> dict:
        return {
            "kind": "hidden-state",
            "model": f"{self.model_name}@{self.layer}",
            "dim": self.dim,
            "layer": self.layer,
            "pooling": self.pooling,
            "final_norm_applied": self.applies_final_norm,
            "device": str(self.device),
        }

    def embed(self, texts) -> np.ndarray:
        items = [" ".join(str(t).split()) or " " for t in texts]
        if not items:
            return np.zeros((0, self._dim or 0), dtype=np.float32)

        chunks = [
            self._embed_batch(items[i : i + self.batch_size])
            for i in range(0, len(items), self.batch_size)
        ]
        matrix = np.vstack(chunks).astype(np.float32)
        self._dim = int(matrix.shape[1])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms > 0)
        return matrix

    def _embed_batch(self, items: list[str]) -> np.ndarray:
        torch = _require_torch()

        batch = self.tokenizer(
            items, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.device)

        captured: dict[str, Any] = {}
        # A weak reference keeps the hook from holding this object alive through the
        # closure, which is a real leak when probes are created per layer.
        self_ref = weakref.ref(self)

        def hook(_module, _args, output):
            probe = self_ref()
            if probe is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach()
            # Intermediate blocks emit the raw residual stream while the model's own
            # last hidden state has already been normalised. Applying the final norm
            # here is what puts every layer in one comparable space.
            if probe._final_norm is not None:
                hidden = probe._final_norm(hidden)
            captured["pooled"] = pool(hidden, batch["attention_mask"], probe.pooling).float().cpu()

        handle = self._module.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                self.model(**batch)
        finally:
            handle.remove()

        if "pooled" not in captured:
            raise RuntimeError(
                f"The hook on {self.layer!r} never fired. That module is not on this "
                "model's forward path; use describe_layers() to see what is."
            )
        return captured["pooled"].numpy()


class MultiLayerProbe:
    """Read several layers in a single forward pass.

    Comparing layers is the whole point, and running the model once per layer costs
    N times as much for exactly the same computation.
    """

    def __init__(self, model_name: str, layers: list[str], **kwargs) -> None:
        self.model_name = model_name
        self.layers = list(layers)
        first = HiddenStateEmbedding(model_name, layer=self.layers[0], **kwargs)
        self.probes = {self.layers[0]: first}
        for layer in self.layers[1:]:
            probe = HiddenStateEmbedding.__new__(HiddenStateEmbedding)
            probe.__dict__.update(first.__dict__)
            probe.layer = layer
            probe._module = resolve_layer(first.model, layer)
            probe._dim = None
            self.probes[layer] = probe

    def embed(self, texts) -> dict[str, np.ndarray]:
        return {layer: probe.embed(texts) for layer, probe in self.probes.items()}

    def describe(self) -> dict:
        return {
            "kind": "multi-layer",
            "model": self.model_name,
            "layers": self.layers,
            "info": describe_layers(self.probes[self.layers[0]].model),
        }
