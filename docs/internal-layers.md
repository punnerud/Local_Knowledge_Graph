# Embeddings from inside a model

An embedding endpoint gives you one pooled vector from the top of the stack. You can instead
tap a chosen point *inside* a local model — which also makes models with no embedding API
usable, since a forward pass is all that is required:

```bash
pip install torch transformers

LKG_EMBED_BACKEND=hf \
LKG_HF_MODEL=HuggingFaceTB/SmolLM2-135M \
LKG_HF_LAYER=blocks.-1 \
mpe-lkg
```

Layers are addressed structurally, not by a per-architecture path: `blocks.0`, `blocks.12`,
`blocks.-1`, `blocks.-1.mlp`, or any explicit dotted module path. The block stack is found by
looking for the longest `nn.ModuleList` whose children share one class, which covers Llama,
Qwen, Mistral, Gemma, Phi, GPT-2, GPT-NeoX, Falcon, BERT, ViT and CLIP without a lookup table.
`LKG_HF_POOLING` selects `last` (default, and the only architecturally correct choice for a
decoder under a causal mask), `mean`, or `cls`.

### Does the depth matter?

`make sweep` runs the same four-topic corpus through several layers and reports how far each
one puts steps of the same topic from steps of a different topic. On `SmolLM2-135M`:

| Layer | Within topic | Across topics | Separation |
|---|---|---|---|
| `blocks.0` | 0.998 | 0.996 | **0.002** |
| `blocks.7` | 0.895 | 0.834 | 0.062 |
| `blocks.15` | 0.914 | 0.863 | 0.051 |
| `blocks.22` | 0.893 | 0.780 | 0.113 |
| `blocks.29` | 0.926 | 0.779 | **0.148** |

The first block cannot tell the topics apart at all — it sees each token before any context
has been mixed in — and that near-zero is the control that says the separation deeper in is
real rather than an artefact of the metric. Separation grows roughly seventyfold with depth.

Two details that quietly ruin a layer comparison if you skip them, and which this handles:
intermediate blocks emit the raw residual stream while the model's own last hidden state has
already been through the final norm, so that norm is applied to every layer to put them in one
space; and the states are captured with forward hooks that pool inside the hook rather than
with `output_hidden_states=True`, which would materialise every layer at once — several
gigabytes on an 8B model before any pooling happens.
