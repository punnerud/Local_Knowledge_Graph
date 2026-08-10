"""Reading embeddings from inside a model.

The addressing and pooling logic is tested against synthetic modules, so it runs
everywhere with no download. The tests that need real weights are marked ``hf`` and
skipped when torch or the model is unavailable.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from layers import (  # noqa: E402
    describe_layers,
    find_block_stack,
    find_final_norm,
    pool,
    resolve_layer,
)


class Block(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.mlp = nn.Linear(dim, dim)
        self.attn = nn.Linear(dim, dim)

    def forward(self, x):
        return self.mlp(x) + self.attn(x)


class OtherBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x)


class ToyModel(nn.Module):
    """Shaped like a decoder: an embedding, a block stack, a final norm."""

    def __init__(self, n_blocks=6, dim=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(20, dim)
        self.layers = nn.ModuleList([Block(dim) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.embed_tokens(x)
        for block in self.layers:
            h = block(h)
        return self.norm(h)


class TestBlockStackDetection:
    def test_finds_the_repeated_stack(self):
        name, stack = find_block_stack(ToyModel(n_blocks=6))
        assert name == "layers"
        assert len(stack) == 6

    def test_prefers_the_longest_uniform_stack(self):
        class TwoStacks(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapters = nn.ModuleList([OtherBlock() for _ in range(3)])
                self.layers = nn.ModuleList([Block() for _ in range(9)])

        name, stack = find_block_stack(TwoStacks())
        assert name == "layers" and len(stack) == 9

    def test_ignores_a_mixed_module_list(self):
        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                # Longer, but heterogeneous: not a block stack.
                self.mixed = nn.ModuleList([Block(), OtherBlock(), Block(), OtherBlock()])
                self.layers = nn.ModuleList([Block() for _ in range(3)])

        name, _ = find_block_stack(Mixed())
        assert name == "layers"

    def test_a_model_with_no_stack_says_so(self):
        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

        with pytest.raises(ValueError, match="block stack"):
            find_block_stack(Flat())

    def test_describe_layers_lists_what_can_be_addressed(self):
        info = describe_layers(ToyModel(n_blocks=4))
        assert info["n_blocks"] == 4
        assert info["block_type"] == "Block"
        assert info["addresses"] == ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
        assert set(info["sub_modules"]) == {"mlp", "attn"}
        assert info["has_final_norm"] is True


class TestLayerAddressing:
    def test_positive_index(self):
        model = ToyModel(n_blocks=6)
        assert resolve_layer(model, "blocks.2") is model.layers[2]

    def test_negative_index_counts_from_the_end(self):
        model = ToyModel(n_blocks=6)
        assert resolve_layer(model, "blocks.-1") is model.layers[5]
        assert resolve_layer(model, "blocks.-2") is model.layers[4]

    def test_sub_module_of_a_block(self):
        model = ToyModel(n_blocks=6)
        assert resolve_layer(model, "blocks.3.mlp") is model.layers[3].mlp

    def test_explicit_dotted_path_bypasses_detection(self):
        model = ToyModel(n_blocks=6)
        assert resolve_layer(model, "layers.1.attn") is model.layers[1].attn

    def test_out_of_range_reports_the_available_range(self):
        with pytest.raises(IndexError, match="6 blocks"):
            resolve_layer(ToyModel(n_blocks=6), "blocks.99")

    def test_non_numeric_index_is_rejected(self):
        with pytest.raises(ValueError, match="non-numeric"):
            resolve_layer(ToyModel(), "blocks.middle")

    def test_final_norm_is_found(self):
        assert isinstance(find_final_norm(ToyModel()), nn.LayerNorm)


class TestPooling:
    def setup_method(self):
        # Two sequences of three tokens; the second is padded to length two.
        self.hidden = torch.tensor(
            [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
             [[4.0, 0.0], [5.0, 0.0], [0.0, 0.0]]]
        )
        self.mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    def test_last_token_skips_right_padding(self):
        """Pooling the pad instead of the last real token is the classic bug."""
        out = pool(self.hidden, self.mask, "last")
        assert out[0, 0].item() == pytest.approx(3.0)
        assert out[1, 0].item() == pytest.approx(5.0)

    def test_last_token_handles_left_padding(self):
        hidden = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        mask = torch.tensor([[0, 1, 1]])
        assert pool(hidden, mask, "last")[0, 0].item() == pytest.approx(2.0)

    def test_mean_ignores_padding(self):
        out = pool(self.hidden, self.mask, "mean")
        assert out[0, 0].item() == pytest.approx(2.0)
        assert out[1, 0].item() == pytest.approx(4.5)

    def test_cls_takes_the_first_token(self):
        out = pool(self.hidden, self.mask, "cls")
        assert out[0, 0].item() == pytest.approx(1.0)

    def test_unknown_pooling_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown pooling"):
            pool(self.hidden, self.mask, "median")


HF_MODEL = "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="module")
def probe():
    pytest.importorskip("transformers")
    from layers import HiddenStateEmbedding

    try:
        return HiddenStateEmbedding(HF_MODEL, layer="blocks.-1")
    except Exception as exc:  # offline, or the model is not cached
        pytest.skip(f"{HF_MODEL} unavailable: {exc}")


@pytest.mark.hf
class TestRealModel:
    def test_produces_normalised_vectors_of_the_model_width(self, probe):
        vectors = probe.embed(["one", "two", "three"])
        assert vectors.shape == (3, probe.dim)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_related_text_scores_above_unrelated(self, probe):
        v = probe.embed(
            ["The capital of France is Paris.",
             "Paris is the French capital city.",
             "Diesel engine maintenance schedules."]
        )
        assert float(v[0] @ v[1]) > float(v[0] @ v[2])

    def test_the_same_text_always_gives_the_same_vector(self, probe):
        a = probe.embed(["deterministic please"])
        b = probe.embed(["deterministic please"])
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_batching_does_not_shift_a_vector_onto_the_wrong_text(self, probe):
        """The invariant that matters at a batch boundary.

        Not elementwise equality: padding changes the shapes the kernels see, so a
        batched run differs from a single one in the fourth decimal. What must hold
        is that each vector still belongs to its own text -- the failure mode is a
        vector sliding onto its neighbour, and that shows up as an off-diagonal
        maximum, not as small noise.
        """
        texts = [f"sentence number {i}" for i in range(10)]
        one_by_one = np.vstack([probe.embed([t]) for t in texts])
        batched = probe.embed(texts)

        agreement = batched @ one_by_one.T
        assert np.all(agreement.diagonal() > 0.999), "a vector drifted from its own text"
        assert np.array_equal(agreement.argmax(axis=1), np.arange(len(texts))), (
            "a vector matches another text better than its own"
        )

    def test_different_layers_give_different_answers(self, probe):
        """If two layers agree exactly, the hook is not tapping where it claims."""
        from layers import MultiLayerProbe

        out = MultiLayerProbe(HF_MODEL, ["blocks.0", "blocks.-1"]).embed(["a probe sentence"])
        assert not np.allclose(out["blocks.0"], out["blocks.-1"], atol=1e-3)

    def test_depth_separates_topics_better_than_the_first_layer(self, probe):
        """The whole point of the feature, as an assertion."""
        from layers import MultiLayerProbe
        from scripts.layer_sweep import separation

        texts = ["The capital of France is Paris.", "Paris is the French capital.",
                 "Diesel engines need oil changes.", "Servicing a diesel engine."]
        labels = ["fr", "fr", "eng", "eng"]

        out = MultiLayerProbe(HF_MODEL, ["blocks.0", "blocks.-1"]).embed(texts)
        first = separation(out["blocks.0"], labels)["separation"]
        last = separation(out["blocks.-1"], labels)["separation"]
        assert last > first

    def test_a_module_off_the_forward_path_is_reported(self, probe):
        from layers import HiddenStateEmbedding

        stray = HiddenStateEmbedding.__new__(HiddenStateEmbedding)
        stray.__dict__.update(probe.__dict__)
        stray._module = torch.nn.Linear(4, 4)  # never called by the model
        stray._dim = None
        with pytest.raises(RuntimeError, match="never fired"):
            stray.embed(["anything"])
