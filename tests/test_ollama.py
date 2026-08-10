"""Tests that need a real Ollama. Skipped automatically when it is not there.

The point of this file is the thing the old code could not do at all: run the same
application against embedding models of different dimensions and have it work.
"""

import numpy as np
import pytest
from conftest import normal_script, read_events

import backends
from backends import OllamaEmbedding

pytestmark = pytest.mark.ollama


def installed() -> set[str]:
    return {m["name"] for m in backends.list_models()}


def require(model: str) -> str:
    names = installed()
    if not names:
        pytest.skip("Ollama is not reachable")
    for name in names:
        if name == model or name.split(":")[0] == model.split(":")[0]:
            return name
    pytest.skip(f"Ollama does not have {model}")


@pytest.fixture(scope="module")
def minilm():
    return OllamaEmbedding(require("all-minilm"))


class TestRealEmbeddings:
    def test_reports_its_own_dimension(self, minilm):
        assert minilm.dim == 384
        assert minilm.describe()["dim"] == 384

    def test_vectors_are_normalised(self, minilm):
        vectors = minilm.embed(["the cat sat on the mat", "a dog in the park"])
        assert vectors.shape == (2, 384)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_batch_row_count_matches_input_count(self, minilm):
        """A silent short return misaligns every vector after the gap."""
        texts = [f"sentence number {i}" for i in range(12)]
        assert minilm.embed(texts).shape[0] == len(texts)

    def test_embedded_newlines_do_not_split_a_record(self, minilm):
        """One text must produce exactly one vector, whatever is inside it."""
        assert minilm.embed(["first line\nsecond line\n\nthird line"]).shape[0] == 1

    def test_related_text_scores_above_unrelated_text(self, minilm):
        vectors = minilm.embed(
            ["The capital of France is Paris.", "Paris is the French capital city.", "Diesel engine maintenance."]
        )
        related = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        assert related > unrelated

    def test_missing_model_names_the_pull_command(self):
        backend = OllamaEmbedding("definitely-not-a-real-model:v9")
        with pytest.raises(backends.BackendError) as excinfo:
            backend.embed(["hello"])
        assert "ollama pull" in excinfo.value.hint


class TestDimensionIndependence:
    """The old similarity search hardcoded 4096 and broke on every other model."""

    @pytest.mark.parametrize("model,expected_dim", [("all-minilm", 384), ("nomic-embed-text", 768)])
    def test_app_runs_end_to_end(self, flask_client, model, expected_dim):
        backend = OllamaEmbedding(require(model))
        if backend.dim != expected_dim:
            pytest.skip(f"{model} reported {backend.dim} dimensions, expected {expected_dim}")

        client, _ = flask_client(normal_script(), embed=backend)
        events = read_events(client.get("/query?query=What+is+the+capital+of+France"))

        assert not [e for e in events if e["type"] == "error"]
        done = [e for e in events if e["type"] == "done"][0]
        assert done["embedding"]["dim"] == expected_dim
        assert [e for e in events if e["type"] == "similar"][0]["items"]

    def test_switching_model_does_not_corrupt_the_store(self, tmp_path):
        """Mixed-dimension rows in one database used to break search silently."""
        from store import EmbeddingStore

        store = EmbeddingStore(str(tmp_path / "mixed.db"))
        small = OllamaEmbedding(require("all-minilm"))
        store.add("small vector text", small.embed(["small vector text"])[0], model=small.model)
        store.add("a fake 4096-d row", np.ones(4096, dtype=np.float32), model="legacy")

        hits = store.find_similar(small.embed(["small vector text"])[0], model=small.model)
        assert len(hits) == 1
        assert hits[0]["text"] == "small vector text"
        store.close()


class TestModelDiscovery:
    def test_auto_selection_prefers_an_embedding_model(self):
        if not any(m["is_embedding"] for m in backends.list_models()):
            pytest.skip("no embedding model installed")
        backend = OllamaEmbedding("")
        assert backend.auto_selected
        assert any(
            token in backend.model.lower() for token in ("embed", "minilm")
        ), f"auto-selected {backend.model}, which is not an embedding model"

    def test_health_is_ok_when_the_chat_model_exists(self, monkeypatch):
        names = installed()
        if not names:
            pytest.skip("Ollama is not reachable")
        monkeypatch.setattr(backends, "DEFAULT_CHAT_MODEL", sorted(names)[0])
        assert backends.health()["ok"] is True


class TestRealChat:
    def test_a_real_model_produces_a_parseable_step(self):
        """The response schema is what stops a step from arriving as prose."""
        chat_model = require("llama3.2")
        chat = backends.OllamaChat(chat_model)
        text = "".join(
            chat.stream(
                [
                    {"role": "system", "content": "Answer in JSON."},
                    {"role": "user", "content": "Give one reasoning step about 2+2."},
                ],
                120,
                schema=backends.STEP_SCHEMA,
            )
        )
        from reasoning import extract_json

        parsed = extract_json(text)
        assert set(parsed) >= {"title", "content", "next_action"}
        assert parsed["next_action"] in ("continue", "final_answer")
