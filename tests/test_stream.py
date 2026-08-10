"""End-to-end behaviour of the /query stream, with no model and no network.

The two reported issues were both "the page stays blank". These tests pin the two
distinct causes: an exception thrown before the stream opened, and a retry loop that
never yielded anything.
"""

import json

import pytest
from conftest import normal_script, read_events, step

from mpe_lkg.backends import BackendError, DeterministicEmbedding


class FailingEmbedding:
    """Stands in for Ollama answering 404 because the model is not installed."""

    def __init__(self, message="Ollama does not have the embedding model 'llama3.1:8b'."):
        self._message = message

    @property
    def dim(self):
        return 8

    def describe(self):
        return {"kind": "failing", "model": "llama3.1:8b", "dim": 8}

    def embed(self, texts):
        raise BackendError(self._message, hint="Install it with:  ollama pull llama3.1:8b")


class TestHappyPath:
    def test_stream_reaches_a_final_answer(self, flask_client):
        client, _ = flask_client(normal_script())
        events = read_events(client.get("/query?query=capital+of+France"))
        kinds = [e["type"] for e in events]

        assert "step" in kinds
        assert kinds.count("final") == 1
        assert kinds.count("done") == 1
        assert kinds[-1] == "done_stream"
        assert "error" not in kinds

    def test_every_step_carries_a_drawable_graph(self, flask_client):
        client, _ = flask_client(normal_script())
        events = read_events(client.get("/query?query=q"))

        for event in [e for e in events if e["type"] in ("step", "final")]:
            graph = event["graph"]
            known = {n["id"] for n in graph["nodes"]}
            assert known, "a step must always produce at least one node"
            for edge in graph["edges"]:
                assert edge["from"] in known and edge["to"] in known
                assert "length" in edge

    def test_node_count_grows_by_one_per_step(self, flask_client):
        client, _ = flask_client(normal_script())
        events = read_events(client.get("/query?query=q"))
        counts = [len(e["graph"]["nodes"]) for e in events if e["type"] == "step"]
        assert counts == list(range(1, len(counts) + 1))

    def test_final_answer_does_not_duplicate_the_last_step(self, flask_client):
        """Two nodes over identical text produce a spurious 1.00 edge between them."""
        client, _ = flask_client(normal_script())
        events = read_events(client.get("/query?query=q"))
        final = [e for e in events if e["type"] == "final"][0]

        labels = [n["label"] for n in final["graph"]["nodes"]]
        assert sum(1 for label in labels if label.startswith("Final Answer")) == 1
        assert all(edge["value"] < 0.999 for edge in final["graph"]["edges"])

    def test_related_items_are_reported_with_named_fields(self, flask_client):
        client, _ = flask_client(normal_script())
        events = read_events(client.get("/query?query=q"))
        similar = [e for e in events if e["type"] == "similar"]
        assert similar
        for item in similar[0]["items"]:
            assert set(item) == {"id", "text", "similarity", "is_question"}
            assert -1.0 <= item["similarity"] <= 1.0


class TestIssueOneBlankPage:
    """An embedding failure must arrive as an event, not as an HTTP 500."""

    def test_embedding_failure_yields_an_error_event(self, flask_client):
        client, _ = flask_client(normal_script(), embed=FailingEmbedding())
        response = client.get("/query?query=q")

        assert response.status_code == 200
        events = read_events(response)
        errors = [e for e in events if e["type"] == "error"]
        assert errors, "the browser must be told why nothing happened"
        assert "ollama pull" in errors[0]["hint"]

    def test_error_stream_still_terminates(self, flask_client):
        client, _ = flask_client(normal_script(), embed=FailingEmbedding())
        events = read_events(client.get("/query?query=q"))
        assert events[-1]["type"] == "done_stream"

    def test_chat_failure_yields_an_error_event(self, flask_client):
        client, _ = flask_client([])  # ScriptedChat with nothing to say
        events = read_events(client.get("/query?query=q"))
        assert [e for e in events if e["type"] == "error"]

    def test_empty_query_is_rejected_clearly(self, flask_client):
        client, _ = flask_client(normal_script())
        assert client.get("/query?query=").status_code == 400
        assert client.get("/query?query=%20%20").status_code == 400


class TestIssueTwoNeverTerminates:
    """Both retry branches used to loop without advancing the step counter."""

    def test_endlessly_long_answers_still_terminate(self, flask_client):
        long_step = step("Long", "x" * 900)
        client, _ = flask_client([long_step], repeat_last=True)

        events = read_events(client.get("/query?query=q"))

        assert events[-1]["type"] == "done_stream"
        assert [e for e in events if e["type"] == "final"]
        steps = [e for e in events if e["type"] == "step"]
        assert steps and all(e["truncated"] for e in steps)
        assert all(len(e["content"]) <= 704 for e in steps)

    def test_model_that_always_wants_to_stop_still_terminates(self, flask_client):
        finish = step("Done", "Answering immediately.", "final_answer")
        client, _ = flask_client([finish], repeat_last=True)

        events = read_events(client.get("/query?query=q"))

        assert events[-1]["type"] == "done_stream"
        # It is nudged up to the minimum before being allowed to finish. The last
        # node is the final answer, so it is not also announced as a step.
        final = [e for e in events if e["type"] == "final"][0]
        assert len(final["graph"]["nodes"]) >= 5

    def test_step_count_is_bounded(self, flask_client):
        never_finish = step("Go on", "Still reasoning about the problem.")
        client, _ = flask_client([never_finish], repeat_last=True)

        events = read_events(client.get("/query?query=q"))
        assert len([e for e in events if e["type"] == "step"]) <= 20
        assert events[-1]["type"] == "done_stream"


class TestMalformedModelOutput:
    def test_non_json_answer_becomes_a_visible_step(self, flask_client):
        """It used to become a node literally labelled 'Parsing Error'."""
        client, _ = flask_client(
            ["I refuse to answer in JSON.", *normal_script()],
        )
        events = read_events(client.get("/query?query=q"))
        first = [e for e in events if e["type"] == "step"][0]

        assert "Parsing Error" not in first["title"]
        assert first["content"] == "I refuse to answer in JSON."

    def test_json_wrapped_in_a_code_fence_is_understood(self, flask_client):
        fenced = "```json\n" + step("Fenced", "Content inside a fence.") + "\n```"
        client, _ = flask_client([fenced, *normal_script()])
        events = read_events(client.get("/query?query=q"))
        first = [e for e in events if e["type"] == "step"][0]
        assert first["title"] == "Fenced"

    def test_apostrophes_are_preserved(self, flask_client):
        """The old streamer stripped every ' from the model's text."""
        client, _ = flask_client([step("T", "It doesn't drop the model's apostrophes."),
                                  *normal_script()])
        events = read_events(client.get("/query?query=q"))
        assert "doesn't" in [e for e in events if e["type"] == "step"][0]["content"]


class TestHealthRoute:
    def test_health_reports_a_problem_when_ollama_is_absent(self, flask_client, monkeypatch):
        from mpe_lkg import backends

        monkeypatch.setattr(backends, "list_models", lambda *a, **k: [])
        client, _ = flask_client(normal_script())
        payload = client.get("/health").get_json()

        assert payload["ok"] is False
        assert "ollama serve" in payload["hint"]

    def test_health_names_the_missing_model(self, flask_client, monkeypatch):
        from mpe_lkg import backends

        monkeypatch.setattr(
            backends, "list_models",
            lambda *a, **k: [{"name": "all-minilm:latest", "is_embedding": True, "capabilities": []}],
        )
        client, _ = flask_client(normal_script())
        payload = client.get("/health").get_json()

        assert payload["ok"] is False
        assert "ollama pull" in payload["hint"]
        assert "all-minilm:latest" in payload["models"]


class TestEmbeddingDimensions:
    @pytest.mark.parametrize("dim", [8, 48, 384, 768, 4096])
    def test_any_embedding_size_works(self, flask_client, dim):
        """The old similarity search hardcoded 4096 and broke on anything else."""
        client, _ = flask_client(normal_script(), embed=DeterministicEmbedding(dim=dim))
        events = read_events(client.get("/query?query=q"))

        assert [e for e in events if e["type"] == "final"]
        assert not [e for e in events if e["type"] == "error"]
        done = [e for e in events if e["type"] == "done"][0]
        assert done["embedding"]["dim"] == dim

    def test_events_are_json_serialisable_end_to_end(self, flask_client):
        client, _ = flask_client(normal_script())
        raw = client.get("/query?query=q").get_data(as_text=True)
        for line in raw.splitlines():
            if line.startswith("data: "):
                json.loads(line[6:])
