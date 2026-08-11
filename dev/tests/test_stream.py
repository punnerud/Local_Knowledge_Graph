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

    def test_a_model_that_is_done_is_allowed_to_be_done(self, flask_client):
        """The floor that forced five steps is gone, and that was the point of it.

        A reported transcript reached its answer at step 4 and was pushed on with
        "you have given 4 of 5 steps", producing three more steps that added
        nothing. Length is decided by whether anything new is still arriving.
        """
        finish = step("Done", "Answering immediately.", "final_answer")
        client, _ = flask_client([finish], repeat_last=True)

        events = read_events(client.get("/query?query=q"))

        assert events[-1]["type"] == "done_stream"
        final = [e for e in events if e["type"] == "final"][0]
        assert len(final["graph"]["nodes"]) <= 3, "a one-step answer must not be padded to five"

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
        monkeypatch.setattr(backends.ollama, "list_models", lambda *a, **k: [])
        client, _ = flask_client(normal_script())
        payload = client.get("/health").get_json()

        assert payload["ok"] is False
        assert "ollama serve" in payload["hint"]

    def test_health_names_the_missing_model(self, flask_client, monkeypatch):
        from mpe_lkg import backends

        fake = [{"name": "all-minilm:latest", "is_embedding": True, "capabilities": []}]
        monkeypatch.setattr(backends, "list_models", lambda *a, **k: fake)
        monkeypatch.setattr(backends.ollama, "list_models", lambda *a, **k: fake)
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


class TestAnswerSynthesis:
    """The answer is written from the graph, not lifted from whichever step was last.

    Measured before this existed: the answer to "What is the capital of France?" did
    not contain the word Paris. It was a footnote about regional capitals, because a
    prompt that rewards exploring alternatives naturally ends on a caveat.
    """

    def test_the_answer_comes_from_the_synthesis_call(self, flask_client):
        client, chat = flask_client(normal_script())
        events = read_events(client.get("/query?query=q"))

        final = [e for e in events if e["type"] == "final"][0]
        assert final["content"] == "The capital of France is Paris."
        assert "Weighing the evidence" not in final["content"], "that was the last step"

    def test_the_synthesis_call_is_given_the_question_and_a_thread(self, flask_client):
        client, chat = flask_client(normal_script())
        read_events(client.get("/query?query=What+is+the+capital+of+France"))

        last_call = chat.calls[-1][0]["content"]
        assert "What is the capital of France" in last_call
        assert "1." in last_call, "the thread is handed over as a numbered list"

    def test_a_failed_synthesis_falls_back_to_the_last_step(self, flask_client):
        """Losing the whole run because one extra call failed would be a bad trade."""
        script = normal_script()[:-1]          # no answer for the synthesis call
        client, _ = flask_client(script)
        events = read_events(client.get("/query?query=q"))

        final = [e for e in events if e["type"] == "final"][0]
        assert final["content"] == "Weighing the evidence gathered so far."
        assert not [e for e in events if e["type"] == "error"]

    def test_the_fallback_does_not_duplicate_the_node(self, flask_client):
        client, _ = flask_client(normal_script()[:-1])
        events = read_events(client.get("/query?query=q"))
        final = [e for e in events if e["type"] == "final"][0]

        labels = [n["label"] for n in final["graph"]["nodes"]]
        assert sum(1 for x in labels if x.startswith("Final Answer")) == 1
        assert all(e["value"] < 0.999 for e in final["graph"]["edges"])

    def test_synthesis_can_be_turned_off(self, flask_client, embedder):
        """The A/B arm the eval measures against."""
        import mpe_lkg.app as app_module
        from mpe_lkg.backends import ScriptedChat

        chat = ScriptedChat(normal_script()[:-1])
        app_module.app.config["BACKENDS_FACTORY"] = lambda: (chat, embedder)
        events = read_events(app_module.app.test_client().get("/query?query=q"))
        final = [e for e in events if e["type"] == "final"][0]
        assert final["content"] == "Weighing the evidence gathered so far."


class TestTimeBudget:
    def test_a_run_stops_when_its_budget_is_spent(self, flask_client):
        """A step cap stops a run that is still getting somewhere; a clock does not."""
        from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat
        from mpe_lkg.reasoning import reason

        never_finish = step("Go on", "Still reasoning about the problem.")
        events = list(reason(
            "q",
            chat=ScriptedChat([never_finish], repeat_last=True, delay=0.05),
            embedder=DeterministicEmbedding(32),
            time_budget=0.2,
            synthesise=False,
        ))
        steps = [e for e in events if e["type"] == "step"]
        assert 0 < len(steps) < 20, "the budget, not the step cap, ended this"
        assert [e for e in events if e["type"] == "final"], "it still answers with what it has"


class TestNoveltyDrivenLength:
    """Length is decided by whether anything new is arriving, not by a constant."""

    def test_a_model_stuck_on_one_move_is_stopped(self, flask_client):
        repeat = step("Alternative Answer Exploration", "Considering alternatives once more.")
        client, _ = flask_client([repeat], repeat_last=True)

        events = read_events(client.get("/query?query=q"))
        repeats = [e for e in events if e["type"] == "repeat"]

        assert repeats, "the second identical move must be refused"
        assert len([e for e in events if e["type"] == "step"]) <= 2
        assert events[-1]["type"] == "done_stream"

    def test_a_refused_step_says_what_it_repeated(self, flask_client):
        repeat = step("Alternative Answer Exploration", "Considering alternatives once more.")
        client, _ = flask_client([repeat], repeat_last=True)
        events = read_events(client.get("/query?query=q"))

        first = [e for e in events if e["type"] == "repeat"][0]
        assert "step 1" in first["reason"]
        assert first["attempt"] == 1

    def test_the_model_is_told_what_it_already_covered(self, flask_client):
        repeat = step("Alternative Answer Exploration", "Considering alternatives once more.")
        client, chat = flask_client([repeat], repeat_last=True)
        read_events(client.get("/query?query=q"))

        redirect = [m for call in chat.calls for m in call if "already covered" in m.get("content", "")]
        assert redirect, "a coverage map, not just a prohibition"
        assert "somewhere none of those go" in redirect[0]["content"]

    def test_distinct_steps_are_never_refused(self, flask_client):
        """The guard that matters: a healthy run must pass through untouched."""
        client, _ = flask_client(normal_script())
        events = read_events(client.get("/query?query=q"))

        assert not [e for e in events if e["type"] == "repeat"]
        assert len([e for e in events if e["type"] == "step"]) == 5

    def test_how_often_a_retry_helped_is_reported(self, flask_client):
        """Whether asking again works is a claim, so it is counted rather than assumed."""
        repeat = step("Alternative Answer Exploration", "Considering alternatives once more.")
        client, _ = flask_client([repeat], repeat_last=True)
        done = [e for e in read_events(client.get("/query?query=q")) if e["type"] == "done"][0]

        assert done["novelty_retries"] >= 1
        assert done["novelty_retries_that_helped"] == 0, "this model never varies, by construction"

    def test_detection_can_be_turned_off(self, flask_client):
        from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat
        from mpe_lkg.reasoning import reason

        repeat = step("Same Move", "Considering alternatives once more.")
        events = list(reason("q", chat=ScriptedChat([repeat], repeat_last=True),
                             embedder=DeterministicEmbedding(32),
                             detect_repeats=False, synthesise=False))
        assert not [e for e in events if e["type"] == "repeat"]


class TestArithmeticGate:
    """Sums the record can evaluate exactly are not a matter of opinion.

    Measured motivation: pushing runs to eight steps lifted hard questions from
    50 % to 62 % and dropped multi-step arithmetic from 94 % to 67 %. The failures
    were "10080 minutes in a fortnight" -- that is a week -- and "6.00 change" from
    a 20 note on 13.50 of goods. Neither is a reasoning failure.
    """

    def test_a_wrong_sum_is_caught_and_redone(self, flask_client):
        wrong = step("Compute", "A fortnight is 14 * 24 * 60 = 10080 minutes.")
        right = step("Compute", "A fortnight is 14 * 24 * 60 = 20160 minutes.", "final_answer")
        client, _ = flask_client([wrong, right, "20160 minutes."])

        events = read_events(client.get("/query?query=q"))
        caught = [e for e in events if e["type"] == "arithmetic"]

        assert caught, "the sum is wrong and the record can prove it"
        assert "20160" in caught[0]["errors"][0]
        assert not [e for e in events if e["type"] == "step" and "10080" in e["content"]]

    def test_a_correct_sum_passes_untouched(self, flask_client):
        client, _ = flask_client([step("Compute", "14 * 24 * 60 = 20160 minutes.", "final_answer"),
                                  "20160 minutes."])
        events = read_events(client.get("/query?query=q"))
        assert not [e for e in events if e["type"] == "arithmetic"]

    def test_rounding_is_not_treated_as_an_error(self, flask_client):
        """28.27 for 28.2743 is correct rounding, and flagging it would be wrong."""
        client, _ = flask_client([step("Area", "The area is 3.14159 * 9 = 28.27.", "final_answer"),
                                  "About 28.27."])
        events = read_events(client.get("/query?query=q"))
        assert not [e for e in events if e["type"] == "arithmetic"]

    def test_a_definition_is_not_a_sum(self, flask_client):
        """x = 5 is a definition. Checking it would invent an error."""
        client, _ = flask_client([step("Set up", "Let x = 5 and y = 12.", "final_answer"),
                                  "x is 5."])
        events = read_events(client.get("/query?query=q"))
        assert not [e for e in events if e["type"] == "arithmetic"]

    def test_a_model_that_will_not_correct_itself_still_terminates(self, flask_client):
        wrong = step("Compute", "14 * 24 * 60 = 10080 minutes.")
        client, _ = flask_client([wrong], repeat_last=True)

        events = read_events(client.get("/query?query=q"))
        assert events[-1]["type"] == "done_stream"
        assert len([e for e in events if e["type"] == "arithmetic"]) <= 3

    def test_how_many_sums_were_checked_is_reported(self, flask_client):
        client, _ = flask_client([step("Compute", "2 + 2 = 5.", "final_answer"),
                                 step("Compute", "2 + 2 = 4.", "final_answer"), "Four."])
        done = [e for e in read_events(client.get("/query?query=q")) if e["type"] == "done"][0]
        assert done["sums_checked"] >= 1
        assert done["sums_corrected"] >= 1

    def test_the_gate_can_be_turned_off(self, flask_client):
        from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat
        from mpe_lkg.reasoning import reason

        wrong = step("Compute", "14 * 24 * 60 = 10080 minutes.", "final_answer")
        events = list(reason("q", chat=ScriptedChat([wrong], repeat_last=True),
                             embedder=DeterministicEmbedding(32),
                             check_arithmetic=False, synthesise=False))
        assert not [e for e in events if e["type"] == "arithmetic"]
