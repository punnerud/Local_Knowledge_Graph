"""The graph as tables: recorded once per run, queryable in SQL, in memory too.

The RDF is the graph as a document; this is the graph as a database. Both draw
from the same job events, so the test that matters most here is the one that
feeds a settle-shaped event stream in and asks the questions a consumer would:
which findings did the run make, how was agreement reached, what did the vote
say. If SQL can answer those, the tables carry the structure.
"""

from __future__ import annotations

import json
import time

import pytest
from conftest import normal_script, step

from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat
from mpe_lkg.graphdb import GraphDB
from mpe_lkg.jobs import Job


def settled_job() -> Job:
    """A job holding the events a settle run produces, without running a model."""
    job = Job("How many minutes are there in a fortnight?")
    job.mode = "settle"
    job.session = "clocks"
    for event in [
        {"type": "round", "round": 1, "of": 3},
        {"type": "finding", "question": "How many days are in a fortnight?",
         "answer": "A fortnight has 14 days."},
        {"type": "finding", "question": "How many minutes are in a day?",
         "answer": "There are 1440 minutes in a day."},
        {"type": "finding", "level": 1, "question": "a sub-run's own finding",
         "answer": "must not be recorded at the top level"},
        {"type": "step", "step": 1, "title": "Convert", "content": "Ask for it."},
        {"type": "calc", "step": 1, "expression": "14*1440", "value": "20,160"},
        {"type": "convert", "step": 1, "request": "1 fortnight to minutes",
         "result": "1 fortnight = 20,160 minute"},
        {"type": "vote", "agree": 2, "disagree": 1, "about": "the total"},
        {"type": "agreed", "by": "vote", "round": 2, "answer": "20160"},
        {"type": "final", "content": "There are 20160 minutes in a fortnight.",
         "graph": {"nodes": [], "edges": [
             {"from": "Step1", "to": "Step2", "value": 0.83}]}},
    ]:
        job.append(event)
    job.answer = "There are 20160 minutes in a fortnight."
    job.done.set()
    return job


class TestRecording:
    def test_a_settle_run_is_answerable_in_sql(self):
        graph = GraphDB(":memory:")
        graph.record(settled_job())

        # Qualified columns: both tables carry `question` and `answer`, and
        # mpedb refuses the ambiguity where sqlite silently picked one. The
        # refusal is the better behaviour.
        findings = graph.query(
            "SELECT findings.question, findings.answer FROM findings"
            " JOIN runs ON findings.run = runs.id"
            " WHERE runs.session = 'clocks' ORDER BY findings.question")
        assert len(findings) == 2
        assert findings[0][1] == "A fortnight has 14 days."

        how = graph.query(
            "SELECT agreement.reached_by, votes.agree, votes.disagree FROM agreement"
            " JOIN votes ON votes.run = agreement.run")
        assert how == [("vote", 2, 1)]

        facts = graph.query("SELECT kind, statement FROM facts ORDER BY kind")
        assert ("calculation", "14*1440 = 20,160") in facts
        assert ("conversion", "1 fortnight = 20,160 minute") in facts

    def test_sub_run_findings_stay_out_of_the_top_level(self):
        graph = GraphDB(":memory:")
        graph.record(settled_job())
        texts = [row[0] for row in graph.query("SELECT question FROM findings")]
        assert "a sub-run's own finding" not in texts

    def test_recording_twice_records_once(self):
        graph = GraphDB(":memory:")
        job = settled_job()
        graph.record(job)
        graph.record(job)
        assert graph.query("SELECT COUNT(*) FROM runs")[0][0] == 1
        assert graph.query("SELECT COUNT(*) FROM findings")[0][0] == 2

    def test_session_stats_count_what_a_session_did(self):
        graph = GraphDB(":memory:")
        graph.record(settled_job())
        stats = graph.session_stats("clocks")
        assert stats == {"runs": 1, "agreed": 1, "exact_facts": 2}
        assert graph.session_stats("elsewhere") == {
            "runs": 0, "agreed": 0, "exact_facts": 0}

    def test_edges_carry_their_similarity(self):
        graph = GraphDB(":memory:")
        graph.record(settled_job())
        assert graph.query("SELECT a, b, sim FROM edges") == [("Step1", "Step2", 0.83)]


class TestThroughTheApi:
    @pytest.fixture
    def client(self, tmp_path):
        import mpe_lkg.app as app_module

        app_module.app.config["DB_PATH"] = str(tmp_path / "app.db")
        app_module.app.config["GRAPH_DB_PATH"] = str(tmp_path / "graph.db")
        app_module.app.config["BACKENDS_FACTORY"] = lambda: (
            ScriptedChat(normal_script(4), repeat_last=True),
            DeterministicEmbedding(24))
        app_module.JOBS = app_module.Registry()
        with app_module.app.test_client() as c:
            yield c, str(tmp_path / "graph.db")

    def test_a_finished_job_lands_in_the_tables(self, client):
        c, graph_path = client
        job_id = c.post("/jobs", json={"query": "What is the capital of France?",
                                       "session": "geo"}).get_json()["id"]
        deadline = time.time() + 20
        while time.time() < deadline:
            if c.get(f"/jobs/{job_id}").get_json()["state"] == "done":
                break
            time.sleep(0.05)

        graph = GraphDB(graph_path)
        runs = graph.query("SELECT id, session, question FROM runs")
        assert (job_id, "geo", "What is the capital of France?") in runs
        assert graph.query("SELECT COUNT(*) FROM steps WHERE run = ?",
                           (job_id,))[0][0] > 0

    def test_the_sessions_listing_carries_the_graphs_numbers(self, client):
        c, _ = client
        job_id = c.post("/jobs", json={"query": "q", "session": "geo"}).get_json()["id"]
        deadline = time.time() + 20
        while time.time() < deadline:
            if c.get(f"/jobs/{job_id}").get_json()["state"] == "done":
                break
            time.sleep(0.05)
        listed = {s["name"]: s for s in c.get("/sessions").get_json()["sessions"]}
        assert listed["geo"]["runs"] == 1
        assert "exact_facts" in listed["geo"]


def test_the_scripted_step_helper_still_matches_the_schema():
    # The settle-shaped stream above is hand-built; this pins that the helper
    # the rest of the suite uses produces steps record() can read.
    parsed = json.loads(step("T", "C"))
    assert {"title", "content", "next_action"} <= set(parsed)
