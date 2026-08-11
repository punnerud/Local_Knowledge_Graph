"""Headless runs: start, poll, take the RDF, kill.

The browser holds one SSE connection for the length of a run, which suits a person
watching and nothing else. These are the properties a script needs, and each one
is a way background work usually goes wrong:

* killing actually stops the work, rather than setting a flag nobody reads
* finishing is observable even when the run raised, or a poller waits forever
* the registry does not grow without bound
"""

from __future__ import annotations

import time

import pytest
from conftest import normal_script, step

from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat
from mpe_lkg.jobs import Job, Registry

rdflib = pytest.importorskip("rdflib", reason="the RDF validator is a dev dependency")


@pytest.fixture
def client(tmp_path):
    import mpe_lkg.app as app_module

    app_module.app.config["DB_PATH"] = str(tmp_path / "jobs.db")
    app_module.app.config["BACKENDS_FACTORY"] = lambda: (
        ScriptedChat(normal_script(5)), DeterministicEmbedding(32))
    app_module.JOBS = app_module.Registry()
    with app_module.app.test_client() as c:
        yield c


def wait_for(client, job_id, states=("done", "error", "cancelled"), timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").get_json()
        if status["state"] in states:
            return status
        time.sleep(0.05)
    raise AssertionError(f"job stayed in {status['state']!r}")


class TestTheJobApi:
    def test_a_run_starts_and_finishes(self, client):
        started = client.post("/jobs", json={"query": "What is the capital of France?"})
        assert started.status_code == 202
        assert started.headers["Location"].endswith(started.get_json()["id"])

        done = wait_for(client, started.get_json()["id"])
        assert done["state"] == "done"
        assert done["answer"]
        assert done["steps"] >= 1

    def test_an_empty_question_is_refused_rather_than_queued(self, client):
        assert client.post("/jobs", json={"query": "   "}).status_code == 400

    def test_an_unknown_job_is_a_404_everywhere(self, client):
        for path in ("/jobs/nope", "/jobs/nope/rdf", "/jobs/nope/stream"):
            assert client.get(path).status_code == 404
        assert client.delete("/jobs/nope").status_code == 404

    def test_listing_shows_what_has_run(self, client):
        client.post("/jobs", json={"query": "one"})
        client.post("/jobs", json={"query": "two"})
        listed = client.get("/jobs").get_json()["jobs"]
        assert {j["question"] for j in listed} == {"one", "two"}

    def test_killing_is_idempotent(self, client):
        job_id = client.post("/jobs", json={"query": "q"}).get_json()["id"]
        wait_for(client, job_id)
        # A finished job killed twice is not an error: a retrying client should
        # not have to distinguish "stopped it" from "it had already stopped".
        assert client.delete(f"/jobs/{job_id}").status_code == 200
        assert client.delete(f"/jobs/{job_id}").status_code == 200


class TestRdfEndpoints:
    def test_the_finished_run_is_valid_turtle(self, client):
        job_id = client.post("/jobs", json={"query": "q"}).get_json()["id"]
        wait_for(client, job_id)
        response = client.get(f"/jobs/{job_id}/rdf")
        assert response.mimetype == "text/turtle"
        graph = rdflib.Graph()
        graph.parse(data=response.get_data(as_text=True), format="turtle")
        assert len(graph) > 5

    def test_ntriples_is_available_and_valid(self, client):
        job_id = client.post("/jobs", json={"query": "q"}).get_json()["id"]
        wait_for(client, job_id)
        response = client.get(f"/jobs/{job_id}/rdf?format=nt")
        assert response.mimetype == "application/n-triples"
        graph = rdflib.Graph()
        graph.parse(data=response.get_data(as_text=True), format="nt")
        assert len(graph) > 5

    def test_the_stream_ends_and_parses(self, client):
        job_id = client.post("/jobs", json={"query": "q"}).get_json()["id"]
        body = client.get(f"/jobs/{job_id}/stream").get_data(as_text=True)
        graph = rdflib.Graph()
        graph.parse(data=body, format="nt")
        assert len(graph) > 5

    def test_the_stream_repeats_nothing(self, client):
        """A triple sent twice is not wrong in RDF, but it is noise on a wire."""
        job_id = client.post("/jobs", json={"query": "q"}).get_json()["id"]
        lines = [x for x in client.get(f"/jobs/{job_id}/stream").get_data(
            as_text=True).splitlines() if x.strip()]
        assert len(lines) == len(set(lines))


class TestCancellation:
    def test_a_kill_stops_the_run_rather_than_only_flagging_it(self, tmp_path):
        """The property that makes DELETE worth having.

        A slow model is the case that matters: a flag read only after the loop
        would cancel nothing, so the check sits between events, which is where a
        reasoning run spends its time.
        """
        import mpe_lkg.app as app_module

        app_module.app.config["DB_PATH"] = str(tmp_path / "slow.db")
        # 0.3s per call, and a script long enough that finishing normally would
        # take far longer than this test allows.
        app_module.app.config["BACKENDS_FACTORY"] = lambda: (
            ScriptedChat([step(f"T{i}", f"Body {i}") for i in range(40)],
                         repeat_last=True, delay=0.3),
            DeterministicEmbedding(32))
        app_module.JOBS = app_module.Registry()

        with app_module.app.test_client() as client:
            job_id = client.post("/jobs", json={"query": "q"}).get_json()["id"]
            time.sleep(0.6)
            client.delete(f"/jobs/{job_id}")
            status = wait_for(client, job_id, timeout=10.0)
            assert status["state"] == "cancelled"
            # It stopped early rather than running the script out.
            assert status["steps"] < 40


class TestTheRegistry:
    def test_finished_jobs_are_dropped_oldest_first(self):
        registry = Registry(max_kept=3)
        for i in range(6):
            job = Job(f"q{i}")
            job.done.set()
            registry.add(job)
        assert len(registry.all()) == 3
        assert [j.question for j in registry.all()] == ["q3", "q4", "q5"]

    def test_a_running_job_is_never_evicted(self):
        """Evicting one would leave a thread writing into an unreachable object."""
        registry = Registry(max_kept=1)
        running = Job("still going")
        registry.add(running)
        for i in range(5):
            finished = Job(f"done{i}")
            finished.done.set()
            registry.add(finished)
        assert registry.get(running.id) is running

    def test_a_run_that_raised_is_finished_rather_than_eternally_running(self):
        def explode():
            yield {"type": "step", "step": 1}
            raise RuntimeError("model fell over")

        job = Job("q")
        job.start(explode())
        assert job.done.wait(timeout=5)
        assert job.state == "error"
        assert "model fell over" in job.error


class TestModes:
    """reason, explore and settle over the same API.

    A settle run is minutes rather than seconds, so the mode is reported back:
    a caller that cannot tell them apart will time out on one of them.
    """

    def test_the_mode_is_recorded_and_returned(self, client):
        started = client.post("/jobs", json={"query": "q", "mode": "reason"})
        assert started.get_json()["mode"] == "reason"

    def test_an_unknown_mode_is_refused_with_the_list(self, client):
        refused = client.post("/jobs", json={"query": "q", "mode": "guess"})
        assert refused.status_code == 400
        assert set(refused.get_json()["modes"]) == {"reason", "explore", "settle"}

    def test_the_default_is_a_single_run(self, client):
        assert client.post("/jobs", json={"query": "q"}).get_json()["mode"] == "reason"
