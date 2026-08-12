"""Sessions: global inside, isolated outside, hints strictly opt-in.

The contract, in the user's words: sessions are "global internally, isolated
otherwise", and hints about related material in other sessions can be asked for
without affecting the session one is in. Every test here defends one clause of
that -- most of all that a hint can never leak into a session's own memory,
because that leak would be invisible until it changed an answer.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import normal_script

from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat
from mpe_lkg.store import EmbeddingStore


def vec(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestTheStore:
    def test_a_row_in_one_session_is_invisible_to_another(self, tmp_path):
        store = EmbeddingStore(str(tmp_path / "s.db"))
        shared = vec(1)
        store.add("belongs to A", shared, session="A")
        found = store.find_similar(shared, session="B")
        assert found == [], "isolation is the whole feature"

    def test_the_default_session_is_a_session_like_any_other(self, tmp_path):
        store = EmbeddingStore(str(tmp_path / "s.db"))
        store.add("unnamed", vec(1))
        assert store.find_similar(vec(1), session="")
        assert store.find_similar(vec(1), session="A") == []

    def test_hints_come_only_from_other_sessions_and_say_which(self, tmp_path):
        store = EmbeddingStore(str(tmp_path / "s.db"))
        store.add("mine", vec(1), session="A")
        store.add("theirs", vec(1), session="B")
        store.add("also theirs", vec(1), session="C")
        hints = store.find_hints(vec(1), exclude_session="A")
        texts = {h["text"] for h in hints}
        assert "mine" not in texts, "a session must never be hinted at itself"
        assert {h["session"] for h in hints} <= {"B", "C"}
        assert all("session" in h for h in hints), "unlabelled hints are unusable"

    def test_hints_do_not_write(self, tmp_path):
        store = EmbeddingStore(str(tmp_path / "s.db"))
        store.add("theirs", vec(1), session="B")
        before = store.count()
        store.find_hints(vec(1), exclude_session="A")
        assert store.count() == before

    def test_the_default_session_attribute_routes_untagged_writes(self, tmp_path):
        """How the reasoning loop's own step writes land in the right session
        without threading a parameter through reason(), explore() and settle()."""
        store = EmbeddingStore(str(tmp_path / "s.db"))
        store.default_session = "A"
        store.add("a step", vec(1))
        assert store.find_similar(vec(1), session="A")
        assert store.find_similar(vec(1), session="B") == []

    def test_forgetting_one_session_leaves_the_others(self, tmp_path):
        store = EmbeddingStore(str(tmp_path / "s.db"))
        store.add("a", vec(1), session="A")
        store.add("b", vec(2), session="B")
        assert store.forget("A") == 1
        assert store.find_similar(vec(1), session="A") == []
        assert store.find_similar(vec(2), session="B")

    def test_sessions_lists_names_and_sizes(self, tmp_path):
        store = EmbeddingStore(str(tmp_path / "s.db"))
        store.add("q", vec(1), session="A", is_question=True)
        store.add("step", vec(2), session="A")
        store.add("other", vec(3), session="B")
        listed = {s["name"]: s for s in store.sessions()}
        assert listed["A"]["rows"] == 2
        assert listed["A"]["questions"] == 1
        assert listed["B"]["rows"] == 1

    def test_an_old_database_without_the_column_migrates(self, tmp_path):
        import sqlite3

        path = str(tmp_path / "old.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, text TEXT NOT NULL,"
                     " embedding BLOB NOT NULL, is_question INTEGER NOT NULL DEFAULT 0,"
                     " dim INTEGER NOT NULL, model TEXT NOT NULL DEFAULT '')")
        v = vec(1)
        conn.execute("INSERT INTO embeddings (text, embedding, dim, model) VALUES (?, ?, ?, '')",
                     ("pre-sessions row", v.tobytes(), v.size))
        conn.commit()
        conn.close()

        store = EmbeddingStore(path)
        # Old rows land in the default session, where an unnamed caller finds them.
        assert store.find_similar(v, session="")
        assert store.find_similar(v, session="A") == []


@pytest.fixture
def client(tmp_path):
    import mpe_lkg.app as app_module

    app_module.app.config["DB_PATH"] = str(tmp_path / "app.db")
    # Without this the sessions endpoint records into the repo root -- the
    # stray-file mistake this project has already committed once.
    app_module.app.config["GRAPH_DB_PATH"] = str(tmp_path / "graph.db")
    app_module.app.config["BACKENDS_FACTORY"] = lambda: (
        ScriptedChat(normal_script(4), repeat_last=True), DeterministicEmbedding(24))
    app_module.JOBS = app_module.Registry()
    with app_module.app.test_client() as c:
        yield c


class TestTheApi:
    def _run(self, client, query, session="", hints=False):
        url = f"/query?query={query}&session={session}" + ("&hints=1" if hints else "")
        body = client.get(url).get_data(as_text=True)
        import json

        return [json.loads(line[6:]) for line in body.splitlines()
                if line.startswith("data: ")]

    def test_two_sessions_do_not_see_each_other(self, client):
        """B's related list may contain B's OWN steps -- it always has -- but
        nothing of A's.

        Two earlier versions of this test were wrong before the code was:
        asserting an empty list failed on B's own step rows (a session is
        global INTERNALLY, so its own steps belong in it), and probing for
        "Paris" failed because the scripted model answers "Paris" to
        everything, so the word exists legitimately in B's own answer row.
        The probe word has to be one that exists ONLY in A's rows.
        """
        self._run(client, "Tell me about the metro system", session="A")
        events = self._run(client, "Describe volcanic rock formation", session="B")
        similar = next(e for e in events if e["type"] == "similar")
        assert not any("metro" in item["text"] for item in similar["items"]), \
            "B must not see A's memory"

    def test_a_repeat_in_the_same_session_is_remembered(self, client):
        self._run(client, "What is the capital of France?", session="A")
        events = self._run(client, "What is the capital of France?", session="A")
        similar = next(e for e in events if e["type"] == "similar")
        assert similar["items"], "the same session must remember"

    def test_hints_arrive_only_when_asked_for(self, client):
        self._run(client, "What is the capital of France?", session="A")
        without = self._run(client, "What is the capital of France?", session="B")
        assert not any(e["type"] == "hints" for e in without)
        with_hints = self._run(client, "What is the capital of France?",
                               session="B", hints=True)
        hint_event = next(e for e in with_hints if e["type"] == "hints")
        assert hint_event["items"]
        assert all(item["session"] == "A" for item in hint_event["items"])

    def test_sessions_endpoint_lists_and_forgets(self, client):
        self._run(client, "First question here", session="A")
        self._run(client, "Second question there", session="B")
        listed = client.get("/sessions").get_json()["sessions"]
        names = {s["name"] for s in listed}
        assert {"A", "B"} <= names
        gone = client.delete("/sessions/A").get_json()
        assert gone["rows"] > 0
        names_after = {s["name"] for s in client.get("/sessions").get_json()["sessions"]}
        assert "A" not in names_after
        assert "B" in names_after

    def test_a_job_carries_its_session(self, client):
        started = client.post("/jobs", json={"query": "q", "session": "mine"})
        assert started.get_json()["session"] == "mine"
