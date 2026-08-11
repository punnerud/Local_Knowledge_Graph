"""Choosing a model from the page, and picking a sensible one automatically.

The reported first-run experience was: three usable models installed, and the app
insisting on pulling a fourth. Auto-selection is the fix; the picker is so the
choice is visible and changeable without reading the README.
"""

import json

import pytest
from conftest import normal_script, read_events

from mpe_lkg import backends


def fake_models(*names_and_kinds):
    return [{"name": n, "is_embedding": e, "capabilities": []} for n, e in names_and_kinds]


@pytest.fixture
def installed(monkeypatch):
    """Pretend Ollama has exactly this set."""

    def apply(*names_and_kinds):
        models = fake_models(*names_and_kinds)
        # Both names: app.py reaches it through the package facade, while health()
        # and pick_chat_model() resolve it inside the ollama module they live in.
        # Patching one and not the other gives a half-faked world.
        monkeypatch.setattr(backends, "list_models", lambda *a, **k: models)
        monkeypatch.setattr(backends.ollama, "list_models", lambda *a, **k: models)
        return models

    return apply


class TestAutomaticChoice:
    def test_it_uses_an_installed_model_rather_than_a_hardcoded_one(self, installed):
        """The exact reported case: llama3.2:3b present, llama3.1:8b not."""
        installed(("llama3.2:3b", False), ("nomic-embed-text:latest", True))
        assert backends.pick_chat_model() == "llama3.2:3b"

    def test_health_is_ok_when_any_chat_model_is_present(self, installed):
        installed(("llama3.2:3b", False), ("all-minilm:latest", True))
        status = backends.health()
        assert status["ok"] is True
        assert status["chat_model"] == "llama3.2:3b"
        assert status["embedding_model"] == "all-minilm:latest"

    def test_an_explicit_request_wins_over_discovery(self, installed):
        installed(("llama3.2:3b", False), ("mistral:7b", False))
        assert backends.pick_chat_model(requested="mistral:7b") == "mistral:7b"

    def test_preference_breaks_a_tie_deterministically(self, installed):
        installed(("zzz:1b", False), ("llama3.1:8b", False), ("aaa:1b", False))
        assert backends.pick_chat_model() == "llama3.1:8b"

    def test_any_model_beats_none_even_outside_the_preference_list(self, installed):
        installed(("some-obscure-model:latest", False))
        assert backends.pick_chat_model() == "some-obscure-model:latest"

    def test_embedding_models_are_never_offered_as_chat_models(self, installed):
        installed(("nomic-embed-text:latest", True), ("all-minilm:latest", True))
        assert backends.chat_models() == []
        assert backends.health()["ok"] is False

    def test_nothing_installed_still_produces_an_actionable_message(self, installed):
        installed()
        status = backends.health()
        assert status["ok"] is False
        assert "ollama serve" in status["hint"]


class TestModelsRoute:
    def test_it_lists_what_is_installed_and_what_is_selected(self, flask_client, installed):
        installed(("llama3.2:3b", False), ("nomic-embed-text:latest", True))
        client, _ = flask_client(normal_script())

        payload = client.get("/models").get_json()
        assert payload["chat"] == ["llama3.2:3b"]
        assert payload["embedding"] == ["nomic-embed-text:latest"]
        assert payload["selected"]["chat"] == "llama3.2:3b"
        assert payload["suggested"], "a first-run user needs something to pick from"

    def test_selecting_an_installed_model_sticks(self, flask_client, installed):
        installed(("llama3.2:3b", False), ("mistral:7b", False))
        client, _ = flask_client(normal_script())

        client.post("/models", json={"chat": "mistral:7b"})
        assert client.get("/models").get_json()["selected"]["chat"] == "mistral:7b"

    def test_a_model_that_is_not_installed_is_refused(self, flask_client, installed):
        """The value goes straight to the model API, so it is not free text."""
        installed(("llama3.2:3b", False))
        client, _ = flask_client(normal_script())

        response = client.post("/models", json={"chat": "../../etc/passwd"})
        assert response.status_code == 400
        assert "not installed" in response.get_json()["error"]

    def test_an_empty_choice_changes_nothing(self, flask_client, installed):
        installed(("llama3.2:3b", False))
        client, _ = flask_client(normal_script())
        assert client.post("/models", json={"chat": ""}).status_code == 200


class TestPullRoute:
    def test_only_the_offered_models_can_be_pulled(self, flask_client):
        """This route causes a multi-gigabyte download; it is an allowlist."""
        client, _ = flask_client(normal_script())
        response = client.post("/pull", json={"model": "attacker/whatever"})
        assert response.status_code == 400

    def test_a_cross_site_request_is_refused(self, flask_client):
        """A page on the internet can POST to a service on your loopback address."""
        client, _ = flask_client(normal_script())
        response = client.post(
            "/pull",
            json={"model": "llama3.2:3b"},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert response.status_code == 403

    def test_progress_is_streamed_and_ends(self, flask_client, monkeypatch):
        chunks = [
            {"status": "pulling manifest"},
            {"status": "downloading", "completed": 50, "total": 100},
            {"status": "success"},
        ]
        monkeypatch.setattr(backends, "pull_model", lambda *a, **k: iter(chunks))
        client, _ = flask_client(normal_script())

        events = read_events(client.post("/pull", json={"model": "llama3.2:3b"}))
        assert [e["type"] for e in events][-1] == "pull_done"
        assert any(e.get("total") == 100 for e in events)

    def test_a_failed_pull_is_reported_rather_than_hanging(self, flask_client, monkeypatch):
        def boom(*_a, **_k):
            raise backends.BackendError("disk full", hint="free some space")
            yield  # pragma: no cover - makes this a generator

        monkeypatch.setattr(backends, "pull_model", boom)
        client, _ = flask_client(normal_script())

        events = read_events(client.post("/pull", json={"model": "llama3.2:3b"}))
        assert events[0]["type"] == "error"
        assert "disk full" in events[0]["message"]


class TestFavicon:
    def test_the_browser_gets_an_answer(self, flask_client):
        """It was a 404 on every single page load, in everyone's terminal."""
        client, _ = flask_client(normal_script())
        response = client.get("/favicon.ico")
        assert response.status_code == 200
        assert response.mimetype == "image/svg+xml"


class TestSuggestions:
    def test_every_suggested_model_is_described_well_enough_to_choose(self):
        for entry in backends.SUGGESTED:
            assert set(entry) == {"name", "size", "role", "note"}
            assert entry["role"] in ("chat", "embedding")
            assert entry["size"], "a user deciding on a download needs the size"

    def test_the_suggestions_are_json_safe(self):
        json.dumps(backends.SUGGESTED)
