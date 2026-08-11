"""The Ollama HTTP contract, against a stub server rather than a real Ollama.

These cover the paths a real Ollama on this machine cannot exercise: an old build
that predates /api/embed, a 404 that means "no such model" rather than "no such
route", and a response that returns the wrong number of rows.

The old-endpoint path is the one that matters most. Issue #1 reported a blank page
on a working Ollama; /api/embed only exists from Ollama v0.3.4, so anyone on an
older build got a 404 that the original code turned into an exception before the
event stream opened. That fallback had no test until now.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from mpe_lkg.backends import BackendError, OllamaEmbedding


class StubOllama:
    """A configurable stand-in for the Ollama HTTP API."""

    def __init__(self, *, has_modern_endpoint=True, missing_model=False, short_rows=False, dim=8):
        self.has_modern_endpoint = has_modern_endpoint
        self.missing_model = missing_model
        self.short_rows = short_rows
        self.dim = dim
        self.paths: list[str] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _json(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                stub.paths.append(self.path)
                if self.path == "/api/tags":
                    self._json(200, {"models": [{"name": "stub-embed", "capabilities": ["embedding"]}]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                stub.paths.append(self.path)
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")

                if self.path == "/api/embed":
                    if stub.missing_model:
                        model = request.get("model", "")
                        self._json(404, {"error": f'model "{model}" not found, try pulling it first'})
                        return
                    if not stub.has_modern_endpoint:
                        # Ollama before v0.3.4: the route simply does not exist.
                        self._json(404, {"error": "404 page not found"})
                        return
                    n = len(request.get("input", []))
                    if stub.short_rows:
                        n = max(0, n - 1)
                    self._json(200, {"embeddings": [[0.1] * stub.dim for _ in range(n)]})
                    return

                if self.path == "/api/embeddings":
                    # The legacy endpoint is singular: one prompt, one vector.
                    assert "prompt" in request, "legacy endpoint takes 'prompt', not 'input'"
                    self._json(200, {"embedding": [0.2] * stub.dim})
                    return

                self._json(404, {"error": "not found"})

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class TestOldOllamaWithoutApiEmbed:
    """Ollama before v0.3.4 has no /api/embed, only the singular /api/embeddings."""

    def test_it_falls_back_and_still_returns_vectors(self):
        with StubOllama(has_modern_endpoint=False) as stub:
            backend = OllamaEmbedding("stub-embed", base_url=stub.url)
            vectors = backend.embed(["one", "two", "three"])

        assert vectors.shape == (3, 8)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_it_tries_the_modern_endpoint_first(self):
        with StubOllama(has_modern_endpoint=False) as stub:
            OllamaEmbedding("stub-embed", base_url=stub.url).embed(["one"])
            posts = [p for p in stub.paths if p.startswith("/api/embed")]

        assert posts[0] == "/api/embed"
        assert "/api/embeddings" in posts

    def test_the_fallback_is_reported_in_describe(self):
        """Which endpoint answered is worth knowing when diagnosing a blank page."""
        with StubOllama(has_modern_endpoint=False) as stub:
            backend = OllamaEmbedding("stub-embed", base_url=stub.url)
            backend.embed(["one"])
            assert backend.describe()["endpoint"] == "/api/embeddings"

    def test_a_modern_server_never_touches_the_legacy_route(self):
        with StubOllama(has_modern_endpoint=True) as stub:
            backend = OllamaEmbedding("stub-embed", base_url=stub.url)
            backend.embed(["one", "two"])

        assert "/api/embeddings" not in stub.paths
        assert backend.describe()["endpoint"] == "/api/embed"


class TestTheTwoKindsOf404:
    """A missing route and a missing model both answer 404 and mean opposite things."""

    def test_a_missing_model_names_the_pull_command(self):
        with StubOllama(missing_model=True) as stub:
            backend = OllamaEmbedding("llama3.1:8b", base_url=stub.url)
            with pytest.raises(BackendError) as excinfo:
                backend.embed(["one"])

        assert "ollama pull llama3.1:8b" in excinfo.value.hint
        assert "/api/embeddings" not in stub.paths, "a missing model must not retry the legacy route"


class TestMisalignedResponses:
    def test_a_short_response_is_refused_rather_than_misaligned(self):
        """Silently dropping a row shifts every later vector onto the wrong text."""
        with StubOllama(short_rows=True) as stub:
            backend = OllamaEmbedding("stub-embed", base_url=stub.url)
            with pytest.raises(BackendError, match="returned 2 vectors"):
                backend.embed(["one", "two", "three"])

    def test_an_unreachable_server_says_how_to_start_it(self):
        # A genuinely closed port, so the connection is refused immediately.
        # Constructing an HTTPServer and not serving it does NOT give that: the
        # socket is already bound and listening, so the connect succeeds and the
        # request then blocks until the request timeout -- two minutes per run.
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        backend = OllamaEmbedding("stub-embed", base_url=f"http://127.0.0.1:{port}")
        with pytest.raises(BackendError) as excinfo:
            backend.embed(["one"])
        assert "ollama serve" in excinfo.value.hint


class TestConstrainedDecoding:
    """A schema alone is not enough to get an answer out of some models.

    Measured on qwen3:4b-instruct-2507: a schema-constrained call spent an
    EIGHT-HUNDRED token budget entirely on whitespace and returned nothing, while
    the same prompt with one added line answered in ten tokens. The model wants to
    explain itself, the grammar forbids prose, and whitespace is the one thing the
    grammar still allows -- so it emits that until the budget runs out.

    It looks like a model that cannot follow a schema. It is a model being given
    two contradictory orders, and this removes one of them.
    """

    def test_a_schema_call_also_asks_for_json_only(self):
        from mpe_lkg.backends.ollama import JSON_ONLY, _json_only

        messages = [{"role": "system", "content": "You reason."},
                    {"role": "user", "content": "Pick one."}]
        out = _json_only(messages)
        assert out[-1]["content"].endswith(JSON_ONLY)
        assert out[0] == messages[0], "earlier turns are untouched"
        assert messages[-1]["content"] == "Pick one.", "the caller's list is not mutated"

    def test_the_last_user_turn_is_the_one_marked(self):
        from mpe_lkg.backends.ollama import JSON_ONLY, _json_only

        out = _json_only([{"role": "user", "content": "first"},
                          {"role": "assistant", "content": "..."},
                          {"role": "user", "content": "second"}])
        assert JSON_ONLY not in out[0]["content"]
        assert JSON_ONLY in out[2]["content"]

    def test_it_is_not_said_twice(self):
        from mpe_lkg.backends.ollama import JSON_ONLY, _json_only

        once = _json_only([{"role": "user", "content": "Pick one."}])
        twice = _json_only(once)
        assert twice[-1]["content"].count(JSON_ONLY) == 1

    def test_a_conversation_with_no_user_turn_still_gets_one(self):
        from mpe_lkg.backends.ollama import JSON_ONLY, _json_only

        out = _json_only([{"role": "system", "content": "You reason."}])
        assert out[-1]["role"] == "user"
        assert out[-1]["content"] == JSON_ONLY
