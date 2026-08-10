"""Pluggable model backends.

The rest of the application never talks to Ollama directly. It asks for an
``EmbeddingBackend`` and a ``ChatBackend`` and uses those, which is what makes the
reasoning loop and the graph testable without a model running anywhere.

Two rules are load-bearing here and are easy to lose in a refactor:

* An embedding backend reports its own dimension. Nothing downstream may assume a
  size. The original code hardcoded 4096 in one place and passed it as a parameter
  in another, so swapping the embedding model broke similarity search silently.
* A backend never swallows an error. A failure is raised as ``BackendError`` with a
  message meant for a human, because the symptom users reported was a blank page
  with nothing in the terminal.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator
from typing import Protocol

import numpy as np
import requests

DEFAULT_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_CHAT_MODEL = os.environ.get("LKG_CHAT_MODEL", "llama3.1:8b")
# Empty means "look at what Ollama actually has and pick something sensible".
DEFAULT_EMBED_MODEL = os.environ.get("LKG_EMBED_MODEL", "")
REQUEST_TIMEOUT = float(os.environ.get("LKG_TIMEOUT", "120"))

# The reasoning loop asks for exactly these three keys. Handing Ollama the schema
# means the model cannot answer with prose that fails to parse, which is where the
# "Step 5: Parsing Error" nodes in the project's own screenshot came from.
STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "content": {"type": "string"},
        "next_action": {"type": "string", "enum": ["continue", "final_answer"]},
    },
    "required": ["title", "content", "next_action"],
}


class BackendError(RuntimeError):
    """A model backend could not answer, with a message worth showing a user."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def user_message(self) -> str:
        return f"{self}\n{self.hint}".strip()


class EmbeddingBackend(Protocol):
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of L2-normalised row vectors."""

    @property
    def dim(self) -> int: ...

    def describe(self) -> dict: ...


class ChatBackend(Protocol):
    def stream(self, messages: list[dict], max_tokens: int, *, schema: dict | None = None) -> Iterator[str]:
        """Yield response text as it arrives."""

    def describe(self) -> dict: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector stays zero rather than becoming NaN; cosine against it is 0.
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


def _clean_for_embedding(text: str) -> str:
    """Collapse whitespace so one record can never become two.

    Every text-in/vector-out endpoint that is line-oriented treats a newline as a
    record separator. A single embedded newline shifts every subsequent vector onto
    the wrong document, and the result looks like a plausible graph rather than an
    error. Collapsing here costs nothing and removes the whole class of bug.
    """
    return " ".join(text.split()) or " "


class OllamaEmbedding:
    """Embeddings from Ollama, with the model chosen automatically if not given."""

    def __init__(
        self,
        model: str = "",
        *,
        base_url: str = DEFAULT_BASE_URL,
        fallback_model: str = DEFAULT_CHAT_MODEL,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._dim: int | None = None
        self._endpoint = "/api/embed"
        self.model = model or self._pick_model(fallback_model)
        self.auto_selected = not model

    def _pick_model(self, fallback: str) -> str:
        """Prefer a real embedding model over a chat model, if one is installed."""
        available = list_models(self.base_url, session=self._session)
        embedders = [m["name"] for m in available if m.get("is_embedding")]
        if embedders:
            return embedders[0]
        names = {m["name"] for m in available}
        if fallback in names:
            return fallback
        # Ollama tags are "name:tag"; accept a bare name the user typed.
        for name in names:
            if name.split(":")[0] == fallback.split(":")[0]:
                return name
        return fallback

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[1])
        return self._dim

    def describe(self) -> dict:
        return {
            "kind": "ollama",
            "model": self.model,
            "dim": self.dim,
            "base_url": self.base_url,
            "auto_selected": self.auto_selected,
            "endpoint": self._endpoint,
        }

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        items = [_clean_for_embedding(t) for t in texts]
        if not items:
            return np.zeros((0, self._dim or 0), dtype=np.float32)

        vectors = self._embed_batch(items)
        if len(vectors) != len(items):
            # A silent success that returns the wrong number of rows is worse than a
            # crash: every vector after the gap belongs to the wrong text.
            raise BackendError(
                f"Embedding model '{self.model}' returned {len(vectors)} vectors "
                f"for {len(items)} inputs.",
                hint="This misaligns every embedding after the gap. Retry, or pick "
                "another model with LKG_EMBED_MODEL.",
            )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise BackendError(f"Embedding model '{self.model}' returned a malformed response.")
        self._dim = int(matrix.shape[1])
        return _l2_normalise(matrix)

    def _embed_batch(self, items: list[str]) -> list[list[float]]:
        try:
            response = self._session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": items},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise BackendError(
                f"Could not reach Ollama at {self.base_url}.",
                hint="Start it with 'ollama serve', or set OLLAMA_URL if it runs elsewhere.",
            ) from exc

        if response.status_code == 404:
            body = response.text
            if "not found" in body and self.model in body:
                raise BackendError(
                    f"Ollama does not have the embedding model '{self.model}'.",
                    hint=f"Install it with:  ollama pull {self.model}",
                )
            # Ollama before v0.3.4 has no /api/embed, only the singular endpoint.
            self._endpoint = "/api/embeddings"
            return [self._embed_one_legacy(text) for text in items]

        if response.status_code != 200:
            raise BackendError(
                f"Ollama returned HTTP {response.status_code} for an embedding request.",
                hint=response.text[:400],
            )

        payload = response.json()
        if payload.get("embeddings"):
            return payload["embeddings"]
        if payload.get("embedding"):
            return [payload["embedding"]]
        raise BackendError(
            f"Ollama returned no embedding for model '{self.model}'.",
            hint=str(payload)[:400],
        )

    def _embed_one_legacy(self, text: str) -> list[float]:
        response = self._session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            raise BackendError(
                f"Ollama returned HTTP {response.status_code} from /api/embeddings.",
                hint=response.text[:400],
            )
        payload = response.json()
        if not payload.get("embedding"):
            raise BackendError(f"Ollama returned no embedding for model '{self.model}'.")
        return payload["embedding"]


class OllamaChat:
    """Streaming chat completions from Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.2,
        session: requests.Session | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self._session = session or requests.Session()

    def describe(self) -> dict:
        return {"kind": "ollama", "model": self.model, "base_url": self.base_url}

    def stream(self, messages: list[dict], max_tokens: int, *, schema: dict | None = None) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            # These belong under "options". The original code sent them at the top
            # level, where Ollama ignores them -- so neither the token limit nor the
            # temperature had any effect at all.
            "options": {"num_predict": max_tokens, "temperature": self.temperature},
        }
        if schema is not None:
            payload["format"] = schema

        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise BackendError(
                f"Could not reach Ollama at {self.base_url}.",
                hint="Start it with 'ollama serve', or set OLLAMA_URL if it runs elsewhere.",
            ) from exc

        if response.status_code == 404 and self.model in response.text:
            raise BackendError(
                f"Ollama does not have the model '{self.model}'.",
                hint=f"Install it with:  ollama pull {self.model}",
            )
        if response.status_code != 200:
            raise BackendError(
                f"Ollama returned HTTP {response.status_code} for a chat request.",
                hint=response.text[:400],
            )

        produced = False
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise BackendError(f"Ollama reported an error: {chunk['error']}")
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                produced = True
                yield piece
        if not produced:
            raise BackendError(
                f"Ollama returned an empty response from model '{self.model}'.",
                hint="The model may have been evicted mid-request; try again.",
            )


class ScriptedChat:
    """A chat backend that replays canned responses. For tests."""

    def __init__(
        self,
        responses: list[str],
        *,
        chunk_size: int = 24,
        repeat_last: bool = False,
        delay: float = 0.0,
    ) -> None:
        self._responses = list(responses)
        self._chunk_size = chunk_size
        self._repeat_last = repeat_last
        # Lets a test observe the in-flight state of the UI, which a backend that
        # answers instantly makes unobservable.
        self._delay = delay
        self.calls: list[list[dict]] = []

    def describe(self) -> dict:
        return {"kind": "scripted", "model": "scripted", "remaining": len(self._responses)}

    def stream(self, messages: list[dict], max_tokens: int, *, schema: dict | None = None) -> Iterator[str]:
        self.calls.append(list(messages))
        if self._responses:
            text = self._responses[0] if (self._repeat_last and len(self._responses) == 1) else self._responses.pop(0)
        elif self._repeat_last:
            text = ""
        else:
            raise BackendError("ScriptedChat ran out of scripted responses.")
        if self._delay:
            time.sleep(self._delay)
        for start in range(0, len(text), self._chunk_size):
            yield text[start : start + self._chunk_size]


class DeterministicEmbedding:
    """Reproducible pseudo-embeddings derived from the text. For tests.

    Identical text yields an identical vector and similar text does not yield a
    similar vector, which is exactly what a test wants: total control, no network.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def describe(self) -> dict:
        return {"kind": "deterministic", "model": f"hash-{self._dim}", "dim": self._dim}

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(_clean_for_embedding(text).encode()).digest()[:8], "big")
            rows.append(np.random.default_rng(seed).standard_normal(self._dim))
        if not rows:
            return np.zeros((0, self._dim), dtype=np.float32)
        return _l2_normalise(np.asarray(rows, dtype=np.float32))


def list_models(base_url: str = DEFAULT_BASE_URL, *, session: requests.Session | None = None) -> list[dict]:
    """Return the installed Ollama models, or an empty list if it is unreachable."""
    session = session or requests
    try:
        response = session.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    models = []
    for entry in payload.get("models", []):
        name = entry.get("name", "")
        caps = entry.get("capabilities") or []
        # Older Ollama builds do not report capabilities, so fall back to the naming
        # convention every published embedding model follows.
        is_embedding = "embedding" in caps or "embed" in name.lower() or "minilm" in name.lower()
        models.append({"name": name, "is_embedding": is_embedding, "capabilities": caps})
    return models


def health(base_url: str = DEFAULT_BASE_URL) -> dict:
    """Everything the UI needs to explain why nothing is happening."""
    models = list_models(base_url)
    if not models:
        return {
            "ok": False,
            "base_url": base_url,
            "models": [],
            "problem": f"No answer from Ollama at {base_url}.",
            "hint": "Start it with 'ollama serve'. If it runs on another host or port, "
            "set OLLAMA_URL before starting this app.",
        }

    names = {m["name"] for m in models}
    chat_ok = DEFAULT_CHAT_MODEL in names or any(
        n.split(":")[0] == DEFAULT_CHAT_MODEL.split(":")[0] for n in names
    )
    if not chat_ok:
        return {
            "ok": False,
            "base_url": base_url,
            "models": sorted(names),
            "problem": f"Ollama is running but does not have the chat model '{DEFAULT_CHAT_MODEL}'.",
            "hint": f"Install it with:  ollama pull {DEFAULT_CHAT_MODEL}\n"
            f"Or point the app at a model you already have by setting "
            f"LKG_CHAT_MODEL to one of: {', '.join(sorted(names))}",
        }

    return {"ok": True, "base_url": base_url, "models": sorted(names), "problem": "", "hint": ""}
