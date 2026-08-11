"""Chat and embeddings from a local Ollama, plus what it has installed."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import numpy as np
import requests

from ._shared import (
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    REQUEST_TIMEOUT,
    BackendError,
    _clean_for_embedding,
    _l2_normalise,
)


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


JSON_ONLY = "Reply with JSON only, no explanation."


def _json_only(messages: list[dict]) -> list[dict]:
    """The same conversation, with the last user turn asking for JSON alone."""
    out = [dict(m) for m in messages]
    for message in reversed(out):
        if message.get("role") == "user":
            if JSON_ONLY not in message.get("content", ""):
                message["content"] = f"{message['content']}\n{JSON_ONLY}"
            return out
    out.append({"role": "user", "content": JSON_ONLY})
    return out


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
        # What this backend has actually cost, as Ollama counted it.
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def describe(self) -> dict:
        return {
            "kind": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
        }

    def reset_usage(self) -> None:
        self.prompt_tokens = self.completion_tokens = self.calls = 0

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
            # A constrained model still wants to explain itself, and the grammar
            # forbids prose -- so it emits whitespace, which the grammar DOES
            # allow, until the token budget runs out. Measured on
            # qwen3:4b-instruct-2507: an eight-hundred-token budget spent
            # entirely on whitespace, returning nothing, while the same prompt
            # with this line answered in ten. It looks like a model that cannot
            # follow a schema and is a model being given contradictory orders.
            payload["messages"] = _json_only(messages)

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
            # Ollama reports what it actually tokenised in the final chunk. Counting
            # that beats dividing characters by four: "the context halved" is a claim
            # worth making exactly rather than approximately.
            if chunk.get("done"):
                self.prompt_tokens += int(chunk.get("prompt_eval_count", 0))
                self.completion_tokens += int(chunk.get("eval_count", 0))
                self.calls += 1
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                produced = True
                yield piece
        if not produced:
            raise BackendError(
                f"Ollama returned an empty response from model '{self.model}'.",
                hint="The model may have been evicted mid-request; try again.",
            )


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


# Only used to break a tie when several chat models are installed. Any installed
# model beats a missing one, so this never causes a "not found".
CHAT_PREFERENCE = ("llama3.1:8b", "llama3.2:3b", "qwen3", "mistral", "gemma3", "phi4")

# Offered in the UI when Ollama has nothing usable. Sizes are what `ollama pull`
# actually downloads, so the page can say what it is about to cost.
SUGGESTED = [
    {"name": "llama3.2:3b", "size": "2.0 GB", "role": "chat", "note": "good default"},
    {"name": "llama3.1:8b", "size": "4.7 GB", "role": "chat", "note": "stronger, slower"},
    {"name": "gemma3:1b", "size": "0.8 GB", "role": "chat", "note": "smallest usable"},
    {"name": "nomic-embed-text", "size": "0.3 GB", "role": "embedding", "note": "recommended"},
    {"name": "all-minilm", "size": "45 MB", "role": "embedding", "note": "tiny"},
]


def chat_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """Installed models that can hold a conversation."""
    return [m["name"] for m in list_models(base_url) if not m["is_embedding"]]


def embedding_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    return [m["name"] for m in list_models(base_url) if m["is_embedding"]]


def pick_chat_model(base_url: str = DEFAULT_BASE_URL, requested: str = "") -> str:
    """Choose a chat model that is actually installed.

    The embedding model has always been discovered rather than assumed; the chat
    model was hardcoded, so a user with a perfectly good model installed under a
    different name was told to pull one they did not need. An explicit request wins,
    and is returned even when absent so the caller can report it honestly.
    """
    if requested:
        return requested

    available = chat_models(base_url)
    if not available:
        return DEFAULT_CHAT_MODEL or CHAT_PREFERENCE[0]

    by_base = {name.split(":")[0]: name for name in available}
    for preferred in CHAT_PREFERENCE:
        if preferred in available:
            return preferred
        if preferred.split(":")[0] in by_base:
            return by_base[preferred.split(":")[0]]
    return sorted(available)[0]


def pull_model(name: str, base_url: str = DEFAULT_BASE_URL):
    """Stream ``ollama pull`` progress as dicts. Yields until the download ends."""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/pull",
            json={"model": name, "stream": True},
            stream=True,
            timeout=(10, 3600),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BackendError(f"Could not start the download of '{name}'.", hint=str(exc)) from exc

    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if chunk.get("error"):
            raise BackendError(f"Ollama could not pull '{name}': {chunk['error']}")
        yield chunk


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
    chat = pick_chat_model(base_url, DEFAULT_CHAT_MODEL)
    chat_ok = chat in names or any(n.split(":")[0] == chat.split(":")[0] for n in names)
    if not chat_ok:
        return {
            "ok": False,
            "base_url": base_url,
            "models": sorted(names),
            "problem": f"Ollama is running but does not have the chat model '{chat}'.",
            "hint": f"Install it with:  ollama pull {chat}\n"
            f"Or pick one you already have in the page, or set LKG_CHAT_MODEL to "
            f"one of: {', '.join(sorted(names))}",
            "chat_model": chat,
            "embedding_model": "",
        }

    return {
        "ok": True,
        "base_url": base_url,
        "models": sorted(names),
        "problem": "",
        "hint": "",
        "chat_model": chat,
        "embedding_model": next(iter(embedding_models(base_url)), ""),
    }
