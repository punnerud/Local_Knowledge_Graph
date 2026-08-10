"""Backends that answer without a model, so the app is testable without one."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Iterator

import numpy as np

from ._shared import BackendError, _clean_for_embedding, _l2_normalise


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
