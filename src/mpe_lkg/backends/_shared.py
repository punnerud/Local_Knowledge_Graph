"""Configuration, protocols and the helpers every backend needs."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Protocol

import numpy as np

DEFAULT_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Empty means "use whatever chat model Ollama actually has". Setting LKG_CHAT_MODEL
# is an override, not a default: hardcoding a name here is what told a user with a
# perfectly good model installed to go and pull one they did not need.
DEFAULT_CHAT_MODEL = os.environ.get("LKG_CHAT_MODEL", "")
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
        # REQUIRED, with "" as the legal way to say "no sum here". It was optional
        # first and the model never once filled it in -- structured output simply
        # omits a field it is not obliged to produce. Required, it has to answer,
        # and an empty string costs nothing.
        #
        # The point: a model writes arithmetic as prose -- "14 days at 1440
        # minutes each, so 20160" -- which nothing can check. Asked for the
        # expression outright, it hands over something an exact evaluator settles.
        "calc": {
            "type": "string",
            "description": "The arithmetic this step relies on, as a bare expression "
                           "using the numbers from THIS question and no others. "
                           "Empty string if the step has no calculation.",
        },
        # Same idea as calc, one level up. An exact evaluator settles what an
        # expression comes to; it cannot tell whether the expression MEANT
        # anything. Measured: on unit questions the model scored 0 of 4 with exact
        # arithmetic available, writing 604800/161 (seconds-in-a-week over DAYS)
        # and (8*7)*7 (days multiplied by 7 a second time). Every expression was
        # evaluated flawlessly on the way to a wrong answer.
        #
        # So it is not asked to compute the conversion. It is asked to NAME it,
        # and the factor comes from a graph of exact ratios instead of from the
        # model's memory.
        "convert": {
            "type": "string",
            "description": "The WHOLE unit conversion this step needs, in one line, "
                           "including the quantity from the question: '23 weeks to "
                           "seconds', not 'weeks to days'. Multi-hop conversions are "
                           "done for you in a single answer, exactly. Never break one "
                           "into steps and never multiply factors yourself. Empty "
                           "string if the step needs no conversion.",
        },
        # What the result IS. Without this the value is unusable at the end:
        # "(86400*378) = 32659200" does not say that it is seconds in 54 weeks, so
        # nothing can match it to the question. Measured, offering unlabelled
        # expressions as candidate answers scored 0 of 5 -- the model rejected all
        # of them -- while the same values labelled scored 9 of 12.
        "calc_of": {
            "type": "string",
            "description": "What the calc RESULT is, named in the question's own "
                           "words and specific enough to tell it apart from every "
                           "other value in this problem. A label that would fit two "
                           "different values identifies neither. Use only wording "
                           "drawn from this question. Empty string if no calculation.",
        },
        # Same contract as calc and convert: name it, never work it out. The
        # calculus battery measured the model dropping a chain factor and
        # reading a tangent at the wrong point -- the guessed-instead-of-asked
        # failure this field closes for derivatives.
        "derivative": {
            "type": "string",
            "description": "A derivative this step needs, in one line with the "
                           "point included: 'd/dx <expression> at x=<point>'. It "
                           "is computed exactly and given back to you -- never "
                           "apply the chain rule yourself. Empty string if the "
                           "step needs no derivative.",
        },
        "next_action": {"type": "string", "enum": ["continue", "final_answer"]},
    },
    "required": ["title", "content", "calc", "calc_of", "convert", "derivative",
                 "next_action"],
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
