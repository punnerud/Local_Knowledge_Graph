"""The arithmetic gate, and the part of it that turned out to matter.

The gate has two halves, and measurement separated them sharply:

* Scraping sums out of prose found 11 checkable claims in 153 steps and zero
  errors, because a model writes "14 days at 1440 minutes each, so 20160" rather
  than an expression. Nearly inert.
* Asking for the expression outright works -- but only once the schema field is
  *required*. Optional, the model never filled it in across three questions;
  structured output simply omits what it is not obliged to produce.

And then the failure that outlived both: the loop computed 20 - 13.5 = 6.5
exactly and still answered 2.50, because the step holding the value was not on
the strongest path and the synthesis never saw it. These tests pin that the
settled sums reach the synthesis, which is the only place they can do any good.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mpe_lkg.arithmetic import as_text, evaluate
from mpe_lkg.backends import STEP_SCHEMA
from mpe_lkg.reasoning import _synthesise

mpeqs = pytest.importorskip("mpeqs", reason="the arithmetic gate is an optional extra")


class Recorder:
    """A chat backend that answers nothing and remembers what it was asked."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def stream(self, messages, max_tokens, schema=None):
        self.prompts.append(messages[-1]["content"])
        yield "an answer"


def test_calc_is_required_so_the_model_cannot_skip_it():
    # Not a style preference. As an optional property this field was supplied
    # zero times out of three real runs; required, it was supplied every time.
    assert "calc" in STEP_SCHEMA["required"]
    assert STEP_SCHEMA["properties"]["calc"]["type"] == "string"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(17/100)*250", "42.5"),
        ("14*24*60", "20160"),
        ("20-13.5", "6.5"),
        ("3*4.5", "13.5"),
        # Written the way models write it, rather than the way Python parses it.
        ("60 × 24", "1440"),
    ],
)
def test_expressions_evaluate_exactly(expression, expected):
    assert as_text(evaluate(expression)) == expected


@pytest.mark.parametrize("text", ["", "   ", "the answer is 42", "20160", None])
def test_non_expressions_yield_nothing_rather_than_guessing(text):
    # Silence is the safe failure here: a wrong correction is worse than none.
    assert evaluate(text) is None


def test_settled_sums_reach_the_synthesis():
    """The measured bug: a correct value computed and then dropped.

    The value lived in a step that was not on the strongest path, so the thread
    handed to the synthesis did not contain it and the answer was wrong. The sums
    are passed separately for exactly that reason.
    """
    chat = Recorder()
    _synthesise(chat, "What is my change?", ["I worked out the total."], ["20-13.5 = 6.5"])

    asked = chat.prompts[0]
    assert "20-13.5 = 6.5" in asked
    assert "Do not recompute" in asked


def test_repeated_sums_are_stated_once():
    # Models re-derive the same expression across angles; seeing it four times
    # tells the synthesis nothing extra and crowds out the thread.
    chat = Recorder()
    _synthesise(chat, "q", ["step"], ["3*4.5 = 13.5", "3*4.5 = 13.5", "20-13.5 = 6.5"])
    assert chat.prompts[0].count("3*4.5 = 13.5") == 1


def test_synthesis_without_sums_is_unchanged():
    """No sums must mean no extra instruction -- most questions have no arithmetic."""
    chat = Recorder()
    _synthesise(chat, "Why is the sky blue?", ["Rayleigh scattering."], [])
    assert "calculated exactly" not in chat.prompts[0]


def test_fractions_survive_without_becoming_floats():
    # The reason for mpeqs rather than eval(): a third stays a third.
    assert evaluate("1/3") == Fraction(1, 3)
    assert evaluate("(1/3)*3") == Fraction(1)
