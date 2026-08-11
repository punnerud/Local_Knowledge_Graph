"""Check the sums a model writes down, without asking it to check them.

Measured on this eval: pushing runs to eight steps lifted hard questions from
50 % to 62 % but dropped straightforward multi-step arithmetic from 94 % to 67 %.
The failures were not reasoning failures. They were "10080 minutes in a
fortnight" (that is a week) and "6.00 change" from a 20 note on 13.50 of goods.

So the arithmetic is handed to something that does arithmetic. ``mpeqs`` evaluates
exactly, in fractions, and refuses rather than guessing. This module finds the
sums in a step's prose and asks it.

Two rules keep this from doing harm, because a wrong correction is worse than no
correction:

* **Only unambiguous claims are checked.** The left side must be arithmetic and
  nothing else -- digits, operators, brackets. ``x = 5`` is a definition, not a
  claim about a sum, and is left alone.
* **Rounding is not an error.** A model writing 28.27 for 28.2743 has rounded
  correctly. The tolerance comes from how many decimals it chose to write, so
  three decimals are held to three and none to a half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

# A claim is an arithmetic expression, then = or "is", then a number. The left
# side must contain an operator: without one there is no sum to check.
CLAIM = re.compile(
    # An expression may open with a bracket -- "(17/100) * 250" is the shape a
    # model writes a percentage in, and it was the first thing this missed.
    r"(?P<expr>\(?[\d.]+\)?(?:\s*[+\-*/^]\s*\(?[\d.()]+\)?)+)"
    r"\s*(?:=|equals?|is)\s*"
    r"(?P<stated>-?[\d.]+)",
    re.IGNORECASE,
)

# What models write instead of what Python parses.
SUBSTITUTIONS = {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "^": "**"}


@dataclass
class Claim:
    expression: str
    stated: str
    exact: Fraction
    ok: bool

    def describe(self) -> str:
        return f"{self.expression} = {float(self.exact):g}, not {self.stated}"


def _normalise(text: str) -> str:
    for wrong, right in SUBSTITUTIONS.items():
        text = text.replace(wrong, right)
    # Thousands separators, but only between digits: "20,160" is a number while
    # "3, 4 and 5" is a list.
    return re.sub(r"(?<=\d),(?=\d\d\d\b)", "", text)


def _tolerance(stated: str) -> Fraction:
    """Half of the last place the model chose to write.

    Someone who writes 28.27 has claimed a value to two decimals and is right if
    the exact answer rounds there. Someone who writes 28.2743 has claimed more and
    is held to it.
    """
    if "." not in stated:
        return Fraction(1, 2)
    return Fraction(1, 2) / (10 ** len(stated.split(".")[1]))


def check(text: str, *, limit: int = 6) -> list[Claim]:
    """Every arithmetic claim in the text, with whether it holds."""
    try:
        import mpeqs
    except ImportError:
        # The feature is optional; without it the app behaves as it did before.
        return []

    found: list[Claim] = []
    for match in CLAIM.finditer(_normalise(text)):
        expression = match.group("expr").strip()
        stated = match.group("stated").strip().rstrip(".")
        if not stated or len(found) >= limit:
            break
        try:
            exact = Fraction(mpeqs.solve({"solver": "arith", "answer": expression}))
            claimed = Fraction(stated)
        except (mpeqs.Refusal, ValueError, ZeroDivisionError, ArithmeticError):
            # A refusal means the record could not derive it, which is not the
            # same as the model being wrong. Say nothing.
            continue
        found.append(Claim(expression, stated, exact, abs(exact - claimed) <= _tolerance(stated)))
    return found


def errors(text: str) -> list[Claim]:
    return [claim for claim in check(text) if not claim.ok]


def correction(bad: list[Claim]) -> str:
    """What to tell the model, naming the sum rather than the conclusion."""
    lines = "\n".join(f"- {claim.describe()}" for claim in bad)
    return (
        f"One of those sums is wrong:\n{lines}\n\n"
        "Redo this step with the corrected figure. Change only what the arithmetic "
        "changes; keep the rest of your reasoning."
    )


def evaluate(expression: str):
    """Evaluate an expression the model handed over, or None if it will not.

    This is the other half of the same idea as ``check``, and the stronger half.
    Rather than searching prose for something shaped like a sum, the step schema
    asks for the expression outright, so what arrives is already machine-readable.
    """
    text = _normalise(str(expression or "")).strip().rstrip("=").strip()
    if not text or not any(op in text for op in "+-*/"):
        return None
    try:
        import mpeqs

        return Fraction(mpeqs.solve({"solver": "arith", "answer": text}))
    except ImportError:
        return None
    except Exception:
        # A refusal means the record could not derive it, which is not evidence
        # about the model. Say nothing rather than guessing.
        return None


def as_text(value: Fraction) -> str:
    """A value a reader recognises: 42.5 rather than 85/2, 20160 rather than 20160/1."""
    if value.denominator == 1:
        return str(value.numerator)
    rounded = round(float(value), 4)
    return f"{rounded:g}"

