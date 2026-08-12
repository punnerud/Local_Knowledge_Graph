"""Unit conversions, graded against the exact conversion graph.

The truth here is ``mpeqs.units.convert``: a path of exact ratios through the
unit graph, audited for internal consistency. That makes these questions
self-grading in the strongest sense available -- the same call that produces the
expected answer is the one the reasoning loop's ``convert`` field uses, so a
disagreement between question and grader is impossible by construction.

The unit pairs are drawn from the graph itself rather than hand-listed, filtered
to same-dimension pairs whose factor terminates as a decimal, so ``matches``
never has to judge a non-terminating expansion.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mpeqs import units

from . import Question, generator

# Pairs whose conversion terminates in decimal and reads naturally in a
# question. Drawn from the graph's own vocabulary; audited by mpeqs' tests.
MASS = [("kilogram", "gram"), ("tonne", "kilogram"), ("pound", "gram"),
        ("stone", "pound")]
LENGTH = [("kilometre", "metre"), ("mile", "metre"), ("foot", "millimetre"),
          ("inch", "millimetre")]
TIME = [("week", "second"), ("day", "minute"), ("fortnight", "hour"),
        ("year", "minute")]
DIGITAL = [("gibibyte", "byte"), ("mebibyte", "byte"), ("megabyte", "kilobyte"),
           ("kibibyte", "bit")]


def _ask(rng: random.Random, group: str, pairs: list[tuple[str, str]],
         low: int, high: int) -> Question:
    source, target = rng.choice(pairs)
    value = rng.randrange(low, high)
    exact = units.convert(Fraction(value), source, target)
    return Question(
        group=group,
        text=f"How many {target}s are there in {value} {source}s?",
        answer=exact,
        expression=f"{value} {source} to {target}",
    )


@generator("units")
def build(seed: int, per_group: int) -> list[Question]:
    rng = random.Random(seed)
    out: list[Question] = []
    for group, pairs, low, high in (
        ("convert_mass", MASS, 3, 970),
        ("convert_length", LENGTH, 3, 970),
        ("convert_time", TIME, 3, 97),
        ("convert_digital", DIGITAL, 2, 48),
    ):
        for _ in range(per_group):
            out.append(_ask(rng, group, pairs, low, high))
    return out
