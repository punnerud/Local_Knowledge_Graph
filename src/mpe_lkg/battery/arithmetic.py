"""The original arithmetic battery, generation sequence untouched.

This module moved from dev/battery.py, and the one thing that must not change in
the move is the RNG sequence: the questions for a given seed are pinned by
recorded claims in docs/claims/battery_*.json, and a re-ordered draw would make
every one of them a comparison against a different battery. The registry hands
this domain the RAW seed for the same reason.
"""

from __future__ import annotations

import random
from fractions import Fraction

from . import Question, generator

SEED = 20260811


def _big(rng: random.Random, digits: int) -> int:
    return rng.randrange(10 ** (digits - 1), 10**digits)


def _build(seed: int, per_group: int) -> list[Question]:
    rng = random.Random(seed)
    out: list[Question] = []

    for _ in range(per_group):
        a, b = _big(rng, 11), _big(rng, 4)
        out.append(Question(
            "long_multiplication",
            f"What is {a} * {b}?",
            Fraction(a * b), f"{a}*{b}",
        ))

    for _ in range(per_group):
        a, b = _big(rng, 12), _big(rng, 12)
        out.append(Question(
            "long_addition",
            f"What is {a} plus {b}?",
            Fraction(a + b), f"{a}+{b}",
        ))

    for _ in range(per_group):
        a, b = _big(rng, 13), _big(rng, 11)
        out.append(Question(
            "long_subtraction",
            f"Subtract {b} from {a}. What is left?",
            Fraction(a - b), f"{a}-{b}",
        ))

    for _ in range(per_group):
        pct, total = rng.randrange(3, 98), _big(rng, 7)
        out.append(Question(
            "awkward_percentage",
            f"What is {pct} percent of {total}?",
            Fraction(pct * total, 100), f"({pct}/100)*{total}",
        ))

    for _ in range(per_group):
        divisor = rng.randrange(17, 999)
        quotient, remainder = _big(rng, 8), rng.randrange(1, 17)
        total = divisor * quotient + remainder
        out.append(Question(
            "division_with_remainder",
            f"How many whole times does {divisor} go into {total}, "
            f"and what is the remainder? Give the remainder.",
            Fraction(remainder), f"{total}-({divisor}*{quotient})",
        ))

    for _ in range(per_group):
        # A word problem, so the arithmetic has to be extracted before it is done.
        crates = rng.randrange(137, 999)
        per_crate = rng.randrange(23, 97)
        broken = rng.randrange(11, 89)
        out.append(Question(
            "word_problem",
            f"A warehouse holds {crates} crates with {per_crate} jars in each. "
            f"{broken} jars are broken and thrown away. How many jars remain?",
            Fraction(crates * per_crate - broken), f"{crates}*{per_crate}-{broken}",
        ))

    for _ in range(per_group):
        # Units chained two deep, which is where "10080 minutes in a fortnight"
        # came from: the model dropped a factor and nothing noticed.
        weeks = rng.randrange(3, 97)
        out.append(Question(
            "unit_chain",
            f"How many seconds are there in {weeks} weeks?",
            Fraction(weeks * 7 * 24 * 60 * 60), f"{weeks}*7*24*60*60",
        ))

    for _ in range(per_group):
        a, b = _big(rng, 6), _big(rng, 6)
        out.append(Question(
            "exact_fraction",
            f"What is {a} divided by {b}? Give the answer to four decimal places.",
            Fraction(a, b), f"{a}/{b}",
        ))

    for _ in range(per_group):
        base, exponent = rng.randrange(11, 99), rng.randrange(5, 13)
        out.append(Question(
            "powers",
            f"What is {base} to the power of {exponent}?",
            Fraction(base**exponent), f"{base}**{exponent}",
        ))

    for _ in range(per_group):
        # Two quantities to compute and then compare -- the shape that catches an
        # answer built from the right sums combined the wrong way round.
        rate_a, hours_a = rng.randrange(1017, 9999), rng.randrange(103, 999)
        rate_b, hours_b = rng.randrange(1017, 9999), rng.randrange(103, 999)
        while rate_a * hours_a == rate_b * hours_b:
            hours_b = rng.randrange(103, 999)
        bigger = max(rate_a * hours_a, rate_b * hours_b)
        out.append(Question(
            "comparison",
            f"Machine A makes {rate_a} parts an hour for {hours_a} hours. "
            f"Machine B makes {rate_b} parts an hour for {hours_b} hours. "
            f"How many parts does the more productive machine make in total?",
            Fraction(bigger),
            f"max({rate_a}*{hours_a},{rate_b}*{hours_b})",
        ))

    return out

@generator("arithmetic")
def build(seed: int, per_group: int) -> list[Question]:
    return _build(seed, per_group)
