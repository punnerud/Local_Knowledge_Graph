"""Arithmetic questions with numbers no model has memorised, and exact answers.

The point of generating these rather than writing them is not volume. It is that
both halves of a test question can go wrong, and writing them by hand gets both
wrong at once:

* **The numbers.** "What is 17 percent of 250" is in every training set several
  thousand times over, so a model answering it correctly has demonstrated recall,
  not arithmetic. An eleven-digit operand drawn from a seeded RNG is not.
* **The answer.** A hand-written expectation is a second chance to be wrong, and
  this eval has already been bitten by one: the expected answer for "circle of
  radius 3 or square of side 5" was recorded as the square, so every arm that
  answered correctly was marked wrong. Here the truth is whatever Python computes
  from the same numbers the question was rendered from, so the two cannot drift.

Seeded, so the battery is identical on every machine and every run. Change the
seed and you have a fresh battery that nothing has been tuned against -- which is
the honest way to check a number that has started looking too good.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

SEED = 20260811


@dataclass(frozen=True)
class Question:
    group: str
    text: str
    # The exact answer, as a Fraction so nothing is lost to float rounding.
    answer: Fraction
    # The expression a step would hand over. This is what the evaluator is checked
    # against, and it is built from the same numbers as the text.
    expression: str

    def matches(self, said: str) -> bool:
        """Is the exact answer present in what the model wrote?

        Deliberately generous about presentation and strict about the value:
        thousands separators and currency marks are stripped, but 510494241751691
        is not 510494241751692. A near miss on a fifteen-digit product is not a
        rounding difference, it is a wrong answer.
        """
        cleaned = said.replace(",", "").replace(" ", "").replace("_", "")
        if self.answer.denominator == 1:
            return str(self.answer.numerator) in cleaned
        # Non-integer answers are accepted at the precision they were asked for.
        exact = float(self.answer)
        return any(f"{exact:.{p}f}".rstrip("0").rstrip(".") in cleaned for p in (4, 3, 2))


def _big(rng: random.Random, digits: int) -> int:
    return rng.randrange(10 ** (digits - 1), 10**digits)


def build(seed: int = SEED, per_group: int = 4) -> list[Question]:
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


def groups() -> list[str]:
    return sorted({q.group for q in build()})


def truth_table(questions: list[Question]) -> str:
    """The battery as text, for pasting into a report."""
    width = max(len(q.group) for q in questions)
    return "\n".join(
        f"{q.group:{width}}  {q.expression:>44}  = {q.answer}" for q in questions
    )


if __name__ == "__main__":
    battery = build()
    print(f"{len(battery)} questions across {len(groups())} groups, seed {SEED}")
    print(truth_table(battery))
    print(f"\nlargest answer has {len(str(max(abs(q.answer.numerator) for q in battery)))} digits")
