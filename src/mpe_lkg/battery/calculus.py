"""Calculus questions, graded by two derivative implementations that must agree.

The truth for every question here comes from ``mpeqs.calculus`` -- and that
module ships the derivative two independent ways (symbolic rules and dual-number
autograd), which this battery leans on: the domain tests grade each question
through BOTH paths and fail if they ever disagree, so no answer in this file
rests on either implementation alone, let alone on the author.

Questions are phrased so the expected answer is a single rational number --
a derivative at a point, a definite integral, a root -- because that is what
``matches`` can grade without judgement. Points and coefficients are drawn so
answers stay rational by construction: quadratics are built FROM their roots.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mpeqs import calculus

from . import Question, generator


def _rational(rng: random.Random, span: int = 9, bottom: int = 4) -> Fraction:
    value = Fraction(rng.randrange(-span, span + 1), rng.randrange(1, bottom))
    return value


@generator("calculus")
def build(seed: int, per_group: int) -> list[Question]:
    rng = random.Random(seed)
    out: list[Question] = []

    for _ in range(per_group):
        # A chain-rule composition: (a*x**2 + b)**n at a rational point.
        a, b = rng.randrange(1, 6), rng.randrange(1, 9)
        n = rng.randrange(2, 5)
        at = Fraction(rng.randrange(-6, 7), rng.choice([1, 2]))
        expression = f"({a}*x**2 + {b})**{n}"
        out.append(Question(
            group="derivative_at",
            text=(f"What is the derivative of ({a}x^2 + {b})^{n} with respect "
                  f"to x, evaluated at x = {at}? Give the exact value."),
            answer=calculus.derivative_at(expression, at=at),
            expression=f"d/dx {expression} at {at}",
        ))

    for _ in range(per_group):
        # A definite polynomial integral with integer bounds.
        c3, c1, c0 = rng.randrange(1, 7), rng.randrange(-9, 10), rng.randrange(-9, 10)
        low = rng.randrange(-3, 2)
        high = low + rng.randrange(1, 5)
        expression = f"{c3}*x**2 + {c1}*x + {c0}"
        out.append(Question(
            group="poly_integral",
            text=(f"What is the definite integral of {c3}x^2 + {c1}x + {c0} "
                  f"from x = {low} to x = {high}? Give the exact value."),
            answer=calculus.integrate(expression, lower=low, upper=high),
            expression=f"integral {expression} from {low} to {high}",
        ))

    for _ in range(per_group):
        # The same derivative wearing words: the slope of a tangent line.
        c2, c1 = rng.randrange(1, 8), rng.randrange(-9, 10)
        at = rng.randrange(-5, 6)
        expression = f"{c2}*x**2 + {c1}*x"
        out.append(Question(
            group="tangent_slope",
            text=(f"A curve follows y = {c2}x^2 + {c1}x. What is the slope of "
                  f"the tangent line at x = {at}?"),
            answer=calculus.derivative_at(expression, at=at),
            expression=f"d/dx {expression} at {at}",
        ))

    for _ in range(per_group):
        # Built FROM its roots, so the truth exists by construction and the
        # discriminant is a perfect square by construction too.
        r1 = rng.randrange(-9, 10)
        r2 = rng.randrange(-9, 10)
        if r1 == r2:
            r2 += 1
        a = rng.randrange(1, 4)
        b, c = -a * (r1 + r2), a * r1 * r2
        larger = max(r1, r2)
        sign = lambda v: f"+ {v}" if v >= 0 else f"- {-v}"  # noqa: E731
        out.append(Question(
            group="quadratic_root",
            text=(f"Solve {a}x^2 {sign(b)}x {sign(c)} = 0. "
                  "What is the largest root?"),
            answer=Fraction(larger),
            expression=f"largest root of {a}*x**2 + {b}*x + {c}",
        ))

    return out
