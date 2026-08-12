"""Combinatorial questions whose answers are theorems, not lookups.

Everything here is graded by a closed form small enough to verify by brute
force, and the tests DO verify it by brute force -- a battery whose own answer
key rests on the author remembering the pigeonhole bound correctly would repeat
the mistake this project already made once with a hand-written expectation.
"""

from __future__ import annotations

import random
from fractions import Fraction

from . import Question, generator

COLOURS = ("red", "blue", "green", "black", "white", "grey")


@generator("logic")
def build(seed: int, per_group: int) -> list[Question]:
    rng = random.Random(seed)
    out: list[Question] = []

    for _ in range(per_group):
        # Draws to GUARANTEE a matching pair: one of each colour can be drawn
        # before the pigeonhole closes, so colours + 1.
        names = rng.sample(COLOURS, rng.randrange(2, 5))
        counts = [rng.randrange(4, 30) for _ in names]
        listed = ", ".join(f"{c} {n}" for n, c in zip(names, counts, strict=True))
        out.append(Question(
            group="pigeonhole_pair",
            text=(f"A drawer holds {listed} socks. How many socks must be drawn "
                  "in the dark to be certain of a matching pair?"),
            answer=Fraction(len(names) + 1),
            expression=f"{len(names)}+1",
        ))

    for _ in range(per_group):
        # To guarantee K of one colour: (K-1) of every colour can be drawn
        # first, so colours*(K-1) + 1 -- provided every pile holds at least K.
        names = rng.sample(COLOURS, rng.randrange(2, 4))
        need = rng.randrange(3, 7)
        counts = [rng.randrange(need + 2, need + 20) for _ in names]
        listed = ", ".join(f"{c} {n}" for n, c in zip(names, counts, strict=True))
        out.append(Question(
            group="pigeonhole_k",
            text=(f"A drawer holds {listed} socks. How many socks must be drawn "
                  f"in the dark to be certain of {need} socks of the same colour?"),
            answer=Fraction(len(names) * (need - 1) + 1),
            expression=f"{len(names)}*({need}-1)+1",
        ))

    for _ in range(per_group):
        # Inclusion-exclusion, with the overlap bounded by both sides.
        first = rng.randrange(20, 90)
        second = rng.randrange(20, 90)
        both = rng.randrange(3, min(first, second))
        out.append(Question(
            group="inclusion_exclusion",
            text=(f"In a class, {first} students study French, {second} study "
                  f"German, and {both} study both. How many students study at "
                  "least one of the two languages?"),
            answer=Fraction(first + second - both),
            expression=f"{first}+{second}-{both}",
        ))

    for _ in range(per_group):
        # Handshakes: n people, every pair once.
        people = rng.randrange(5, 40)
        out.append(Question(
            group="handshakes",
            text=(f"{people} people are at a meeting and every pair shakes hands "
                  "exactly once. How many handshakes happen?"),
            answer=Fraction(people * (people - 1), 2),
            expression=f"{people}*({people}-1)/2",
        ))

    return out
