"""Questions with no computable truth, graded by self-agreement -- and said so.

Everything else in the battery grades against something outside the model: an
exact evaluator, a conversion graph, a theorem. These questions have no such
anchor, so the only measurement available is whether the model agrees with
ITSELF when asked repeatedly at sampling temperature -- the knowledge-edge probe
from ``reasoning.edge``, applied as a battery.

What that measures, precisely: at the edge of what it knows a model does not
fall silent, it confabulates, and confabulates differently each time. Steadiness
is therefore evidence of knowledge and unsteadiness is evidence of its absence.

What it cannot measure, and the reason these rows are never mixed into a
correctness score: a model can be confidently and consistently wrong. Measured
in this repository -- asked for Rue Cler's postal code, one model answered 75006
every single time, and the answer is 75007. A consumer of battery results sees
these rows under ``consistency`` with their own rate, never under ``correct``.

Two kinds of question, with opposite expectations:

* ``anchored`` -- facts broad enough that a model with any grounding holds them
  steadily. Expected steady.
* ``invented`` -- specifics about places assembled from a seeded RNG, which it
  is overwhelmingly unlikely anything true is known about. Expected UNSTEADY:
  a model that answers these steadily is manufacturing confidence, which is
  exactly the behaviour worth catching.
"""

from __future__ import annotations

import random

from . import Question, generator

# Broad anchors. The point is not that these are obscure -- it is that a model
# should hold them WITHOUT wobbling, so unsteadiness here is a red flag about
# the model as a whole.
ANCHORED = [
    "What is the capital of {}?",
    "On which continent is {}?",
    "What is the main language spoken in {}?",
]
COUNTRIES = ["France", "Japan", "Brazil", "Egypt", "Norway", "Canada",
             "Australia", "Kenya", "Portugal", "Thailand", "Peru", "Finland"]

# Parts for places that almost certainly do not exist. The names are plausible
# in shape and empty in reference, which is the point: there is nothing true to
# know, so a steady answer is a manufactured one.
STREET_FIRST = ["Vester", "Lind", "Karlen", "Mork", "Brenn", "Solha", "Tver",
                "Gamle", "Nord", "Aust"]
STREET_SECOND = ["gate", "veien", "stien", "alle", "bakken", "toppen"]
TOWNS = ["Fjellmo", "Sundvik", "Ostero", "Kleppdal", "Vasslia", "Heimnes"]


@generator("consistency")
def build(seed: int, per_group: int) -> list[Question]:
    rng = random.Random(seed)
    out: list[Question] = []

    for _ in range(per_group):
        template = rng.choice(ANCHORED)
        country = rng.choice(COUNTRIES)
        out.append(Question(
            group="anchored",
            text=template.format(country),
            mode="consistency",
        ))

    for _ in range(per_group):
        street = rng.choice(STREET_FIRST) + rng.choice(STREET_SECOND)
        town = rng.choice(TOWNS)
        number = rng.randrange(3, 90)
        out.append(Question(
            group="invented",
            text=(f"What is the postal code of {street} {number} in {town}?"),
            mode="consistency",
        ))

    return out
