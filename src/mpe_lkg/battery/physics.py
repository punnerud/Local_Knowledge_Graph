"""Physical composition: light, wind, speed and distance, graded exactly.

The battery's rule is that the truth is computed by the same code that renders
the question -- and physics questions threaten that rule, because they lean on
constants. The resolution: EVERY constant is stated in the question text. The
model is never asked to remember the speed of light or the distance to Mars; it
is asked to compose figures it has been handed, across units, which is exactly
the failure mode this project has measured models into and built machinery for.

The constants are reference values, fixed here so the battery is stable, and
carried into the question verbatim so the tests can re-derive every answer from
the question's own text -- a question whose stated figures disagreed with its
graded answer would fail its own test.

Answers are exact Fractions; most do not terminate as decimals (the speed of
light has ugly prime factors), and ``matches`` grades those at the 2-4 decimal
places a sensible answer would state.
"""

from __future__ import annotations

import random
from fractions import Fraction

from . import Question, generator

# Metres per second, exact by definition of the metre.
LIGHT = Fraction(299_792_458)

# Average distances in kilometres, fixed reference values. Stated in every
# question that uses them, so nothing depends on the model or the reader
# agreeing with the almanac.
BODIES = {
    "the Moon": Fraction(384_400),
    "the Sun": Fraction(149_597_870),
    "Mars at its closest": Fraction(54_600_000),
    "Mars on average": Fraction(225_000_000),
    "Jupiter on average": Fraction(778_500_000),
}

# One mile is exactly 201168/125 metres; the factor to km/h from mph.
MILE_KM = Fraction(201_168, 125_000)

# Equatorial circumference, km.
EQUATOR = Fraction(40_075)


@generator("physics")
def build(seed: int, per_group: int) -> list[Question]:
    rng = random.Random(seed)
    out: list[Question] = []

    for _ in range(per_group):
        body = rng.choice(sorted(BODIES))
        distance = BODIES[body]
        out.append(Question(
            group="light_travel",
            text=(f"Light travels at 299,792,458 metres per second. The average "
                  f"distance to {body} is {int(distance):,} kilometres. How many "
                  f"seconds does light take to cover that distance? Give the "
                  "answer to two decimal places."),
            answer=distance * 1000 / LIGHT,
            expression=f"{distance}*1000/299792458",
        ))

    for _ in range(per_group):
        # Wind speed held in one unit, distance asked in another: the mixed-unit
        # composition the assembly used to guess exponents on.
        speed = rng.randrange(8, 130)
        hours = rng.randrange(2, 48)
        out.append(Question(
            group="wind_distance",
            text=(f"A storm wind blows steadily at {speed} miles per hour. One "
                  f"mile is exactly 1.609344 kilometres. How many kilometres "
                  f"does the wind travel in {hours} hours?"),
            answer=Fraction(speed) * hours * MILE_KM,
            expression=f"{speed}*{hours}*1.609344",
        ))

    for _ in range(per_group):
        speed = rng.randrange(15, 900)
        out.append(Question(
            group="around_earth",
            text=(f"The Earth's equator is 40,075 kilometres around. Travelling "
                  f"at a constant {speed} kilometres per hour, how many hours "
                  f"does one full circuit take?"),
            answer=EQUATOR / speed,
            expression=f"40075/{speed}",
        ))

    for _ in range(per_group):
        speed = rng.randrange(10, 200)
        out.append(Question(
            group="speed_conversion",
            text=(f"A vehicle moves at {speed} miles per hour. One mile is "
                  f"exactly 1.609344 kilometres. What is its speed in metres "
                  f"per second?"),
            answer=Fraction(speed) * MILE_KM * 1000 / 3600,
            expression=f"{speed}*1609.344/3600",
        ))

    for _ in range(per_group):
        # g stated, height BUILT from the answer: h = g*t^2/2 with integer t,
        # so h's denominator is 200 and three decimals state it exactly. The
        # first version formatted with %g, whose six significant digits
        # truncated 3575.745 to "3575.74" -- a question whose stated figure
        # disagreed with its graded answer, caught by the round-trip test
        # before any model saw it.
        t_fall = Fraction(rng.randrange(2, 30))
        g = Fraction(981, 100)
        height = g * t_fall * t_fall / 2
        thousandths = height * 1000
        assert thousandths.denominator == 1
        stated = f"{int(thousandths) / 1000:.3f}".rstrip("0").rstrip(".")
        out.append(Question(
            group="free_fall",
            text=(f"An object falls from rest, and the height fallen follows "
                  f"h = g*t^2/2 with g = 9.81 metres per second squared. It "
                  f"falls {stated} metres. How many seconds was it falling?"),
            answer=t_fall,
            expression=f"sqrt(2*{stated}/9.81)",
        ))

    for _ in range(per_group):
        # v = u + a*t, all three stated, solve for v.
        u = rng.randrange(0, 30)
        a = rng.randrange(2, 12)
        t_run = rng.randrange(3, 40)
        out.append(Question(
            group="acceleration",
            text=(f"A vehicle starts at {u} metres per second and accelerates "
                  f"at {a} metres per second squared for {t_run} seconds, "
                  f"following v = u + a*t. What is its final speed in metres "
                  "per second?"),
            answer=Fraction(u + a * t_run),
            expression=f"{u}+{a}*{t_run}",
        ))

    for _ in range(per_group):
        # KE = m*v^2/2, both stated, v even so the answer is an integer.
        mass = rng.randrange(2, 40)
        speed = rng.randrange(2, 30) * 2
        out.append(Question(
            group="kinetic_energy",
            text=(f"A mass of {mass} kilograms moves at {speed} metres per "
                  f"second. Using KE = m*v^2/2, what is its kinetic energy "
                  "in joules?"),
            answer=Fraction(mass * speed * speed, 2),
            expression=f"{mass}*{speed}**2/2",
        ))

    return out
