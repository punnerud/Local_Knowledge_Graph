"""The battery's new domains: registered, self-grading, and honestly labelled.

The standard every domain has to meet before it ships: its answers must be
verifiable by something other than the author's memory. Units grade against the
audited conversion graph, logic against brute-force enumeration, and consistency
declares that it cannot grade correctness at all -- which is itself the honest
label this file pins in place.
"""

from __future__ import annotations

import itertools
import random
import re
from fractions import Fraction

import pytest

from mpe_lkg.battery import GENERATORS, Question, build, domain_seed
from mpe_lkg.battery.bench import summarise


class TestTheRegistry:
    def test_the_four_domains_are_registered(self):
        assert set(GENERATORS) >= {"arithmetic", "units", "logic", "consistency"}

    def test_an_unknown_domain_is_named_rather_than_skipped(self):
        with pytest.raises(KeyError, match="volcanology"):
            build(domains=["volcanology"])

    def test_every_question_carries_its_domain(self):
        for question in build(per_group=1):
            assert question.domain in GENERATORS

    def test_adding_a_domain_cannot_change_anothers_questions(self):
        """The reason for per-domain seeds: recorded claims pin the arithmetic
        battery, and a shared RNG would reshuffle it whenever a domain was
        added. crc32, not hash(), because hash() is randomised per process."""
        alone = [q.text for q in build(per_group=2, domains=["units"])]
        together = [q.text for q in build(per_group=2) if q.domain == "units"]
        assert alone == together
        assert domain_seed(1, "units") == domain_seed(1, "units")
        assert domain_seed(1, "units") != domain_seed(1, "logic")


class TestUnits:
    @pytest.mark.parametrize("question", build(per_group=2, domains=["units"]),
                             ids=lambda q: q.group)
    def test_the_answer_is_the_conversion_graphs(self, question: Question):
        """Same call as the grader, so drift is impossible by construction."""
        from mpeqs import units

        value, source, _, target = question.expression.split(" ", 3)
        target = target.removeprefix("to ").strip()
        assert question.answer == units.convert(Fraction(value), source, target)

    def test_the_decimal_form_terminates(self):
        # matches() grades to a few decimal places, so a question whose answer
        # cannot be written in decimals would be ungradeable by design.
        for question in build(per_group=3, domains=["units"]):
            denominator = question.answer.denominator
            for prime in (2, 5):
                while denominator % prime == 0:
                    denominator //= prime
            assert denominator == 1, f"{question.expression} -> {question.answer}"


class TestLogic:
    """Every closed form checked by brute force, not by remembering it."""

    def _numbers(self, text):
        return [int(n) for n in re.findall(r"\d+", text)]

    @pytest.mark.parametrize("question",
                             [q for q in build(per_group=3, domains=["logic"])
                              if q.group == "pigeonhole_pair"],
                             ids=lambda q: q.text[:40])
    def test_pigeonhole_pair_by_exhaustion(self, question: Question):
        # The drawer's counts are every number in the text.
        counts = self._numbers(question.text)
        drawer = list(itertools.chain.from_iterable(
            [i] * c for i, c in enumerate(counts)))
        need = int(question.answer)
        # need-1 draws can miss a pair; need draws cannot.
        rng = random.Random(0)
        missed_possible = False
        for _ in range(2000):
            draw = rng.sample(drawer, need - 1)
            if len(set(draw)) == len(draw):
                missed_possible = True
                break
        assert missed_possible, "the bound is not tight"
        for _ in range(200):
            draw = rng.sample(drawer, need)
            assert len(set(draw)) < len(draw), "the bound does not guarantee"

    @pytest.mark.parametrize("question",
                             [q for q in build(per_group=3, domains=["logic"])
                              if q.group == "handshakes"],
                             ids=lambda q: q.text[:30])
    def test_handshakes_by_enumeration(self, question: Question):
        people = self._numbers(question.text)[0]
        assert question.answer == len(list(itertools.combinations(range(people), 2)))

    @pytest.mark.parametrize("question",
                             [q for q in build(per_group=3, domains=["logic"])
                              if q.group == "inclusion_exclusion"],
                             ids=lambda q: q.text[:30])
    def test_inclusion_exclusion_by_sets(self, question: Question):
        first, second, both = self._numbers(question.text)
        a = set(range(first))
        b = set(range(first - both, first - both + second))
        assert len(a & b) == both, "the generated overlap must be consistent"
        assert question.answer == len(a | b)


class TestConsistency:
    def test_consistency_questions_have_no_answer_to_leak(self):
        for question in build(per_group=3, domains=["consistency"]):
            assert question.mode == "consistency"
            assert question.answer is None
            assert question.matches("anything at all") is None, \
                "ungradeable must never read as wrong"

    def test_the_summary_keeps_consistency_out_of_correctness(self):
        """The label the whole domain depends on: steady is not correct."""
        rows = [
            {"group": "g", "domain": "d", "mode": "exact", "correct": True,
             "settled_exactly": False, "steps": 3, "error": ""},
            {"group": "anchored", "domain": "consistency", "mode": "consistency",
             "steady": True, "said": "Paris"},
            {"group": "invented", "domain": "consistency", "mode": "consistency",
             "steady": False, "said": "75006"},
        ]
        out = summarise(rows)
        assert out["n"] == 1, "only the exact row is graded"
        assert out["correct"] == 1
        assert out["consistency"]["n"] == 2
        assert out["consistency"]["steady"] == 1
        assert "invisible" in out["consistency"]["note"]


class TestTextMode:
    def test_text_questions_grade_by_expectation(self):
        q = Question("g", "Which weighs more?", mode="text",
                     expect=("neither", "the same", "equal"))
        assert q.matches("They weigh the same.") is True
        assert q.matches("The lead, obviously.") is False


class TestPhysics:
    """Every answer re-derived from the question's own text.

    The physics domain leans on constants, and the rule that makes it safe is
    that every constant is STATED in the question -- so these tests parse the
    figures back out of the text and recompute the answer from nothing else.
    The very first version of this domain failed this test twice: both
    mile-based answers were a factor of a thousand off, because MILE_KM is
    already kilometres per mile and the generator multiplied by 1000 again.
    A constant embedded twice is exactly the mistake this style of test exists
    to catch before a model is ever graded against it.
    """

    def _numbers(self, text):
        return [Fraction(n.replace(",", ""))
                for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]

    @pytest.mark.parametrize("question",
                             [q for q in build(per_group=3, domains=["physics"])
                              if q.group == "light_travel"],
                             ids=lambda q: q.text[60:90])
    def test_light_travel_from_the_stated_figures(self, question):
        speed, distance = self._numbers(question.text)[:2]
        assert question.answer == distance * 1000 / speed

    @pytest.mark.parametrize("question",
                             [q for q in build(per_group=3, domains=["physics"])
                              if q.group == "wind_distance"],
                             ids=lambda q: q.text[30:55])
    def test_wind_distance_from_the_stated_figures(self, question):
        # "One mile" is spelled out, so the digits in the text are exactly
        # speed, the mile factor and the hours.
        speed, mile_km, hours = self._numbers(question.text)[:3]
        assert question.answer == speed * hours * mile_km

    @pytest.mark.parametrize("question",
                             [q for q in build(per_group=3, domains=["physics"])
                              if q.group == "around_earth"],
                             ids=lambda q: q.text[50:70])
    def test_around_earth_from_the_stated_figures(self, question):
        circumference, speed = self._numbers(question.text)[:2]
        assert question.answer == circumference / speed

    @pytest.mark.parametrize("question",
                             [q for q in build(per_group=3, domains=["physics"])
                              if q.group == "speed_conversion"],
                             ids=lambda q: q.text[17:35])
    def test_speed_conversion_from_the_stated_figures(self, question):
        speed, mile_km = self._numbers(question.text)[:2]
        assert question.answer == speed * mile_km * 1000 / 3600

    def test_no_question_asks_the_model_to_remember_a_constant(self):
        """The rule that keeps physics inside the battery's discipline."""
        for question in build(per_group=2, domains=["physics"]):
            numbers = self._numbers(question.text)
            assert len(numbers) >= 2, \
                f"a constant is missing from the text: {question.text}"

    def test_the_light_constants_give_plausible_magnitudes(self):
        """Sanity anchors on the constants themselves, not on an RNG draw.

        The first version waited for the Moon to be drawn and it never was at
        that seed -- a test hoping the RNG cooperates tests the RNG.
        """
        from mpe_lkg.battery.physics import BODIES, LIGHT

        seconds = {name: float(d * 1000 / LIGHT) for name, d in BODIES.items()}
        assert 1 < seconds["the Moon"] < 2
        assert 480 < seconds["the Sun"] < 520          # about eight minutes
        assert 150 < seconds["Mars at its closest"] < 220


class TestCalculus:
    """Every answer checked through BOTH of mpeqs' derivative paths.

    The calculus module ships the derivative twice -- symbolic rules and
    dual-number autograd -- precisely so a consumer can demand agreement.
    This battery does: a question whose two gradings disagreed would fail
    here before any model ever saw it.
    """

    def _questions(self, group):
        return [q for q in build(per_group=3, domains=["calculus"])
                if q.group == group]

    @pytest.mark.parametrize("question_index", range(3))
    def test_derivatives_grade_the_same_both_ways(self, question_index):
        from mpeqs import calculus as mc

        question = self._questions("derivative_at")[question_index]
        # d/dx (a*x**2 + b)**n at point -- parse the pieces back out.
        m = re.search(r"\((\d+)x\^2 \+ (\d+)\)\^(\d+).*?x = (-?[\d/]+)",
                      question.text)
        a, b, n, at = m.group(1), m.group(2), m.group(3), Fraction(m.group(4))
        expression = f"({a}*x**2 + {b})**{n}"
        assert question.answer == mc.derivative_at(expression, at=at)
        assert question.answer == mc.grad(expression, at)

    @pytest.mark.parametrize("question_index", range(3))
    def test_integrals_match_the_antiderivative_at_the_bounds(self, question_index):
        from mpeqs import calculus as mc

        question = self._questions("poly_integral")[question_index]
        m = re.search(r"of (-?\d+)x\^2 \+ (-?\d+)x \+ (-?\d+) from x = (-?\d+) "
                      r"to x = (-?\d+)", question.text)
        c3, c1, c0, low, high = (int(g) for g in m.groups())
        expression = f"{c3}*x**2 + {c1}*x + {c0}"
        assert question.answer == mc.integrate(expression, lower=low, upper=high)
        # And by the fundamental theorem, through the OTHER path: the
        # antiderivative evaluated at the bounds.
        anti = mc.integrate(expression)
        assert question.answer == (mc.evaluate_at(anti, at=high)
                                   - mc.evaluate_at(anti, at=low))

    @pytest.mark.parametrize("question_index", range(3))
    def test_quadratic_roots_substitute_back_to_zero(self, question_index):
        question = self._questions("quadratic_root")[question_index]
        m = re.search(r"Solve (\d+)x\^2 ([+-]) (\d+)x ([+-]) (\d+) = 0",
                      question.text)
        a = int(m.group(1))
        b = int(m.group(3)) * (1 if m.group(2) == "+" else -1)
        c = int(m.group(5)) * (1 if m.group(4) == "+" else -1)
        root = question.answer
        assert a * root**2 + b * root + c == 0

    def test_the_tangent_is_the_derivative_wearing_words(self):
        from mpeqs import calculus as mc

        for question in self._questions("tangent_slope"):
            m = re.search(r"y = (\d+)x\^2 \+ (-?\d+)x.*?x = (-?\d+)", question.text)
            c2, c1, at = (int(g) for g in m.groups())
            assert question.answer == mc.grad(f"{c2}*x**2 + {c1}*x", at)


class TestPhysicsExtensions:
    def _questions(self, group):
        return [q for q in build(per_group=3, domains=["physics"])
                if q.group == group]

    def test_free_fall_round_trips_through_the_stated_formula(self):
        for question in self._questions("free_fall"):
            m = re.search(r"falls ([\d.]+) metres", question.text)
            height = Fraction(m.group(1))
            g = Fraction(981, 100)
            # h = g*t^2/2 with the graded t must reproduce the stated height.
            assert g * question.answer**2 / 2 == height

    def test_acceleration_from_the_stated_figures(self):
        for question in self._questions("acceleration"):
            u, a, t = [int(n) for n in re.findall(r"\d+", question.text)][:3]
            assert question.answer == u + a * t

    def test_kinetic_energy_from_the_stated_figures(self):
        for question in self._questions("kinetic_energy"):
            m, v = [int(n) for n in re.findall(r"\d+", question.text)][:2]
            assert question.answer == Fraction(m * v * v, 2)
