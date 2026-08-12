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
