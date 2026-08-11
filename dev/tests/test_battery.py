"""The arithmetic battery: that it is well-formed, and that the evaluator settles it.

Two things are being tested here and they are worth keeping apart.

**The battery itself.** A generated question set is only worth having if the
generator is right, and a generator that quietly produces four-digit "eleven
digit" numbers, or duplicate questions, or answers that disagree with the text,
would make every downstream measurement meaningless while looking fine. So the
battery is checked before it is used.

**The evaluator.** Every question carries the expression a step would hand over.
mpeqs must settle all of them exactly -- if it cannot, the gate cannot help the
model on that shape of problem, and that is worth knowing here rather than
discovering it in an eval run.

What is NOT tested here is whether the model gets them right. That needs Ollama
and takes minutes, so it lives in test_battery_model.py behind the ollama mark.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
from battery import SEED, Question, build, groups

from mpe_lkg.arithmetic import as_text, evaluate

BATTERY = build()


def test_the_battery_is_the_same_everywhere():
    """Seeded, so a number measured on one machine means something on another."""
    assert build() == build()
    assert build(SEED) == BATTERY
    assert build(SEED + 1) != BATTERY


def test_every_group_is_represented():
    covered = {q.group for q in BATTERY}
    assert covered == set(groups())
    assert len(covered) >= 10, "a battery of one shape measures one shape"


def test_no_two_questions_are_the_same():
    # Duplicates would weight one shape of problem twice without saying so.
    texts = [q.text for q in BATTERY]
    assert len(set(texts)) == len(texts)


@pytest.mark.parametrize("question", BATTERY, ids=lambda q: q.text[:40])
def test_the_expression_evaluates_to_the_stated_answer(question: Question):
    """The two halves of a question cannot be allowed to drift apart.

    A hand-written expectation is a second chance to be wrong, and this eval has
    already been bitten by one: "circle of radius 3 or square of side 5" was
    recorded as the square, so every arm answering correctly was marked wrong.
    """
    settled = evaluate(question.expression)
    assert settled is not None, f"mpeqs will not evaluate {question.expression!r}"
    assert settled == question.answer


@pytest.mark.parametrize("question", BATTERY, ids=lambda q: q.group)
def test_the_numbers_are_too_long_to_have_been_memorised(question: Question):
    """A model answering "17 percent of 250" has demonstrated recall, not arithmetic.

    The size that matters is the size of the WORK, and it can sit at either end.
    A division with remainder has a two-digit answer and a thirteen-digit
    dividend; "83 to the power of 11" has two small operands and a twenty-one
    digit result. Both are beyond recall. A question is too easy only when
    neither end is large.
    """
    operands = [int(n) for n in re.findall(r"\d+", question.expression)]
    largest = max(max(operands), abs(question.answer.numerator))
    assert largest >= 1000, (
        f"{question.expression} = {question.answer}: neither the operands nor the "
        "result is large enough to be beyond recall")


def test_the_grader_accepts_presentation_and_refuses_wrong_values():
    q = Question("t", "irrelevant", Fraction(510494241751692), "x")
    assert q.matches("The answer is 510494241751692.")
    assert q.matches("510,494,241,751,692")
    assert q.matches("about 510 494 241 751 692 give or take")
    # One digit out on a fifteen-digit product is a wrong answer, not a rounding
    # difference, and a grader that shrugs at it measures nothing.
    assert not q.matches("510494241751691")
    assert not q.matches("510494241752")


def test_the_grader_accepts_a_rounded_decimal_at_the_precision_asked_for():
    q = Question("t", "irrelevant", Fraction(1, 3), "1/3")
    assert q.matches("0.3333")
    assert q.matches("roughly 0.333")
    assert not q.matches("0.34")


def test_answers_are_reported_in_a_form_a_person_reads():
    # 85/2 is correct and useless in a sentence.
    assert as_text(evaluate("(17/100)*250")) == "42.5"
    assert as_text(evaluate("76835376543*6644")) == "510494241751692"


@pytest.mark.parametrize(
    "hostile",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "(1).__class__.__bases__[0].__subclasses__()",
        "9**9**9",
        "[x for x in range(10**9)]",
        "eval('1+1')",
    ],
)
def test_the_evaluator_refuses_anything_that_is_not_arithmetic(hostile: str):
    """The expression comes from a language model, so it is untrusted input.

    mpeqs parses to an AST and walks it against a whitelist of node types and
    function names, with __builtins__ emptied -- it does not execute what it is
    given. This pins that, because "it only ever gets arithmetic" is an
    assumption about a model's output rather than a property of the code.
    """
    assert evaluate(hostile) is None
