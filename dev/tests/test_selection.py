"""The answer as a choice, not a retelling.

The failure this replaces has a name in the literature -- Result-Ignore: the final
answer states something other than what the tool returned. It is measured at 30%
on an 8B Llama, and llama3.2:3b sits below that model on every grounding
benchmark that covers it.

The fix is structural rather than persuasive. The model picks a LETTER; the value
comes back from our own record. It never writes the number, so it cannot write a
different one. What remains is choosing the wrong fact, which is a smaller and
more tractable problem than the one it replaces.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mpe_lkg.reasoning import (
    _ambiguous,
    _compress,
    _relations,
    _select,
    describes,
    unsupported_numbers,
)

SECONDS = ("seconds in 54 weeks", Fraction(32659200))
DAYS = ("days in 54 weeks", Fraction(378))
DAY_SECONDS = ("seconds in a day", Fraction(86400))


class Chooser:
    """A chat backend that always picks the letter it was told to."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    def stream(self, messages, max_tokens, schema=None):
        self.prompts.append(messages[-1]["content"])
        self.schemas.append(schema)
        yield f'{{"answer": "{self.answer}"}}'


class Embedder:
    """Similarity by shared words, with no hashing at all.

    Two earlier versions of this double were wrong in ways that failed the gate
    tests while the code under test was fine: `hash()` is randomised per process,
    and a 512-bucket sha1 still collided, scoring "why is the sky blue" at 0.35
    against "seconds in 54 weeks". A test double that invents similarity tests
    nothing, so the vocabulary is built from the batch itself and collisions are
    impossible.
    """

    def embed(self, texts):
        import numpy as np

        def words(text: str) -> set[str]:
            return {w.strip("?.,!").lower() for w in text.split()
                    if len(w.strip("?.,!")) > 2}

        every = sorted({w for text in texts for w in words(text)})
        index = {word: i for i, word in enumerate(every)}
        vectors = []
        for text in texts:
            vector = np.zeros(max(len(every), 1), dtype=np.float32)
            for word in words(text):
                vector[index[word]] = 1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.array(vectors)


class TestTheValueComesFromTheRecord:
    def test_the_chosen_letter_returns_our_own_fraction(self):
        """The whole point: what it chose and what is reported cannot drift apart."""
        chat = Chooser("A")
        chosen = _select(chat, Embedder(), "How many seconds are there in 54 weeks?",
                         [SECONDS, DAY_SECONDS])
        assert chosen is not None
        label, value = chosen
        assert value == Fraction(32659200)
        assert isinstance(value, Fraction)

    def test_the_model_is_only_ever_offered_letters(self):
        """A number it cannot type is a number it cannot get wrong.

        The literature's warning about constrained decoding is that wrapping the
        REASONING in a schema costs accuracy. This is a separate, short second
        call whose only job is selection, which is the shape that is safe.
        """
        chat = Chooser("B")
        _select(chat, Embedder(), "How many seconds are there in 54 weeks?",
                [SECONDS, DAY_SECONDS])
        allowed = chat.schemas[0]["properties"]["answer"]["enum"]
        assert allowed == ["A", "B", "NONE"]
        assert not any(char.isdigit() for char in "".join(allowed))

    def test_none_falls_back_rather_than_inventing(self):
        assert _select(Chooser("NONE"), Embedder(), "How many seconds in 54 weeks?",
                       [SECONDS]) is None

    def test_an_answer_outside_the_enum_is_refused(self):
        # A backend that ignores the schema must not be trusted by accident.
        assert _select(Chooser("Z"), Embedder(), "How many seconds in 54 weeks?",
                       [SECONDS]) is None


class TestTheGate:
    """The one change that could make the app WORSE, so it is tested, not assumed.

    An enum forces a choice. Asked "what is the capital of France?" with a settled
    number in front of it, the model chose the number 3 times out of 3 -- and 3
    out of 3 again when the options carried their labels. So the code decides
    whether a numeric answer is on the table; NONE is the second line, not the
    first.
    """

    @pytest.mark.parametrize(
        "question",
        ["What is the capital of France?", "Why is the sky blue?", "Who wrote Hamlet?"],
    )
    def test_prose_questions_never_reach_the_model(self, question: str):
        chat = Chooser("A")
        assert _select(chat, Embedder(), question, [SECONDS, DAY_SECONDS]) is None
        assert chat.prompts == [], "the model must not even be asked"

    def test_a_matching_question_does_reach_it(self):
        chat = Chooser("A")
        assert _select(chat, Embedder(), "How many seconds in 54 weeks?",
                       [SECONDS]) is not None
        assert chat.prompts

    def test_no_facts_means_no_call(self):
        chat = Chooser("A")
        assert _select(chat, Embedder(), "How many seconds in 54 weeks?", []) is None
        assert chat.prompts == []


class TestWhatMayBeOffered:
    @pytest.mark.parametrize(
        ("label", "usable"),
        [
            ("seconds in 54 weeks", True),
            ("parts made by Machine B", True),
            # Measured: asked what the result IS, the model wrote these.
            ("604800/7", False),
            ("(86400*378)", False),
            ("seconds", False),
        ],
    )
    def test_only_descriptions_may_be_offered(self, label: str, usable: bool):
        assert describes(label) is usable

    def test_a_label_that_fits_two_values_identifies_neither(self):
        """Measured: a run labelled both machines' totals 'total_parts'.

        The choice was then a coin toss, and it came up wrong. A label that does
        not discriminate is worse than no label, because it looks like one.
        """
        facts = [("total parts", Fraction(3801755)), ("total parts", Fraction(3169068))]
        assert _ambiguous(facts) == {"total parts"}
        assert _compress([*facts, SECONDS]) == [SECONDS]

    def test_duplicates_collapse(self):
        assert _compress([SECONDS, SECONDS, DAY_SECONDS]) == [SECONDS, DAY_SECONDS]

    def test_the_offer_is_bounded(self):
        many = [(f"value number {i} of the thing", Fraction(i)) for i in range(30)]
        assert len(_compress(many)) <= 8


class TestRelations:
    def test_comparisons_are_settled_here_rather_than_by_the_model(self):
        """Comparing 3801755 with 3169068 is a Fraction comparison, not a judgement."""
        text = _relations([Fraction(3801755), Fraction(3169068)], ["A", "B"])
        assert "A > B" in text
        assert _relations([Fraction(1), Fraction(2)], ["A", "B"]).strip().endswith("A < B")
        assert "A = B" in _relations([Fraction(5), Fraction(5)], ["A", "B"])

    def test_one_fact_has_nothing_to_compare(self):
        assert _relations([Fraction(1)], ["A"]) == ""


class TestTheResultIgnoreDetector:
    """A measurement, not a rewrite: it turns a silent wrong answer into a count."""

    def test_a_number_no_tool_computed_is_flagged(self):
        assert unsupported_numbers("There are 10080 seconds.", [SECONDS]) == ["10080"]

    def test_a_computed_number_is_not_flagged_however_it_is_written(self):
        assert unsupported_numbers("There are 32,659,200 seconds.", [SECONDS]) == []
        assert unsupported_numbers("32659200 seconds", [SECONDS]) == []

    def test_short_numbers_are_left_alone(self):
        # Step counts, years and small quantities are ordinary prose.
        assert unsupported_numbers("It took 3 steps over 2 tries in 1989.", [SECONDS]) == []


class TestNoExampleLeaks:
    """A concrete example in a prompt comes back as an answer.

    Measured: the schema said 'like (17/100)*250' and a run answered a
    fifteen-digit multiplication with "Calculator_verification: 42.5". Another
    answered it with "Second in 23 week: 13910400", the example from the
    conversion field. The model was not reasoning badly -- it was copying the
    only concrete numbers it had been shown.
    """

    def test_no_prompt_shows_the_model_a_number_it_could_copy(self):
        import re

        from mpe_lkg.backends import STEP_SCHEMA
        from mpe_lkg.reasoning import ANGLE_STEP_PROMPT, ANSWER_PROMPT, SYSTEM_PROMPT

        texts = [SYSTEM_PROMPT, ANSWER_PROMPT, ANGLE_STEP_PROMPT]
        texts += [str(field.get("description", ""))
                  for field in STEP_SCHEMA["properties"].values()]
        for text in texts:
            # A number bound to a noun about the OUTPUT ("700 characters") is an
            # instruction and cannot be mistaken for the question's data. A bare
            # one is something to copy, and that is what leaked.
            bare = re.findall(r"\d[\d.,/*+()-]{2,}(?!\s*(?:characters|steps|sentences))",
                              text)
            assert not bare, f"copyable numbers {bare} in: {text[:90]}"
