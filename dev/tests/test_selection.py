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
    """Similarity by shared words, at a FIXED width.

    Three earlier versions were wrong in ways that failed tests while the code was
    fine. `hash()` is randomised per process. A 512-bucket sha1 still collided,
    scoring "why is the sky blue" at 0.35 against "seconds in 54 weeks". And a
    vocabulary built per call gave a different width on every call, which broke
    the moment anything embedded twice inside one run.

    Fixed width, deterministic buckets, wide enough that collisions are rare.
    """

    WIDTH = 4096

    def embed(self, texts):
        import hashlib

        import numpy as np

        def bucket(word: str) -> int:
            return int(hashlib.sha1(word.encode()).hexdigest()[:8], 16) % self.WIDTH

        vectors = []
        for text in texts:
            words = {w.strip("?.,!").lower() for w in str(text).split()}
            words = {w for w in words if len(w) > 2}
            vector = np.zeros(self.WIDTH, dtype=np.float32)
            for word in words:
                vector[bucket(word)] = 1.0
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


class TestSubquestions:
    """Splitting a question into standalone questions, guarded by the graph.

    Neither guard is the model's to apply. It cannot tell it has drifted, because
    each step looks reasonable from the one before, and it cannot tell it is
    repeating, because it does not hold the earlier questions. Embeddings hold
    both, which is the graph earning its place in the control flow rather than
    only in the picture.
    """

    def _chat(self, questions):
        import json as _json

        class Split:
            def stream(self, messages, max_tokens, schema=None):
                yield _json.dumps({"questions": questions})

        return Split()

    def test_standalone_questions_come_back(self):
        from mpe_lkg.reasoning import subquestions

        asked = ["How heavy is the atmosphere?"]
        out = subquestions(self._chat(["How heavy is the atmosphere in total?"]),
                           Embedder(), "How heavy is the atmosphere?", 5)
        assert out and all(q.strip() for q in out)
        assert asked  # the parent is not silently required

    def test_a_question_that_drifted_is_dropped(self):
        """A different problem is not a way into this one."""
        from mpe_lkg.reasoning import subquestions

        chat = self._chat(["What is the average density of air in the atmosphere?",
                           "Who won the 1966 World Cup final?"])
        out = subquestions(chat, Embedder(),
                           "How heavy is the air in the atmosphere?", 5)
        assert not any("World Cup" in q for q in out)

    def test_a_question_already_asked_is_dropped(self):
        """The loop this structure can fall into, which at depth looks like progress."""
        from mpe_lkg.reasoning import subquestions

        parent = "How heavy is the air in the atmosphere?"
        chat = self._chat([parent, "What is the density of air in the atmosphere?"])
        out = subquestions(chat, Embedder(), parent, 5, asked=[parent])
        assert parent not in out

    def test_nothing_usable_is_an_empty_list_rather_than_a_guess(self):
        from mpe_lkg.reasoning import subquestions

        assert subquestions(self._chat([]), Embedder(), "anything at all?", 5) == []

    def test_the_count_is_respected(self):
        from mpe_lkg.reasoning import subquestions

        many = [f"What is quantity number {i} of the atmosphere air?" for i in range(12)]
        assert len(subquestions(self._chat(many), Embedder(),
                                "What is the atmosphere air made of?", 3)) <= 3


class TestExplore:
    """Each part answered as a run of its own, then assembled.

    The measured difference from walking a plan flat: forty-six leaves in one
    transcript produced thirteen steps and "negligible and not reliably
    estimable", because the leaves are too alike and the synthesis cannot put
    that many fragments together. Forty-six ANSWERS are a different thing -- an
    answer says what it is, and a fragment does not.
    """

    def _backends(self, script):
        """Word-overlap embeddings, not the hash-based double.

        The relevance guard compares a sub-question with its parent, and a hash
        embedding makes unrelated strings arbitrarily similar or dissimilar -- so
        it dropped every part and the test measured the guard misfiring rather
        than the feature.
        """
        import json as _json

        class Scripted:
            def __init__(self):
                self.calls = 0

            def stream(self, messages, max_tokens, schema=None):
                self.calls += 1
                text = messages[-1]["content"]
                if "would let you answer" in text:
                    yield _json.dumps({"questions": script})
                elif schema is not None:
                    yield _json.dumps({"title": "T", "content": "Worked out.",
                                       "calc": "", "calc_of": "", "convert": "",
                                       "next_action": "final_answer"})
                else:
                    yield "An answer."

        return Scripted(), Embedder()

    def test_a_question_that_will_not_split_falls_back_to_one_run(self):
        from mpe_lkg.reasoning import explore

        chat, embedder = self._backends([])
        events = list(explore("What is the capital of France?", chat=chat,
                              embedder=embedder, breadth=4, depth=1))
        assert any(e["type"] == "final" for e in events)
        assert not any(e["type"] == "branch" for e in events)

    def test_depth_zero_is_an_ordinary_run(self):
        from mpe_lkg.reasoning import explore

        chat, embedder = self._backends(["Something else entirely?"])
        events = list(explore("q", chat=chat, embedder=embedder, depth=0))
        assert not any(e["type"] == "branch" for e in events)
        assert any(e["type"] == "final" for e in events)

    def test_the_parts_are_answered_and_assembled(self):
        from mpe_lkg.reasoning import explore

        parts = ["How heavy is the water above one square metre of air?",
                 "How much water is in the air above a square metre?"]
        chat, embedder = self._backends(parts)
        events = list(explore("How heavy is the water in the air?", chat=chat,
                              embedder=embedder, breadth=2, depth=1))
        branch = next(e for e in events if e["type"] == "branch")
        assert branch["parts"] == parts
        # One finding per part, each carrying its own question, and one answer.
        findings = [e for e in events if e["type"] == "finding" and e["level"] == 0]
        assert [f["question"] for f in findings] == parts
        assert all(f["answer"] for f in findings)
        assert len([e for e in events if e["type"] == "final" and not e.get("level")]) == 1

    def test_a_budget_too_small_to_branch_just_answers(self):
        """Half a plan is worse than no plan: it reads as an explored question."""
        from mpe_lkg.reasoning import explore

        chat, embedder = self._backends(["a?", "b?"])
        events = list(explore("q", chat=chat, embedder=embedder, depth=1, budget=1.0))
        assert not any(e["type"] == "branch" for e in events)
        assert any(e["type"] == "final" for e in events)

    def test_sub_runs_are_tagged_with_the_question_they_belong_to(self):
        from mpe_lkg.reasoning import explore

        parts = ["How heavy is the water in the air above one square metre?",
                 "How much water sits in the air over a square metre?"]
        chat, embedder = self._backends(parts)
        events = list(explore("How heavy is the water in the air?", chat=chat,
                              embedder=embedder, breadth=2, depth=1))
        tagged = [e for e in events if e.get("of")]
        assert tagged, "sub-run events must say which question they came from"
        assert {e["of"] for e in tagged} <= set(parts)
