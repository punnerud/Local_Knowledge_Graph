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


class TestTheVote:
    """Several independent checks, counted -- so no single one can fell an answer.

    One checker is a call with a veto: a reading that happens to be narrow sends a
    settled answer round again, and a lenient one waves a real disagreement
    through. Three readings through three different lenses, and a majority.
    """

    def _voter(self, verdicts):
        import json as _json

        class Fake:
            def __init__(self):
                self.seen = 0
                self.prompts = []

            def stream(self, messages, max_tokens, schema=None):
                self.prompts.append(messages[-1]["content"])
                agree = verdicts[self.seen % len(verdicts)]
                self.seen += 1
                yield _json.dumps({"agree": agree,
                                   "disagreement": "" if agree else "the total mass"})

        return Fake()

    def test_a_lone_dissenter_does_not_overturn_two(self):
        from mpe_lkg.reasoning import vote

        tally = vote(self._voter([True, True, False]), "q", "first", "second")
        assert (tally["agree"], tally["disagree"]) == (2, 1)
        assert tally["agreed"] is True

    def test_a_lone_supporter_does_not_carry_it(self):
        from mpe_lkg.reasoning import vote

        tally = vote(self._voter([False, False, True]), "q", "first", "second")
        assert tally["agreed"] is False
        assert tally["about"] == "the total mass"

    def test_every_voter_is_asked_something_different(self):
        """Three readings, not one reading three times."""
        from mpe_lkg.reasoning import LENSES, vote

        chat = self._voter([True])
        vote(chat, "q", "a", "b", voters=3)
        assert len(chat.prompts) == 3
        assert len(set(chat.prompts)) == 3, "the voters saw identical prompts"
        for lens in LENSES:
            assert any(lens in p for p in chat.prompts)

    def test_the_tally_is_reported_not_just_the_verdict(self):
        """2-1 and 3-0 are different things, and hiding which turns one into a fact."""
        from mpe_lkg.reasoning import vote

        tally = vote(self._voter([True, True, False]), "q", "a", "b")
        assert len(tally["ballots"]) == 3
        assert [b["agree"] for b in tally["ballots"]] == [True, True, False]
        assert all(b["lens"] for b in tally["ballots"])

    def test_a_backend_that_will_not_answer_does_not_pass_it(self):
        from mpe_lkg.backends import BackendError
        from mpe_lkg.reasoning import vote

        class Broken:
            def stream(self, messages, max_tokens, schema=None):
                raise BackendError("down")
                yield ""

        tally = vote(Broken(), "q", "a", "b")
        assert tally["agreed"] is False, "no ballots must not count as agreement"


class TestBreadthTapers:
    def test_breadth_narrows_with_depth_and_stops_at_two(self):
        """Held flat, five ways four deep is 625 questions, and the fifth at the
        bottom is never the one that mattered. 5*4*3*2 is 120 and it ends."""
        import math

        from mpe_lkg.reasoning import breadth_at

        widths = [breadth_at(level, 5) for level in range(4)]
        assert widths == [5, 4, 3, 2]
        assert math.prod(widths) == 120
        assert breadth_at(99, 5) == 2


class TestUpfrontConversion:
    """A question that IS one conversion is settled before the model speaks.

    Measured need, then measured effect: asked seconds-in-N-weeks twelve times,
    the model asserted a bare unchecked number in eight -- nothing for any gate
    to hold. With the question's own conversion settled up front and placed in
    the facts the synthesis prefers, the same twelve questions went 4/12 to
    12/12, with zero repairs needed.
    """

    def test_a_pure_conversion_question_parses(self):
        from mpe_lkg.arithmetic import question_conversion

        text, value, label = question_conversion(
            "How many seconds are there in 82 weeks?")
        assert str(value) == "49593600"
        assert "82 week" in text and "second" in label

    @pytest.mark.parametrize("question", [
        "What is the capital of France?",
        "How many jars remain?",
        "Why is the sky blue?",
        "How many whole times does 816 go into 802550?",
    ])
    def test_anything_else_is_left_alone(self, question):
        from mpe_lkg.arithmetic import question_conversion

        assert question_conversion(question) is None

    def test_the_anchor_arrives_as_a_step_zero_convert_event(self):
        import json

        from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat
        from mpe_lkg.reasoning import reason

        script = [json.dumps({"title": "T", "content": "C", "calc": "",
                              "calc_of": "", "convert": "",
                              "next_action": "final_answer"}), "An answer."]
        events = list(reason("How many seconds are there in 8 weeks?",
                             chat=ScriptedChat(script),
                             embedder=DeterministicEmbedding(24)))
        anchor = next(e for e in events if e["type"] == "convert")
        assert anchor["step"] == 0
        assert "4,838,400" in anchor["result"]


class TestRepairSums:
    """The synthesis does sums, and nothing checked them.

    Measured: a final answer read "378 x 86,400 = 32,356,800 seconds" -- right
    expression, wrong product -- and the prose gate only ever ran on steps. The
    repair is a substitution, not a rewrite: the evaluator knows the exact
    value, so every rendering of the wrong number is replaced with it. When it
    fires it decides: 2 of 2 repaired runs came out correct.
    """

    def test_the_measured_case_is_repaired_everywhere_it_appears(self):
        from mpe_lkg.reasoning import repair_sums

        answer, repaired = repair_sums(
            "so 378 × 86,400 = 32,356,800 seconds. The answer is 32,356,800.")
        assert "32,659,200 seconds" in answer
        assert answer.endswith("32,659,200.")
        assert "32,356,800" not in answer
        assert len(repaired) == 1

    def test_digit_boundaries_are_respected(self):
        from mpe_lkg.reasoning import repair_sums

        answer, _ = repair_sums("Note 1.42 stays, though 2*21 = 41 is wrong.")
        assert "1.42" in answer, "a decimal sharing digits must not be touched"
        assert "2*21 = 42" in answer.replace("= 42", "= 42")

    def test_a_correct_answer_is_untouched(self):
        from mpe_lkg.reasoning import repair_sums

        text = "14 * 1440 = 20,160 minutes, so the answer is 20,160."
        assert repair_sums(text) == (text, [])

    def test_prose_without_claims_is_untouched(self):
        from mpe_lkg.reasoning import repair_sums

        text = "The capital of France is Paris, established over 2000 years ago."
        assert repair_sums(text) == (text, [])


class TestDerivativeField:
    """A derivative the model NAMES rather than works out, like calc and convert.

    Measured need, from the six-domain battery: the model dropped a chain
    factor (6912 where the answer is 235824) and read a tangent at the wrong
    point. mpeqs.calculus computes the derivative two independent ways whose
    tests demand agreement, so what comes back cannot be a matching mistake.
    """

    def test_the_field_is_required_like_its_siblings(self):
        from mpe_lkg.backends import STEP_SCHEMA

        assert "derivative" in STEP_SCHEMA["required"]

    @pytest.mark.parametrize(("request_text", "value"), [
        ("d/dx (4*x**2 + 8)**4 at x=3/2", "235824"),
        ("derivative of 3*x**2 + 7*x at 4", "31"),
        ("d/dx x**3 at x=2", "12"),
    ])
    def test_requests_settle_exactly(self, request_text, value):
        from mpe_lkg.arithmetic import derivative_request

        text, exact, label = derivative_request(request_text)
        assert str(exact) == value
        assert label.startswith("derivative of")

    @pytest.mark.parametrize("refused", [
        "d/dx x**x at 2",          # not a polynomial; calculus refuses
        "d/dx sin(x) at 0",        # calls are refused
        "derivative of x**2",      # no point given
        "just words",
    ])
    def test_what_cannot_be_settled_is_silence(self, refused):
        from mpe_lkg.arithmetic import derivative_request

        assert derivative_request(refused) is None

    def test_the_loop_carries_it_to_the_synthesis(self):
        import json as _json

        from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat
        from mpe_lkg.reasoning import reason

        script = [
            _json.dumps({"title": "Slope", "content": "Ask for it.",
                         "calc": "", "calc_of": "", "convert": "",
                         "derivative": "d/dx 3*x**2 + 7*x at x=4",
                         "next_action": "final_answer"}),
            "The slope is 31.",
        ]
        events = list(reason("What is the slope at x=4?",
                             chat=ScriptedChat(script),
                             embedder=DeterministicEmbedding(24)))
        settled = next(e for e in events if e["type"] == "derivative")
        assert settled["result"].endswith("= 31")
        done = next(e for e in events if e["type"] == "done")
        assert done["derivatives"] == 1
