"""Is an answer near the edge of what the model knows?

Two probes, and they are not equally strong. Saying which is which matters more
than the feature does.

**Arithmetic** perturbs the numbers and grades against the exact evaluator. It
needs no judgement and admits no argument: a model that answers 17*250 and fails
18*251 was recalling, not calculating. Measured on qwen3:4b, "12 plus 30" holds
3 of 3 and "17433193048 multiplied by 3565" holds 0 of 3.

**Consistency** asks what a knower would also know, several times, and looks for
the model disagreeing with itself. It has a real limit, measured and not designed
around: a model can be CONSISTENTLY WRONG. Asked for Rue Cler's postal code it
answers 75006 every time, and it is 75007. This probe cannot see that, and a
caller should not be told it can.
"""

from __future__ import annotations

from fractions import Fraction

from mpe_lkg.reasoning import PROBE_TEMPERATURE, _steady, edge, perturbations


class Answers:
    """A backend whose answers vary or do not, and which records its temperature."""

    def __init__(self, replies, temperature=0.0):
        self.replies = replies
        self.temperature = temperature
        self.seen_temperatures = []
        self.calls = 0

    def stream(self, messages, max_tokens, schema=None):
        self.seen_temperatures.append(self.temperature)
        reply = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        yield reply


class Words:
    def embed(self, texts):
        import numpy as np

        vocabulary = sorted({w.lower() for t in texts for w in str(t).split()})
        index = {w: i for i, w in enumerate(vocabulary)}
        out = []
        for text in texts:
            vector = np.zeros(max(len(vocabulary), 1), dtype=np.float32)
            for word in str(text).split():
                vector[index[word.lower()]] = 1.0
            norm = np.linalg.norm(vector)
            out.append(vector / norm if norm else vector)
        return np.array(out)


class TestPerturbations:
    def test_the_numbers_move_and_the_truth_comes_from_the_evaluator(self):
        out = perturbations("What is 12 plus 30?")
        assert out, "an arithmetic question should perturb"
        for changed, truth in out:
            assert changed != "What is 12 plus 30?"
            assert isinstance(truth, Fraction)
        # Nudged by one, and scaled by a thousand: recall and method are
        # different failures and need different probes.
        assert any("13 plus 31" in c for c, _ in out)
        assert any("12000" in c for c, _ in out)

    def test_a_question_with_no_arithmetic_perturbs_into_nothing(self):
        assert perturbations("What is the capital of France?") == []
        assert perturbations("Why is the sky blue?") == []

    def test_the_stated_truth_is_actually_true(self):
        """The probe is worthless if its own answers are wrong."""
        for changed, truth in perturbations("What is 12 plus 30?"):
            numbers = [int(n) for n in __import__("re").findall(r"\d+", changed)]
            assert truth == sum(numbers[:2])


class TestConsistency:
    def test_the_same_answer_every_time_is_steady(self):
        chat = Answers(["Paris"])
        steady, said = _steady(chat, Words(), "What is the capital of France?")
        assert steady is True
        assert said == "Paris"

    def test_a_different_answer_each_time_is_not(self):
        chat = Answers(["Rue des Rosiers", "Place du Luxembourg", "Rue de Grenelle"])
        steady, _ = _steady(chat, Words(), "Which street is north of Rue Cler?")
        assert steady is False

    def test_saying_it_does_not_know_is_not_steady(self):
        chat = Answers(["I do not know."])
        steady, _ = _steady(chat, Words(), "anything?")
        assert steady is False

    def test_the_probe_samples_even_when_the_caller_does_not(self):
        """At temperature 0 the probe measures nothing.

        Measured: greedy decoding returned the same string for the capital of
        France AND for an invented street, both 1.000. The variation is the
        measurement, so a caller running the loop greedily would otherwise
        silently disable it.
        """
        chat = Answers(["Paris"], temperature=0.0)
        _steady(chat, Words(), "What is the capital of France?")
        assert chat.seen_temperatures
        assert all(t == PROBE_TEMPERATURE for t in chat.seen_temperatures)
        assert chat.temperature == 0.0, "the caller's setting must be put back"


class TestEdge:
    def test_arithmetic_is_graded_against_the_truth_not_a_judgement(self):
        chat = Answers(["44", "48", "42000"])
        report = edge(chat, Words(), "What is 12 plus 30?", "42")
        assert report["asked"] == 3
        assert all(c["kind"] == "arithmetic" for c in report["checks"])
        assert report["support"] == 1.0
        assert report["at_edge"] is False

    def test_wrong_answers_to_moved_numbers_place_it_at_the_edge(self):
        chat = Answers(["99999"])
        report = edge(chat, Words(), "What is 12 plus 30?", "42")
        assert report["held"] == 0
        assert report["at_edge"] is True

    def test_nothing_asked_is_not_confidence(self):
        """An unprobed answer scores zero, so nobody reads silence as support."""
        class Empty:
            temperature = 0.7

            def stream(self, messages, max_tokens, schema=None):
                yield '{"questions": []}'

        report = edge(Empty(), Words(), "Why is the sky blue?", "Rayleigh scattering")
        assert report["asked"] == 0
        assert report["support"] == 0.0
        assert report["at_edge"] is False, "unknown is not the same as refuted"


class TestTheProbeAgainstTheVote:
    """Evidence beats opinion, but only the evidence that is actually evidence.

    The arithmetic probe is graded against the exact evaluator, so it settles a
    disagreement outright -- that is not a majority overruling a minority, it is
    arithmetic. The consistency probe is the model agreeing with itself, which a
    confidently memorised wrong answer also does, so it is never allowed to
    decide anything on its own.
    """

    def test_only_the_arithmetic_probe_calls_itself_decisive(self):
        chat = Answers(["44", "48", "42000"])
        assert edge(chat, Words(), "What is 12 plus 30?", "42")["decisive"] is True

    def test_a_consistency_probe_is_never_decisive(self):
        import json

        class Neighbours:
            temperature = 0.7

            def __init__(self):
                self.calls = 0

            def stream(self, messages, max_tokens, schema=None):
                self.calls += 1
                if schema is not None:
                    yield json.dumps({"questions": ["Which river runs through it?"]})
                else:
                    yield "The Seine"

        report = edge(Neighbours(), Words(), "What is the capital of France?", "Paris")
        assert report["kind"] == "consistency"
        assert report["decisive"] is False
        assert report["support"] == 1.0, "it still reports what it found"

    def test_nothing_probed_is_not_decisive_and_not_at_the_edge(self):
        class Silent:
            temperature = 0.7

            def stream(self, messages, max_tokens, schema=None):
                yield '{"questions": []}'

        report = edge(Silent(), Words(), "Why is the sky blue?", "Scattering")
        assert report["decisive"] is False
        assert report["at_edge"] is False
        assert report["support"] == 0.0

    def test_an_answer_that_fails_moved_numbers_is_flagged_however_confident(self):
        """The case this exists for: fluent, agreed upon, and wrong.

        Two explorations can agree on a fifteen-digit product and both be wrong,
        and three voters reading it can all nod. Moving the numbers does not care
        how many agreed.
        """
        chat = Answers(["definitely not the right number"])
        report = edge(chat, Words(), "What is 12 plus 30?", "42")
        assert report["at_edge"] is True
        assert report["decisive"] is True
        assert report["held"] == 0
