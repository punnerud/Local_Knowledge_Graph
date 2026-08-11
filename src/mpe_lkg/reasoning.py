"""The step-by-step reasoning loop.

Yields plain dicts. The Flask layer turns them into server-sent events, and tests
consume them directly. Every path out of this generator ends in either a ``final``
or an ``error`` event, which is the property that fixes the reported symptom of a
page where nothing ever appears.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from fractions import Fraction

from .arithmetic import as_text, convert, correction, evaluate, readable
from .arithmetic import errors as arithmetic_errors
from .backends import STEP_SCHEMA, BackendError, ChatBackend, EmbeddingBackend
from .graph import build_graph, edge_weight_spread, serialize_graph_data, strongest_path
from .novelty import assess, similarity

MAX_STEPS = 20
# Wall clock, not a step count. A hard step cap stops a run that is still getting
# somewhere and lets a slow one run forever; a budget bounds the wait a person
# actually experiences, and lets a productive run keep going until it is spent.
TIME_BUDGET = float(os.environ.get("LKG_TIME_BUDGET", "120"))
# No floor. A fixed minimum forced padding: in a reported transcript the model
# reached the answer at step 4 and was pushed on with "you have given 4 of 5
# steps", producing three more that added nothing. Length is decided by whether
# anything new is still arriving.
MIN_STEPS = 0
# Consecutive repeats that end a run. One is a stumble; two in a row is the model
# out of ideas, and continuing past that is how a transcript fills with restatement.
DRY_LIMIT = 2
MAX_STEP_CHARS = 700
# How many times a single step may be re-asked before we take what we were given.
# The original code retried without bound and without incrementing the step
# counter, so a model that kept answering too long, or kept trying to finish early,
# held the loop forever while the browser sat waiting on a stream that never spoke.
MAX_RETRIES_PER_STEP = 3
# A wrong sum is worth redoing, but a model that keeps writing the same wrong sum
# will do so however often it is told. Bounded, like every other retry here.
MAX_ARITHMETIC_RETRIES = 3

# MEASURED, AND THE SHORT VERSION LOST. This 233-token prompt is resent on every
# call and reads like shouting, so it looked like an obvious saving. Three separate
# terser rewrites all scored worse on the same 20 questions -- 82 to 88 percent
# against 97 -- and a control with a step floor showed it is the WORDING, not the
# amount of reasoning: at 3.7 steps against this prompt's 3.5, the short version
# still lost. It costs about 1100 prompt tokens per run and buys nine points.
SYSTEM_PROMPT = (
    "You are an expert AI assistant that explains your reasoning step by step. For each step, "
    "provide a title that describes what you're doing in that step, along with the content. "
    "Decide if you need another step or if you're ready to give the final answer. Respond in "
    "JSON format with 'title', 'content', and 'next_action' (either 'continue' or "
    "'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF "
    "YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE "
    "EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN "
    "YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. "
    "WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO "
    "SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. "
    "USE BEST PRACTICES. Keep the content of each step under "
    f"{MAX_STEP_CHARS} characters. "
    "If a step needs a unit conversion, put the WHOLE conversion in a 'convert' field in "
    "one line, as '<quantity> <unit> to <unit>', using the quantity from THIS question. "
    "Any number of intermediate units is handled for you in one exact answer. Never split "
    "a conversion into steps and never multiply conversion factors together yourself -- "
    "that is the single most common way this goes wrong. Leave it empty if this question "
    "involves no units. "
    "If a step relies on a calculation, ALSO put that calculation in a 'calc' field as a "
    "bare arithmetic expression with no words and no equals sign, built ONLY from numbers "
    "that appear in this question. It is evaluated exactly and the result is given back to "
    "you, so you never have to do the sum yourself. Leave it empty when there is no sum."
)

# Kept so the result stays reproducible rather than becoming folklore. Pass it as
# system_prompt= to reproduce the losing arm.
SHORT_SYSTEM_PROMPT = (
    "You reason one step at a time. Each step has a short title naming what you are doing, "
    "and content doing it.\n"
    "Do not answer in your first step. First check what the question assumes and whether it "
    "is well posed, then look for a reading under which your obvious answer would be wrong.\n"
    "When a further step would only restate something you have already said, set next_action "
    "to 'final_answer'.\n"
    f"Keep each step under {MAX_STEP_CHARS} characters."
)


def extract_json(text: str) -> dict:
    """Best-effort parse of a step, for backends that ignore the response schema."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fall back to the last brace-delimited object in the text.
    for candidate in reversed(re.findall(r"\{[^{}]*\}", cleaned)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {"title": "", "content": cleaned, "next_action": "continue"}


def _collect(chat: ChatBackend, messages: list[dict], max_tokens: int) -> str:
    return "".join(chat.stream(messages, max_tokens, schema=STEP_SCHEMA))


def _short_title(title: str, content: str, fallback: str) -> str:
    """A label short enough to read on a node, without spending an extra LLM call.

    The original made a second model call per step purely to shorten a title. That
    doubled the request count for something a slice does just as well.
    """
    text = (title or content or fallback).strip()
    text = " ".join(text.split())
    if len(text) <= 20:
        return text
    cut = text[:20]
    # Prefer a word boundary if one is close to the end.
    if " " in cut[10:]:
        cut = cut[: cut.rindex(" ")]
    return cut


ANSWER_PROMPT = (
    "Question: {question}\n\n"
    "This is the strongest thread through your own reasoning:\n{thread}\n\n"
    "Answer the question now, in one or two sentences. State the answer itself. Do not "
    "describe what you considered, do not hedge, and do not add caveats. If the question "
    "cannot be answered as asked, say plainly why."
)


DECOMPOSE_PROMPT = (
    "Question: {question}\n\n"
    "Before answering, list the distinct things worth checking. Aim for {n} of them. Each is "
    "a short angle name of a few words -- an assumption to test, a quantity to work out, a "
    "reading of the question that might change the answer, or a way to check the result.\n"
    "They must not overlap. Reply as JSON: {{\"angles\": [\"...\", \"...\"]}}"
)

ANGLE_SCHEMA = {
    "type": "object",
    "properties": {"angles": {"type": "array", "items": {"type": "string"}}},
    "required": ["angles"],
}

ANGLE_STEP_PROMPT = (
    "Now do this one: {angle}\n"
    "Work it out concretely -- do not restate the plan or what you have already covered."
)


def plan_angles(chat, question: str, want: int = 7) -> list[str]:
    """Ask the model to break the question into angles worth checking.

    Steps that each answer a named angle differ from one another by construction,
    which is what a run of eight steps needs in order to be worth drawing. Left to
    itself over that length a model restates.
    """
    try:
        raw = "".join(chat.stream(
            [{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question, n=want)}],
            400, schema=ANGLE_SCHEMA,
        ))
    except BackendError:
        return []

    parsed = extract_json(raw)
    angles = parsed.get("angles") or []
    seen, out = set(), []
    for angle in angles:
        text = " ".join(str(angle).split())[:80]
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out[: want + 2]


def _spine(node_ids: list[str], labels: list[str], vectors, top_k: int = 2) -> list[int]:
    """Indices of the steps on the strongest path, in order.

    This is the graph earning its place: of everything the model said, these are the
    steps that hang together, and they are what the answer gets written from.
    """
    if len(node_ids) < 2:
        return list(range(len(node_ids)))
    graph = serialize_graph_data(build_graph(node_ids, labels, vectors, top_k=top_k))
    path, _, _ = strongest_path(graph)
    if not path:
        return list(range(len(node_ids)))
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    return sorted(position[p] for p in path if p in position)


def _redirect(novelty, step_texts: list[str]) -> str:
    """Name the ground already covered, and ask for somewhere else.

    A coverage map rather than a prohibition: "do not repeat yourself" gives the
    model nothing to aim at, while a list of what is already done leaves the
    unexplored part by subtraction.
    """
    covered = "\n".join(f"- {t[:90]}" for t in step_texts[-6:])
    what = novelty.describe()
    return (
        f"That step {what}. You have already covered:\n{covered}\n\n"
        "Give a step that goes somewhere none of those go: a different method, a "
        "different assumption to test, or a part of the question not yet touched. "
        "If there is genuinely nothing left to explore, set next_action to "
        "final_answer."
    )


# How close a fact's label must sit to the question before its value may be
# offered as the answer. THE GATE IS NOT THE MODEL'S JOB, and that is measured:
# asked "what is the capital of France?" with a settled number in front of it, the
# model chose the number 3 times out of 3 -- and 3 out of 3 again when the options
# carried their labels. A forced choice among numbers produces a number. So the
# code decides whether a numeric answer is even on the table.
#
# 0.30 sits well below the labelled-fact questions and well above the prose ones;
# it is checked by test rather than asserted here.
FACT_RELEVANCE = 0.30

# The most letters to offer. Measured candidate counts are median 3, max 8 per
# run, so this rarely bites -- it is a bound, not a policy.
MAX_CHOICES = 8

SELECT_PROMPT = (
    "Question: {question}\n\n"
    "Each of these was computed exactly:\n{facts}\n{relations}\n"
    "Which letter is the answer to the question as asked? "
    "Answer NONE if no letter is."
)


def _relations(values: list[Fraction], keys: list[str]) -> str:
    """How the values compare, settled here rather than by the model.

    Comparing 3801755 against 3169068 is exactly the kind of thing a small model
    gets wrong, and it is a Fraction comparison. Handing over the answer removes
    the question.
    """
    lines = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            sign = ">" if values[i] > values[j] else "<" if values[i] < values[j] else "="
            lines.append(f"  {keys[i]} {sign} {keys[j]}")
    return "\nHow they compare:\n" + "\n".join(lines) + "\n" if lines else ""


def describes(label: str) -> bool:
    """Is this a description of a value, or an expression wearing one as a hat?

    Measured: asked what a result IS, the model wrote "604800/7" and "Weeks to
    seconds via minutes". The first is the sum again, the second is a method. A
    label that does not name a quantity cannot identify it, and offering one as a
    candidate answer is offering a wrong answer with a confident face.

    The conversions never fail this, because their labels are built here rather
    than written by a model -- which is why the two questions that came out right
    were both answered from a conversion.
    """
    words = [w for w in re.findall(r"[A-Za-z]+", label) if len(w) > 1]
    if len(words) < 2:
        return False
    # More operator than word is an expression, whatever it is called.
    return len(re.findall(r"[+\-*/=^()]", label)) <= 1


def _ambiguous(facts: list[tuple[str, Fraction]]) -> set[str]:
    """Labels that fit more than one value, and so identify neither.

    Measured: a run labelled both machines' totals "total_parts", and the choice
    between them was then a coin toss the relations could not settle -- the model
    picked the smaller when asked for the larger. A label that does not
    discriminate is worse than no label, because it looks like one.
    """
    seen: dict[str, set] = {}
    for label, value in facts:
        seen.setdefault(label.strip().lower(), set()).add(value)
    return {label for label, values in seen.items() if len(values) > 1}


def _compress(facts: list[tuple[str, Fraction]]) -> list[tuple[str, Fraction]]:
    """Fewer, better-labelled facts. Eight steps of working beat three facts, badly.

    Duplicates go, and so does any fact whose label is a prefix of another's -- a
    step computing "days in 54 weeks" on the way to "seconds in 54 weeks" is
    working, not an answer, and offering it is offering a wrong answer.
    """
    seen: dict[tuple[str, Fraction], None] = {}
    for label, value in facts:
        if label:
            seen.setdefault((label.strip().lower(), value), None)
    unique = [(label, value) for label, value in
              {(lab, val): (lab, val) for lab, val in
               [(f[0].strip(), f[1]) for f in facts if f[0]]}.values()]

    def subsumed(label: str) -> bool:
        low = label.lower()
        return any(low != other.lower() and low in other.lower() for other, _ in unique)

    vague = _ambiguous(unique)
    usable = [(label, value) for label, value in unique
              if describes(label) and label.strip().lower() not in vague]
    kept = [(label, value) for label, value in usable if not subsumed(label)]
    return (kept or usable)[:MAX_CHOICES]


# Numbers small enough to be ordinary prose rather than a computed result: a count
# of steps, a YEAR, "one of three". Four digits was the first guess and it flagged
# 1989, so five it is -- which still catches 10080, the wrong answer this exists
# to see.
GUARD_MIN_DIGITS = 5


def unsupported_numbers(answer: str, facts: list[tuple[str, Fraction]]) -> list[str]:
    """Long numbers in the answer that no tool actually computed.

    A measurement, not a rewrite. The literature calls this failure Result-Ignore
    -- the final answer states something other than what the tool returned -- and
    detects it exactly this way. Reporting the count turns a silent wrong answer
    into a number we can watch, which is worth more than a silent correction.

    Only the selection path is immune by construction; this watches the other one.
    """
    known = {str(value.numerator) for _, value in facts if value.denominator == 1}
    known |= {as_text(value).replace(".", "") for _, value in facts}
    found = []
    pattern = rf"\d[\d,]{{{GUARD_MIN_DIGITS - 1},}}"
    for literal in re.findall(pattern, answer):
        bare = literal.replace(",", "")
        if bare not in known:
            found.append(bare)
    return found


def _select(chat: ChatBackend, embedder: EmbeddingBackend, question: str,
            facts: list[tuple[str, Fraction]]):
    """Pick the fact that answers the question, and return ITS value.

    The model answers with a letter. It never writes the number, never carries it,
    and never compares two of them -- all three are things it was measured to get
    wrong on fifteen-digit values, and none of them are things it needs to do.
    The value comes back from the record, so what it chose and what is reported
    cannot drift apart.

    Returns None when nothing here answers the question, which leaves the ordinary
    synthesis to run.
    """
    facts = _compress(facts)
    if not facts:
        return None

    # The gate. See FACT_RELEVANCE: an enum forces a choice, so the code has to
    # decide whether a numeric answer is on the table at all.
    try:
        vectors = embedder.embed([question] + [label for label, _ in facts])
    except BackendError:
        return None
    if max(similarity(vectors[0], v) for v in vectors[1:]) < FACT_RELEVANCE:
        return None

    keys = [chr(65 + i) for i in range(len(facts))]
    # Labels only. Showing the values was tried and measured worse (1/6 against
    # 4/6), which fits the design rather than contradicting it: the point of
    # answering by letter is that the model never handles the number, and putting
    # the numbers back in front of it gives it something to be wrong about.
    listed = "\n".join(f"  {k} = {label}"
                       for k, (label, _) in zip(keys, facts, strict=True))
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": [*keys, "NONE"]}},
        "required": ["answer"],
    }
    prompt = SELECT_PROMPT.format(
        question=question, facts=listed,
        relations=_relations([value for _, value in facts], keys))
    try:
        raw = "".join(chat.stream([{"role": "user", "content": prompt}], 40, schema=schema))
    except BackendError:
        return None

    chosen = str(extract_json(raw).get("answer", "")).strip()
    if chosen not in keys:
        return None
    label, value = facts[keys.index(chosen)]
    return label, value


def _synthesise(
    chat: ChatBackend, question: str, thread: list[str],
    settled: list[str] | None = None, converted: list[str] | None = None,
) -> str:
    """One call that turns the thread into an answer. Empty string if it fails."""
    if not thread:
        return ""
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(thread, 1))
    # Sums an exact evaluator has already settled. They are given separately from
    # the thread because they are not the model's opinion and are not up for
    # revision -- and because the step that produced one is often off the spine.
    sums = ""
    if settled:
        unique = list(dict.fromkeys(settled))
        sums = "\n\nThese have been calculated exactly and are correct:\n" + "\n".join(
            f"  {s}" for s in unique
        ) + "\nUse these figures. Do not recompute them."
    # Conversions get their own heading and an instruction to prefer them. The
    # measured failure was not a wrong conversion -- "161 day = 13910400 second"
    # was computed exactly -- but an answer of 10080 written beside it.
    units_block = ""
    if converted:
        units_block = "\n\nThese conversions were done exactly:\n" + "\n".join(
            f"  {c}" for c in converted
        ) + "\nIf one of them answers the question directly, give that number."
    try:
        raw = "".join(chat.stream(
            [{"role": "user", "content": ANSWER_PROMPT.format(
                question=question, thread=numbered) + sums + units_block}],
            300,
        ))
    except BackendError:
        # A failed synthesis must not lose the run: the caller falls back to the
        # last step, which is what this replaced.
        return ""

    answer = " ".join(raw.split()).strip()
    if answer.startswith("{"):
        # Some models answer this prompt in the step schema anyway, having been
        # asked for JSON on every previous turn. Take the content rather than
        # showing the user a serialised object.
        answer = str(extract_json(answer).get("content", answer)).strip()
    return answer


def reason(
    prompt: str,
    *,
    chat: ChatBackend,
    embedder: EmbeddingBackend,
    store=None,
    max_steps: int = MAX_STEPS,
    min_steps: int = MIN_STEPS,
    top_k: int = 2,
    synthesise: bool = True,
    time_budget: float = TIME_BUDGET,
    detect_repeats: bool = True,
    max_novelty_retries: int = 3,
    system_prompt: str = "",
    decompose: int = 0,
    check_arithmetic: bool = True,
    # OFF by default, because it was measured and it lost. See _select: the
    # machinery is sound and the guarantee is real -- the answer cannot be a
    # number no tool computed -- but on llama3.2:3b it picks the wrong FACT more
    # often than the synthesis wrote the wrong number. Full battery: 9/20 for the
    # synthesis against 7/20 for selection, and computed-but-unused rose from 2
    # to 7. The research says the ceiling here is the model's grounding (50.79%
    # for llama3.2:3b against 88.19% for qwen3:4b), so this is worth turning on
    # when the model changes -- and worth leaving off until it does.
    select_answer: bool = False,
) -> Iterator[dict]:
    """Run the reasoning loop, yielding one event dict at a time."""
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    node_ids: list[str] = []
    labels: list[str] = []
    vectors: list = []
    title_vectors: list = []
    step_texts: list[str] = []
    dry_streak = 0
    retries_spent = 0
    retries_that_helped = 0
    sums_checked = 0
    sums_corrected = 0
    # Every sum the run settled exactly, kept with the step it came from. Both
    # halves of that were measured the hard way:
    #
    # Computing a number exactly is worth nothing if it does not reach the answer:
    # the loop settled "20 - 13.5 = 6.5" and still answered 2.50. So they are handed
    # to the synthesis rather than left in the step text.
    #
    # But handing over ALL of them is worse than handing over none. A model
    # exploring eight angles also writes "17*250" and "(17/1)*100", and presenting
    # those under a heading saying they are correct is an invitation to pick one:
    # multi-step accuracy fell to 50% against 100%, on answers like "604800 minutes
    # in a fortnight" -- exactly right, for seconds in a week. So each sum keeps the
    # index and title of its step, and only those on the strongest path are shown.
    settled: list[tuple[int, str, str]] = []
    converted: list[str] = []
    # Facts that can be OFFERED as the answer: a label saying what the value is,
    # and the exact value itself. Kept apart from the display strings because the
    # answer comes from the Fraction, never from text parsed back out of a model.
    labelled: list[tuple[str, Fraction]] = []
    unsupported: list[str] = []
    conversions = 0
    arithmetic_retries = 0
    total_thinking_time = 0.0
    final_answer: str | None = None
    step_events = 0

    def graph_payload() -> tuple[dict, dict | None]:
        graph = build_graph(node_ids, labels, vectors, top_k=top_k)
        serialized = serialize_graph_data(graph)
        path, weights, mean = strongest_path(serialized)
        path_data = (
            {"strongest_path": path, "path_weights": weights, "avg_similarity": mean}
            if path is not None
            else None
        )
        return serialized, path_data

    # When asked to decompose, the plan is made once and then walked. Each step is
    # aimed at a named angle, so the steps differ by construction rather than by
    # hoping a model asked for "another step" finds something new to say.
    angles: list[str] = []
    if decompose:
        started = time.time()
        angles = plan_angles(chat, prompt, decompose)
        total_thinking_time += time.time() - started
        if angles:
            yield {"type": "plan", "angles": angles}

    deadline = time.time() + time_budget
    try:
        while len(node_ids) < max_steps:
            if time.time() > deadline:
                # Out of budget. Stop reasoning and go answer with what there is,
                # rather than truncating mid-thought with nothing to show.
                break
            step_number = len(node_ids) + 1
            step_json = None
            truncated = False

            if angles:
                if step_number <= len(angles):
                    messages.append({
                        "role": "user",
                        "content": ANGLE_STEP_PROMPT.format(angle=angles[step_number - 1]),
                    })
                elif not final_answer:
                    # The plan is walked; nothing is added by asking for more.
                    break

            for attempt in range(MAX_RETRIES_PER_STEP):
                started = time.time()
                raw = _collect(chat, messages, 300)
                total_thinking_time += time.time() - started
                step_json = extract_json(raw)
                content = str(step_json.get("content", "")).strip()

                if len(content) <= MAX_STEP_CHARS:
                    break
                if attempt == MAX_RETRIES_PER_STEP - 1:
                    # Take what we have rather than asking again forever.
                    step_json["content"] = content[:MAX_STEP_CHARS].rstrip() + "..."
                    truncated = True
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": "Your last response was too long. Give the same step again, "
                        f"under {MAX_STEP_CHARS} characters.",
                    }
                )

            assert step_json is not None
            content = str(step_json.get("content", "")).strip()
            if not content:
                content = "The model returned an empty step."
            title = str(step_json.get("title", "")).strip()
            next_action = str(step_json.get("next_action", "continue")).strip()

            # Named before computed. The factor comes from a graph of exact
            # ratios rather than from the model's memory, which is where every
            # measured unit failure came from.
            asked = str(step_json.get("convert", "")).strip()
            if check_arithmetic and asked:
                done = convert(asked)
                if done is not None:
                    text, exact, label = done
                    labelled.append((label, exact))
                    conversions += 1
                    # Kept apart from the sums, and NOT filtered to the spine.
                    # A sum off the strongest path is usually a dead end the model
                    # wandered into; a conversion cannot be wrong -- an unknown unit
                    # or a cross-dimension request refuses rather than answering --
                    # so every one of them is a fact about the question worth having.
                    if text not in converted:
                        converted.append(text)
                    if as_text(exact) not in content.replace(",", ""):
                        content = f"{content} ({text})"
                        step_json["content"] = content
                    yield {
                        "type": "convert",
                        "step": step_number,
                        "request": asked,
                        "result": text,
                    }

            calc = str(step_json.get("calc", "")).strip()
            if check_arithmetic and calc:
                exact = evaluate(calc)
                if exact is not None:
                    sums_checked += 1
                    stated = as_text(exact)
                    settled.append((len(step_texts), title or f"Step {step_number}",
                                    f"{calc} = {readable(exact)}"))
                    # Only a LABELLED value may be offered as the answer later.
                    # "(86400*378) = 32659200" does not say what it is, and offering
                    # unlabelled expressions as candidates scored 0 of 5.
                    described = str(step_json.get("calc_of", "")).strip()
                    if described:
                        labelled.append((described, exact))
                    # The exact value is appended rather than substituted: the
                    # model's own wording stays, and the number it can be held to
                    # sits beside it. The synthesis step reads this.
                    if stated not in content.replace(",", ""):
                        sums_corrected += 1
                        content = f"{content} ({calc} = {stated})"
                        step_json["content"] = content
                    yield {
                        "type": "calc",
                        "step": step_number,
                        "expression": calc,
                        "value": stated,
                    }

            if check_arithmetic:
                wrong = arithmetic_errors(content)
                sums_checked += 1
                if wrong and arithmetic_retries < MAX_ARITHMETIC_RETRIES:
                    # A sum the record can evaluate exactly is not a matter of
                    # opinion. Hand back the corrected figure and let the model
                    # redo the step around it.
                    arithmetic_retries += 1
                    sums_corrected += 1
                    messages.append({"role": "user", "content": correction(wrong)})
                    yield {
                        "type": "arithmetic",
                        "step": step_number,
                        "errors": [c.describe() for c in wrong],
                    }
                    continue

            content_vec = embedder.embed([content])[0]
            title_vec = embedder.embed([title or content[:60]])[0]

            novelty = (
                assess(title_vec, content_vec, title_vectors, vectors, labels)
                if detect_repeats else assess(title_vec, content_vec, [], [], [])
            )

            if novelty.is_repeat and retries_spent < max_novelty_retries:
                # Do not keep the step. Tell the model what ground it has already
                # covered and ask for somewhere else, rather than letting the
                # transcript fill with the same move under a new heading.
                retries_spent += 1
                dry_streak += 1
                messages.append({"role": "user", "content": _redirect(novelty, step_texts)})
                yield {
                    "type": "repeat",
                    "step": step_number,
                    "title": title,
                    "reason": novelty.describe(),
                    "attempt": retries_spent,
                }
                if dry_streak >= DRY_LIMIT:
                    # Twice in a row is the model out of ideas. Continuing is how a
                    # transcript fills with restatement.
                    break
                continue

            if novelty.is_repeat:
                # Out of retries. Keep it, but say so rather than hiding it.
                dry_streak += 1
            else:
                if retries_spent:
                    retries_that_helped += 1
                dry_streak = 0

            node_id = f"Step{step_number}"
            node_ids.append(node_id)
            step_texts.append(content)
            labels.append(f"Step {step_number}: {_short_title(title, content, node_id)}")
            vectors.append(content_vec)
            title_vectors.append(title_vec)
            if store is not None:
                store.add(content, content_vec, model=embedder.describe().get("model", ""))

            messages.append({"role": "assistant", "content": json.dumps(step_json)})
            wants_to_finish = next_action == "final_answer" or "boxed" in content.lower()

            # Angles still unworked. A plain step floor was tried before and it
            # produced padding -- the model reached the answer at step 4, was told
            # it had given 4 of 5 steps, and wrote three that added nothing. The
            # difference here is that each remaining step has a named job rather
            # than a quota to fill, so "keep going" means something specific.
            angles_left = bool(angles) and step_number < len(angles)

            if wants_to_finish and not angles_left and len(node_ids) >= max(min_steps, 1):
                # This step *is* the answer. Announcing it as a step and then again as
                # the final answer would print the same text twice on the page and
                # draw two nodes over identical content.
                final_answer = content
                break

            serialized, path_data = graph_payload()
            step_events += 1
            yield {
                "type": "step",
                "step": step_number,
                "title": title or f"Step {step_number}",
                "content": content,
                "truncated": truncated,
                "graph": serialized,
                "path_data": path_data,
            }

            if wants_to_finish:
                # Nudge once per remaining step, but never without having made
                # progress -- the step counter has already advanced by now.
                messages.append(
                    {
                        "role": "user",
                        "content": f"You have given {len(node_ids)} of {min_steps} steps. Look for "
                        "errors or alternatives in your answer, then continue your reasoning.",
                    }
                )

        if synthesise:
            # Ask the graph's strongest thread for an answer, rather than taking
            # whatever the last step happened to say. Measured on the baseline: the
            # answer to "What is the capital of France?" did not contain the word
            # Paris. It was a footnote about regional capitals, because the last step
            # is where a prompt that rewards exploring alternatives naturally ends.
            spine = _spine(node_ids, labels, vectors, top_k=top_k)
            started = time.time()
            on_spine = set(spine)
            # Sums from abandoned branches are not evidence about the answer. If the
            # spine happens to hold none, showing all of them is no better than
            # showing none, so nothing is shown.
            thread_sums = [f"{title}: {sum_}" for i, title, sum_ in settled if i in on_spine]

            # Selection first. The model picks a LETTER and the value comes back
            # from the record, so what it chose and what is reported cannot drift
            # apart -- which is the whole failure this replaces. The literature
            # calls it Result-Ignore, and it is measured at 30% on an 8B Llama.
            chosen = (_select(chat, embedder, prompt, labelled)
                      if (select_answer and check_arithmetic) else None)
            if chosen is not None:
                label, value = chosen
                selected = as_text(value)
                final_answer = f"{label.capitalize()}: {selected}."
                yield {"type": "selected", "label": label, "value": selected}
            else:
                synthesised = _synthesise(
                    chat, prompt, [step_texts[i] for i in spine], thread_sums, converted
                )
                if synthesised:
                    final_answer = synthesised
                    unsupported = unsupported_numbers(synthesised, labelled)
            total_thinking_time += time.time() - started

        if final_answer is None:
            final_answer = step_texts[-1] if step_texts else "No final answer."

        if node_ids and step_texts and final_answer == step_texts[-1]:
            # The answer is the last step's own text -- either because synthesis was
            # off, or because it failed and this is the fallback. Either way, relabel
            # that node rather than adding a second one holding identical text: two
            # nodes over the same content produce an edge of exactly 1.00 between
            # them, which is the duplicate pair from this project's own screenshot.
            labels[-1] = f"Final Answer: {_short_title('', final_answer, 'final')}"
        else:
            node_ids.append(f"Step{len(node_ids) + 1}")
            labels.append(f"Final Answer: {_short_title('', final_answer, 'final')}")
            vectors.append(embedder.embed([final_answer])[0])
            if store is not None:
                store.add(final_answer, vectors[-1], model=embedder.describe().get("model", ""))

        serialized, path_data = graph_payload()
        yield {
            "type": "final",
            "content": final_answer,
            "graph": serialized,
            "path_data": path_data,
        }
        yield {
            "type": "done",
            "total_time": total_thinking_time,
            "steps": step_events,
            "edge_spread": edge_weight_spread(serialized),
            "sums_checked": sums_checked,
            "sums_corrected": sums_corrected,
            "conversions": conversions,
            # Long numbers in the answer that no tool computed. Zero on the
            # selection path by construction; this is the synthesis path's score.
            "unsupported_numbers": unsupported,
            "novelty_retries": retries_spent,
            "novelty_retries_that_helped": retries_that_helped,
            "embedding": embedder.describe(),
            "chat": chat.describe(),
        }

    except BackendError as exc:
        yield {"type": "error", "message": str(exc), "hint": exc.hint}
    except Exception as exc:  # noqa: BLE001 - the stream must always say what happened
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}", "hint": ""}
