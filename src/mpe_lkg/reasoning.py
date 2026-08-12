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

from .arithmetic import (
    as_text,
    convert,
    correction,
    evaluate,
    product_unit,
    question_conversion,
    readable,
    restate,
)
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

NEXT_STEP_PROMPT = (
    "Continue. Give the next step, or set next_action to 'final_answer' if the "
    "question is settled."
)

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


# How deep the decomposition may go. Each level multiplies the leaf count, so
# depth 2 at seven angles is up to 49 things to work through and depth 3 is 343 --
# far past any useful run. The bound is on DEPTH rather than on total steps
# because a depth limit is a statement about how far a question is taken apart,
# while a step cap just truncates wherever the budget happens to run out.
MAX_DEPTH = 3

# What one leaf is worth in wall clock. Measured: a step against a 3B-4B model on
# this machine takes three to six seconds, so this is the observed cost with room
# for a slow one rather than a guess.
SECONDS_PER_LEAF = 8.0

# A one-word part is a leaf without asking. This started at three words, which
# was wrong and silently disabled the whole feature: an angle name IS a few words
# by design ("atmospheric water mass"), so every part was a leaf by length and
# nothing was ever split. The cheap guard swallowed the thing it was guarding.
LEAF_WORDS = 1

# Asked whether a part "can be settled in a single step", the model said yes to
# everything -- including "atmospheric water mass", which is a humidity times a
# volume. That is a request for permission to stop, and it always grants it.
#
# So the question is turned around into the first-principles one: what has to be
# KNOWN before this can be worked out? A part is a leaf when the answer is
# nothing, which the model can say honestly instead of being asked to justify
# more work.
SPLIT_PROMPT = (
    "Working question: {question}\n\n"
    "One part of it: {part}\n\n"
    "What must be KNOWN before this part can be worked out? List the quantities "
    "or facts it is built from -- at most {n}, each a few words.\n"
    "If it rests on nothing else -- a figure that can simply be looked up, stated "
    "or measured directly -- reply with an empty list.\n"
    "Do not list the part itself, and do not list anything that is not needed to "
    "produce it.\n"
    'Reply as JSON: {{"angles": ["...", "..."]}}'
)


SUBQUESTION_PROMPT = (
    "Question: {question}\n\n"
    "Name {n} different questions whose answers, together, would let you answer "
    "this one. Each must be a COMPLETE question that stands on its own and could "
    "be handed to someone who has not seen the original.\n"
    "They should approach the problem from different directions rather than being "
    "steps of one method, so that if one leads nowhere the others still do.\n"
    'Reply as JSON: {{"questions": ["...?", "...?"]}}'
)

SUBQUESTION_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}

# How far a sub-question may drift from the one that spawned it. Below this it is
# not a way into the problem, it is a different problem -- and recursion turns a
# small drift at the top into an unrelated tree at the bottom.
MIN_RELEVANCE = 0.45

# And how similar two questions may be before the second is the first again.
# Asking a question already asked is the loop this whole structure can fall into,
# and at depth it does not look like a loop, it looks like progress.
SAME_QUESTION = 0.93


def subquestions(chat, embedder, question: str, want: int = 5,
                 asked: list | None = None) -> list[str]:
    """Questions that between them answer this one, filtered by the graph.

    Two guards, and neither is the model's to apply. It cannot tell that it has
    drifted, because each step looks reasonable from the step before, and it
    cannot tell that it is repeating, because it does not hold the earlier
    questions. Embeddings hold both.
    """
    try:
        raw = "".join(chat.stream(
            [{"role": "user", "content": SUBQUESTION_PROMPT.format(question=question, n=want)}],
            400, schema=SUBQUESTION_SCHEMA,
        ))
    except BackendError:
        return []

    candidates = []
    for item in extract_json(raw).get("questions") or []:
        text = " ".join(str(item).split())[:160]
        if text and text.lower() != question.lower():
            candidates.append(text)
    if not candidates:
        return []

    try:
        vectors = embedder.embed([question, *(asked or []), *candidates])
    except BackendError:
        return candidates[:want]

    root = vectors[0]
    seen = list(vectors[1:1 + len(asked or [])])
    kept = []
    for text, vector in zip(candidates, vectors[1 + len(asked or []):], strict=True):
        if similarity(root, vector) < MIN_RELEVANCE:
            continue          # drifted off the problem
        if any(similarity(vector, other) > SAME_QUESTION for other in seen):
            continue          # asked already, one way or another
        seen.append(vector)
        kept.append(text)
    return kept[:want]


def plan_tree(chat, question: str, want: int = 7, *, depth: int = 2,
              _part: str = "", _seen: set | None = None) -> list[dict]:
    """Break a question down, and break the parts down again, to first principles.

    A flat plan can only ever be as fine as the model's first pass. Asking each
    part whether it is a single step of work, and splitting it again when it is
    not, is what lets a large question end up as eighty small ones instead of
    eight vague ones.

    The decision is made ONCE PER PART, in a structured call, rather than in every
    reasoning step. That is deliberate: model-authored control signals inside the
    step loop were measured unreliable in this repository -- asked what a value
    was, it answered "604800/7" -- and a planning call has nothing else to do.

    Returns a list of ``{"angle": str, "parts": [...]}``, nested to ``depth``.
    """
    seen = _seen if _seen is not None else set()
    if depth <= 0:
        return []

    if not _part:
        names = plan_angles(chat, question, want)
    elif len(_part.split()) <= LEAF_WORDS:
        # Short enough to be one piece of work. Asking would cost a call to be
        # told what the length already says.
        return []
    else:
        names = _split(chat, question, _part, want)

    tree = []
    for name in names:
        key = name.lower().strip()
        # A part that repeats one already planned is the model going in a circle,
        # and recursion turns a circle into an avalanche.
        if key in seen:
            continue
        seen.add(key)
        tree.append({
            "angle": name,
            "parts": plan_tree(chat, question, want, depth=depth - 1,
                               _part=name, _seen=seen),
        })
    return tree


def _split(chat, question: str, part: str, want: int) -> list[str]:
    """The parts of one part, or nothing if it is already a single step of work."""
    try:
        raw = "".join(chat.stream(
            [{"role": "user", "content": SPLIT_PROMPT.format(
                part=part, question=question, n=want)}],
            300, schema=ANGLE_SCHEMA,
        ))
    except BackendError:
        return []
    angles = extract_json(raw).get("angles") or []
    out, seen = [], set()
    for angle in angles:
        text = " ".join(str(angle).split())[:80]
        key = text.lower()
        if text and key not in seen and key != part.lower():
            seen.add(key)
            out.append(text)
    return out[:want]


def leaves(tree: list[dict], _trail: tuple = ()) -> list[tuple[str, tuple]]:
    """Every piece of work in the tree, depth first, with the path that reached it.

    A branch that was split contributes its LEAVES and not itself: the parent is
    the question those leaves answer between them, so working it as well would be
    doing the same thing twice at two levels of detail.
    """
    out = []
    for node in tree:
        trail = (*_trail, node["angle"])
        if node["parts"]:
            out.extend(leaves(node["parts"], trail))
        else:
            out.append((node["angle"], trail))
    return out


def _angle_prompt(angles: list[str], trails: list[tuple], index: int) -> str:
    """What to ask for this piece of work, and what it is a piece OF.

    A leaf three levels down reads as a non sequitur on its own -- "the seconds in
    an hour" is not obviously part of anything. Carrying the trail costs a line
    and keeps the step anchored to the question it serves.
    """
    angle = angles[index]
    trail = trails[index] if index < len(trails) else ()
    if len(trail) > 1:
        return ANGLE_STEP_PROMPT.format(
            angle=f"{angle}\n(this is part of: {' -> '.join(trail[:-1])})")
    return ANGLE_STEP_PROMPT.format(angle=angle)


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
    # How far to take the question apart. 1 is the flat plan that shipped: one
    # round of angles, one step each. Above 1 each part is asked whether it is a
    # single piece of work, and split again when it is not -- which is what lets a
    # large question become eighty small ones rather than eight vague ones.
    depth: int = 1,
    check_arithmetic: bool = True,
    # OFF by default, and the default is a judgement about WHICH MODEL, not about
    # the design. Measured on the full battery, both directions:
    #
    #   llama3.2:3b   9/20 -> 7/20 correct, computed-but-unused 2 -> 7   WORSE
    #   qwen3:4b     33/40 -> 33/40 correct, computed-but-unused 5 -> 1   even,
    #                and it removes four fifths of the failure it exists for
    #
    # On a capable model it is free and does the job: the answer cannot be a
    # number no tool computed, because the model returns a letter and the value
    # comes from our record. On a weaker one it trades writing the wrong number
    # for choosing the wrong fact, which is a worse trade.
    #
    # Off, because this package picks whatever model is installed and llama3.2:3b
    # is a common one. Turn it on with a model that scores well on grounding:
    #
    #     reason(..., select_answer=True)
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
    # A question that is itself one conversion is settled before the model says
    # anything: the exact value goes into the facts the synthesis prefers and
    # the selection can offer, and the reasoning runs as usual around an anchor
    # instead of a hope. Measured need: asked seconds-in-N-weeks twelve times,
    # the model asserted a bare unchecked number in eight of them.
    if check_arithmetic:
        upfront = question_conversion(prompt)
        if upfront is not None:
            text, exact, label = upfront
            converted.append(text)
            labelled.append((label, exact))
            conversions += 1
            yield {"type": "convert", "step": 0, "request": prompt[:80], "result": text}
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
    trails: list[tuple] = []
    if decompose:
        started = time.time()
        if depth > 1:
            tree = plan_tree(chat, prompt, decompose, depth=min(depth, MAX_DEPTH))
            found = leaves(tree)
            angles = [name for name, _ in found]
            trails = [trail for _, trail in found]
            if tree:
                # The budget follows the plan. A tree with 43 leaves against a
                # 120-second wall clock is 43 pieces of work truncated at
                # whichever one the clock lands on -- which is the same as not
                # having planned. Taking a question apart and then refusing to
                # spend the time on the parts is the worst of both.
                max_steps = max(max_steps, len(angles) + 2)
                time_budget = max(time_budget, len(angles) * SECONDS_PER_LEAF)
                yield {"type": "tree", "tree": tree, "leaves": len(angles),
                       "budget": round(time_budget)}
        else:
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

            # A user turn before every step, always. Without a plan this branch
            # used to add nothing, so the conversation became system, user, then
            # assistant after assistant -- malformed, and only silently tolerated.
            # Found in the browser the first time a run used qwen3, which rejects
            # it outright: HTTP 400, "Cannot have 2 or more assistant messages at
            # the end of the list". llama3.2 had been accepting it all along.
            if angles and step_number <= len(angles):
                messages.append({
                    "role": "user",
                    "content": _angle_prompt(angles, trails, step_number - 1),
                })
            elif angles and not final_answer:
                # The plan is walked; nothing is added by asking for more.
                break
            elif step_number > 1:
                messages.append({"role": "user", "content": NEXT_STEP_PROMPT})

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
                    if check_arithmetic:
                        synthesised, repaired = repair_sums(synthesised)
                        for statement in repaired:
                            sums_corrected += 1
                            yield {"type": "repaired", "statement": statement}
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

EXPLORE_PROMPT = (
    "{question}\n\n"
    "These smaller questions have already been worked out, and their answers are "
    "established:\n{findings}\n\n"
    "Use those figures to answer the question above. They are the inputs -- do not "
    "look for others and do not re-derive them. If they genuinely do not settle "
    "it, say exactly which further quantity is missing."
)

# What one branch is worth. Recursion multiplies, so the budget is spent
# breadth-first from the root: a run cut short then has whole answers rather than
# a half-explored corner.
BRANCH_BUDGET = 45.0

# Breadth NARROWS with depth: 5 at the top, then 4, then 3, then 2. Held flat it
# is 5^d -- 625 questions four deep -- and the fifth question at depth four is
# never the one that mattered. Tapered it is 5*4*3*2 = 120, which is a large
# exploration that still ends.
MIN_BREADTH = 2

# How many times a level may re-open after seeing its own answers. One is enough
# to act on what the work turned up; more and the budget goes to widening rather
# than to answering.
REOPENINGS = 1


def _followup(question: str, findings: list[tuple[str, str]]) -> str:
    """The question again, with what has been learned, so the next split knows it."""
    learned = "\n".join(f"  {q} -> {a}" for q, a in findings)
    return (f"{question}\n\nWhat has been established so far:\n{learned}\n\n"
            "What does this leave unanswered?")


def breadth_at(level: int, top: int) -> int:
    """How many ways to split at this depth. One fewer each level, never below two."""
    return max(top - level, MIN_BREADTH)


def explore(
    prompt: str,
    *,
    chat: ChatBackend,
    embedder: EmbeddingBackend,
    breadth: int = 5,
    depth: int = 1,
    budget: float = 300.0,
    _asked: list[str] | None = None,
    _level: int = 0,
    **kwargs,
) -> Iterator[dict]:
    """Answer by answering smaller questions, each as a run of its own.

    The difference from ``depth=`` inside ``reason`` is what a part gets to be. A
    leaf in a plan is a line in one long transcript, and forty-six of them
    measured badly: the leaves are too alike, the repeat detector stops the run,
    and the synthesis cannot assemble that many fragments. Here each part is a
    COMPLETE question with a run and an answer of its own, so what comes back is
    forty-six answers rather than forty-six fragments -- and an answer, unlike a
    fragment, says what it is.

    Both guards live in ``subquestions``: drift from the parent, and a question
    already asked. Neither is the model's to apply, and at depth a repeat does not
    look like a loop, it looks like progress.
    """
    started = time.time()
    asked = list(_asked or [prompt])

    if depth <= 0 or budget < BRANCH_BUDGET:
        yield from reason(prompt, chat=chat, embedder=embedder, **kwargs)
        return

    here = breadth_at(_level, breadth)
    parts = subquestions(chat, embedder, prompt, here, asked=asked)
    if not parts:
        # Nothing worth splitting into is a finding, not a failure: the question
        # is already the size of one run.
        yield from reason(prompt, chat=chat, embedder=embedder, **kwargs)
        return

    yield {"type": "branch", "level": _level, "question": prompt, "parts": parts}
    asked.extend(parts)

    findings: list[tuple[str, str]] = []
    for index, part in enumerate(parts, 1):
        left = budget - (time.time() - started)
        if left < BRANCH_BUDGET:
            # Say what was dropped. A silent truncation reads as "explored
            # everything" when it did not.
            yield {"type": "budget", "level": _level, "dropped": len(parts) - index + 1}
            break

        answer = ""
        for event in explore(
            part, chat=chat, embedder=embedder, breadth=breadth,
            depth=depth - 1, budget=min(left, budget / max(len(parts), 1)),
            _asked=asked, _level=_level + 1, **kwargs,
        ):
            if event["type"] == "final":
                answer = event["content"]
            # Sub-runs stream too, tagged with their level so a reader can see
            # which question a step belongs to.
            yield {**event, "level": _level + 1, "of": part}
        if answer:
            findings.append((part, answer))
            yield {"type": "finding", "level": _level, "question": part, "answer": answer}

    if not findings:
        yield from reason(prompt, chat=chat, embedder=embedder, **kwargs)
        return

    # Loop back. A plan made before any work is a guess about what the work will
    # need, and the answers routinely raise something the plan could not have
    # known to ask. Without this the exploration can only ever be as good as its
    # first guess.
    #
    # It cannot run away: `asked` holds every question so far, and subquestions()
    # drops anything already asked or drifted -- which is why re-opening is safe
    # here and would not be with a model deciding when to stop.
    for opening in range(REOPENINGS):
        left = budget - (time.time() - started)
        if left < BRANCH_BUDGET:
            break
        raised = subquestions(chat, embedder, _followup(prompt, findings),
                             max(here - 1, MIN_BREADTH), asked=asked)
        if not raised:
            break
        yield {"type": "reopened", "level": _level, "round": opening + 1,
               "questions": raised}
        asked.extend(raised)
        for extra in raised:
            left = budget - (time.time() - started)
            if left < BRANCH_BUDGET:
                break
            answer = ""
            for event in explore(
                extra, chat=chat, embedder=embedder, breadth=breadth,
                depth=depth - 1, budget=min(left, budget / max(len(raised), 1)),
                _asked=asked, _level=_level + 1, **kwargs,
            ):
                if event["type"] == "final":
                    answer = event["content"]
                yield {**event, "level": _level + 1, "of": extra}
            if answer:
                findings.append((extra, answer))
                yield {"type": "finding", "level": _level, "question": extra,
                       "answer": answer}

    # The assembly is a REASONING RUN, not a chat call. Measured: as a bare call
    # it was handed the Earth's surface area, the atmosphere's height and the
    # density of water vapour, and answered "cannot be estimated using the
    # provided information" -- it had every figure and would not multiply them.
    # Run as reason() it gets the arithmetic gate, the unit graph and the exact
    # evaluator, which is the whole point of having built them.
    # Restated in SI before composing. Measured: the same estimate came back as
    # 4.08e10, 4.08e16 and 4.08e7 kilograms across three runs, with an identical
    # mantissa each time -- the numbers were multiplied correctly and the exponent
    # was guessed from the units. Removing the guess beats asking for care.
    listed = "\n".join(f"  {q}\n    -> {restate(a)}" for q, a in findings)
    # And the unit the answer must carry, worked out rather than guessed:
    # m^2 * m * kg/m^3 is kg. Measured, this was the last guess left -- given the
    # figures in SI the model composed 4.08e16 correctly every time and then wrote
    # "grams" where the answer is kilograms.
    unit = product_unit(" ".join(a for _, a in findings))
    if unit:
        listed += f"\n\nMultiplying those quantities gives an answer in {unit}."
    answer = ""
    for event in reason(EXPLORE_PROMPT.format(question=prompt, findings=listed),
                        chat=chat, embedder=embedder, **kwargs):
        if event["type"] == "final":
            answer = event["content"]
        elif event["type"] in ("step", "calc", "convert"):
            yield {**event, "level": _level, "assembling": True}

    if not answer:
        return

    ids = [f"Q{i}" for i in range(1, len(findings) + 1)]
    labels = [q[:20] for q, _ in findings]
    try:
        vectors = embedder.embed([q for q, _ in findings])
        graph = serialize_graph_data(build_graph(ids, labels, vectors, top_k=2))
    except BackendError:
        graph = {"nodes": [], "edges": []}

    yield {
        "type": "final",
        "content": answer,
        "graph": graph,
        "path_data": {"strongest_path": ids, "path_weights": [], "avg_similarity": 0.0},
    }
    yield {
        "type": "done",
        "total_time": round(time.time() - started, 2),
        "branches": len(findings),
        "asked": len(asked),
        "level": _level,
    }

CHECK_PROMPT = (
    "Question: {question}\n\n"
    "First answer:  {first}\n"
    "Second answer: {second}\n\n"
    "These were worked out independently. Do they agree on the ANSWER -- the same "
    "value, the same conclusion -- setting aside wording, rounding and how much "
    "detail each gives?\n"
    "Judge it on this in particular: {lens}\n"
    "If they disagree, name the single quantity or claim they disagree about, in a "
    "few words. If they agree, leave it empty.\n"
    'Reply as JSON: {{"agree": true, "disagreement": ""}}'
)

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "agree": {"type": "boolean"},
        "disagreement": {"type": "string"},
    },
    "required": ["agree", "disagreement"],
}

# Two answers this close are the same answer differently worded, and asking a
# model to adjudicate them wastes a call to be told so.
SAME_ANSWER = 0.94

# Rounds of exploration before a disagreement is reported as a disagreement. Two
# is the minimum that can agree at all; beyond four the budget is better spent
# saying what is unresolved than looking again.
MAX_ROUNDS = 4


def settle(
    prompt: str,
    *,
    chat: ChatBackend,
    embedder: EmbeddingBackend,
    rounds: int = 3,
    breadth: int = 4,
    depth: int = 1,
    budget: float = 300.0,
    **kwargs,
) -> Iterator[dict]:
    """Answer only when two independent explorations agree.

    The rule this exists for: no single model call decides the answer. Every
    answer here is explored -- broken into questions, each answered by its own
    run -- and then explored AGAIN. Agreement between two independent runs is the
    finishing condition. One run's synthesis is a proposal, not a verdict.

    A disagreement is not resolved by asking a model which it prefers, because
    that is the guess this is built to avoid. It is resolved by exploring again,
    with the disagreement named as the thing to settle. If the rounds run out the
    disagreement is REPORTED, which is a worse-looking answer and a more honest
    one.

    Loops are held off by embeddings, as everywhere else here: an answer that
    matches one already given is agreement rather than a new round, and
    ``subquestions`` refuses to re-ask what has been asked.
    """
    started = time.time()
    rounds = max(2, min(rounds, MAX_ROUNDS))
    per_round = budget / rounds

    answers: list[str] = []
    vectors: list = []
    focus = ""

    for round_number in range(1, rounds + 1):
        left = budget - (time.time() - started)
        if left <= 0:
            break

        question = prompt if not focus else (
            f"{prompt}\n\nThe unresolved point is: {focus}. Settle that in particular."
        )
        yield {"type": "round", "round": round_number, "of": rounds, "focus": focus}

        answer = ""
        for event in explore(question, chat=chat, embedder=embedder, breadth=breadth,
                             depth=depth, budget=min(left, per_round), **kwargs):
            if event["type"] == "final":
                answer = event["content"]
            elif event["type"] not in ("done",):
                yield {**event, "round": round_number}
        if not answer:
            continue

        try:
            vector = embedder.embed([answer])[0]
        except BackendError:
            vector = None

        for earlier, earlier_vector in zip(answers, vectors, strict=True):
            same = (vector is not None and earlier_vector is not None
                    and similarity(vector, earlier_vector) >= SAME_ANSWER)
            tally = None
            if not same:
                tally = vote(chat, prompt, earlier, answer)
                yield {"type": "vote", "round": round_number, **tally}
            if same or (tally and tally["agreed"]):
                # How agreement was reached, not just that it was. Two answers
                # that matched word for word were never put to a vote, and
                # reporting that as "0 of 0 checks" reads as a failed vote.
                yield {"type": "agreed", "round": round_number, "answer": answer,
                       "by": "wording" if same else "vote",
                       "votes": (tally or {}).get("agree", 0),
                       "of": len((tally or {}).get("ballots", []))}
                yield _final(answer, round_number, started, agreed=True)
                return

        answers.append(answer)
        vectors.append(vector)
        if len(answers) > 1:
            tally = vote(chat, prompt, answers[-2], answers[-1])
            yield {"type": "vote", "round": round_number, **tally}
            focus = tally["about"] or focus
            yield {"type": "deviation", "round": round_number, "about": focus,
                   "votes": tally["disagree"], "of": len(tally["ballots"])}

            # A disagreement is where it matters most whether either side is on
            # thin ice, so the probe runs here rather than on every answer. It is
            # EVIDENCE against the vote's OPINION, and only the arithmetic probe
            # is allowed to settle anything: its truth comes from the evaluator,
            # while consistency is the model agreeing with itself, which a
            # confidently memorised wrong answer does too.
            supported = None
            for candidate in (answers[-2], answers[-1]):
                report = edge(chat, embedder, prompt, candidate)
                yield {"type": "probe", "round": round_number, "answer": candidate,
                       **{k: v for k, v in report.items() if k != "checks"},
                       "checks": report["checks"][:6]}
                if report["decisive"] and not report["at_edge"]:
                    supported = candidate if supported is None else None
                elif report["decisive"] and report["at_edge"] and supported is None:
                    supported = ""      # this one is out; the other may stand

            if supported:
                # One side is held up by checks nothing can argue with. That is
                # not a majority overruling a minority, it is arithmetic.
                yield {"type": "settled_by_probe", "round": round_number,
                       "answer": supported}
                yield _final(supported, round_number, started, agreed=True)
                return

    # No two rounds agreed. Saying so beats picking one, which would be exactly
    # the single-model verdict this is built to avoid.
    if not answers:
        yield from reason(prompt, chat=chat, embedder=embedder, **kwargs)
        return

    unresolved = focus or "the answers did not converge"
    body = "\n".join(f"  round {i}: {a}" for i, a in enumerate(answers, 1))
    yield {"type": "unresolved", "about": unresolved, "answers": answers}
    yield _final(
        f"Unresolved after {len(answers)} independent explorations, which disagree "
        f"about {unresolved}.\n{body}",
        len(answers), started, agreed=False)


# How many independent checks decide whether two answers agree. One checker is a
# single call with a veto, and a call that happens to read the question narrowly
# can send a settled answer round again -- or wave a real disagreement through.
# Odd, so a majority always exists.
VOTERS = 3

# What each voter is asked to weigh. Same question, different ground, so three
# voters are three readings rather than the same reading three times.
LENSES = (
    "Do they give the same VALUE, allowing for rounding and units?",
    "Do they reach the same CONCLUSION, whatever numbers they show?",
    "Would someone acting on the first do the same thing as someone acting on the second?",
)


def vote(chat, question: str, first: str, second: str, voters: int = VOTERS) -> dict:
    """Ask several independent checks whether two answers agree, and count them.

    A tally rather than a verdict, and reported rather than resolved internally:
    the point of exploring twice is that no single call decides, and replacing one
    explorer's guess with one checker's guess would give the decision straight
    back. Each voter reads through a different lens, so a unanimous verdict means
    three ways of looking rather than one looked at three times.
    """
    ballots = []
    for index in range(max(1, voters)):
        lens = LENSES[index % len(LENSES)]
        try:
            raw = "".join(chat.stream(
                [{"role": "user", "content": CHECK_PROMPT.format(
                    question=question, first=first, second=second, lens=lens)}],
                200, schema=CHECK_SCHEMA,
            ))
        except BackendError:
            continue
        parsed = extract_json(raw)
        ballots.append({
            "lens": lens,
            "agree": parsed.get("agree") is True,
            "about": " ".join(str(parsed.get("disagreement", "")).split())[:120],
        })

    agree = sum(1 for b in ballots if b["agree"])
    against = [b["about"] for b in ballots if not b["agree"] and b["about"]]
    return {
        "ballots": ballots,
        "agree": agree,
        "disagree": len(ballots) - agree,
        # A majority, not a veto. One dissenting reading does not overturn two.
        "agreed": bool(ballots) and agree * 2 > len(ballots),
        "about": against[0] if against else "the answer itself",
    }


def repair_sums(answer: str) -> tuple[str, list[str]]:
    """Fix wrong arithmetic in a final answer, deterministically.

    The step gate retries a step whose sums are wrong, but the SYNTHESIS was
    never checked -- and it does sums. Measured: a final answer read "378 x
    86,400 = 32,356,800 seconds", right expression, wrong product, and nothing
    looked. The claim regex catches exactly this shape, the evaluator knows the
    exact value, so the repair is a substitution and not a rewrite: every
    rendering of the wrong number is replaced with the exact one, digit
    boundaries respected so 42 inside 1421 is left alone.
    """
    wrong = arithmetic_errors(answer)
    repaired = []
    for claim in wrong:
        exact = readable(claim.exact)
        variants = {claim.stated}
        if claim.stated.lstrip("-").isdigit():
            variants.add(f"{int(claim.stated):,}")
        for variant in variants:
            # The lookahead blocks only a CONTINUATION of the number -- ".5" or
            # ",000" -- not a sentence-ending period. The first version blocked
            # any ".", so "the answer is 32,356,800." kept its wrong number
            # while the equation beside it was fixed.
            answer = re.sub(
                rf"(?<![\d.]){re.escape(variant)}(?![.,]?\d)", exact, answer)
        repaired.append(f"{claim.expression} = {exact}, stated as {claim.stated}")
    return answer, repaired


def _final(answer: str, rounds: int, started: float, *, agreed: bool) -> dict:
    return {
        "type": "final",
        "content": answer,
        "graph": {"nodes": [], "edges": []},
        "path_data": {"strongest_path": [], "path_weights": [], "avg_similarity": 0.0},
        "agreed": agreed,
        "rounds": rounds,
        "total_time": round(time.time() - started, 2),
    }

NEIGHBOUR_PROMPT = (
    "Question: {question}\n"
    "Answer given: {answer}\n\n"
    "Write {n} questions that someone who genuinely knows that answer would also "
    "be able to answer, going from the obvious towards the obscure -- the last "
    "should be something only a specialist would know.\n"
    "Each must have a short, definite, factual answer. No opinions, no "
    "explanations, nothing that depends on the date.\n"
    'Reply as JSON: {{"questions": ["...?", "...?"]}}'
)

# Two answers to the same question this close are the same answer. Below it the
# model gave two different answers to one question, which is the signal.
CONSISTENT = 0.88

# The probe MUST sample. Measured, four askings each:
#
#   temperature 0.0   capital of France 1.000   street north of Rue Cler 1.000
#   temperature 1.0   capital of France 1.000   street north of Rue Cler 0.537
#
# At temperature 0 greedy decoding returns the same string whether the model
# knows the answer or is inventing it, so the probe reports perfect confidence in
# pure fabrication. The variation IS the measurement, and a caller running the
# rest of the loop at temperature 0 would otherwise silently disable it.
PROBE_TEMPERATURE = 1.0

# How many times each probe is asked. Two can disagree; three says which way.
ASKINGS = 3


def perturbations(question: str, limit: int = 3) -> list[tuple[str, Fraction]]:
    """The same arithmetic question with the numbers changed, and the true answers.

    The sharpest probe available, because it needs no judgement at all: the truth
    comes from the exact evaluator, not from a model. A model that answers
    "17 * 250" correctly and "18 * 251" wrongly was recalling, not calculating,
    and the first answer is worth what the second one is.
    """
    numbers = re.findall(r"\d+(?:\.\d+)?", question)
    if len(numbers) < 2:
        return []

    out = []
    # Nudged, then scaled: a near neighbour catches recall, a large one catches a
    # method that only works at the size it was learned on.
    for factor, shift in ((1, 1), (1, 3), (1000, 0)):
        changed = question
        parts = []
        for text in numbers:
            try:
                value = Fraction(text)
            except (ValueError, ZeroDivisionError):
                return []
            moved = value * factor + shift
            moved = moved if moved.denominator != 1 else Fraction(int(moved))
            parts.append((text, as_text(moved)))
        for old, new in parts:
            changed = re.sub(rf"(?<!\d){re.escape(old)}(?!\d)", new, changed, count=1)
        expression = _expression_of(changed)
        if expression is None:
            continue
        truth = evaluate(expression)
        if truth is not None:
            out.append((changed, truth))
        if len(out) >= limit:
            break
    return out


def _expression_of(question: str):
    """The arithmetic a question is asking for, if it is asking for arithmetic."""
    numbers = re.findall(r"\d+(?:\.\d+)?", question)
    if len(numbers) != 2:
        return None
    lowered = question.lower()
    for words, operator in (
        (("multiplied", "times", "product", "*"), "*"),
        (("plus", "sum", "added", "add", "+"), "+"),
        (("minus", "subtract", "less", "-"), "-"),
        (("divided", "over", "/"), "/"),
    ):
        if any(word in lowered for word in words):
            return f"{numbers[0]}{operator}{numbers[1]}"
    return None


def neighbourhood(chat, question: str, answer: str, want: int = 4) -> list[str]:
    """Questions a genuine knower of this answer could also answer.

    Ordered from the obvious towards the obscure on purpose: "Paris is the
    capital" is worth little on its own, because it is in every corpus a hundred
    thousand times. Whether the same model can name the river, the region and a
    street says whether there is knowledge behind it or a single memorised string.
    """
    try:
        raw = "".join(chat.stream(
            [{"role": "user", "content": NEIGHBOUR_PROMPT.format(
                question=question, answer=answer, n=want)}],
            400, schema=SUBQUESTION_SCHEMA,
        ))
    except BackendError:
        return []
    out, seen = [], set()
    for item in extract_json(raw).get("questions") or []:
        text = " ".join(str(item).split())[:160]
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out[:want]


def _steady(chat, embedder, question: str, askings: int = ASKINGS) -> tuple[bool, str]:
    """Does the model give the same answer to this every time?

    The whole probe rests on this, and on one property: no ground truth is needed.
    At the edge of what it knows a model does not fall silent, it confabulates --
    and confabulates DIFFERENTLY each time. Agreement with itself is therefore
    evidence, and disagreement is proof.
    """
    # Sampled, not greedy -- see PROBE_TEMPERATURE. Set on a copy so the caller's
    # own settings are untouched, and restored even if a call raises.
    previous = getattr(chat, "temperature", None)
    if previous is not None:
        chat.temperature = PROBE_TEMPERATURE

    answers = []
    for _ in range(askings):
        try:
            raw = "".join(chat.stream(
                [{"role": "user", "content":
                  f"{question}\nAnswer in a few words. If you do not know, say "
                  f"exactly: I do not know."}], 60))
        except BackendError:
            continue
        answers.append(" ".join(raw.split()).strip())
    if previous is not None:
        chat.temperature = previous
    if not answers:
        return False, ""
    if any("do not know" in a.lower() for a in answers):
        return False, answers[0]

    try:
        vectors = embedder.embed(answers)
    except BackendError:
        return len(set(answers)) == 1, answers[0]
    worst = min(similarity(vectors[i], vectors[j])
                for i in range(len(vectors)) for j in range(i + 1, len(vectors)))
    return worst >= CONSISTENT, answers[0]


def edge(chat, embedder, question: str, answer: str, *, want: int = 4) -> dict:
    """Is this answer near the edge of what the model knows?

    Two probes, chosen by what the question is:

    * **Arithmetic** -- the same sum with the numbers moved, graded against the
      exact evaluator. Needs no judgement and admits no argument.
    * **Anything else** -- questions a knower would also answer, each asked
      several times. Not graded for correctness, which we do not have, but for
      whether the model agrees with itself.

    Returns a fraction held and the checks behind it. A low score does not make
    the answer wrong; it says the answer is not supported by anything around it,
    which is the difference between knowing Paris and having read it.
    """
    checks = []

    for changed, truth in perturbations(question):
        try:
            raw = "".join(chat.stream(
                [{"role": "user", "content": f"{changed}\nReply with the number alone."}],
                60))
        except BackendError:
            continue
        got = raw.replace(",", "").replace(" ", "")
        checks.append({
            "probe": changed,
            "kind": "arithmetic",
            "held": as_text(truth) in got,
            "expected": as_text(truth),
        })

    if not checks:
        for probe in neighbourhood(chat, question, answer, want):
            steady, said = _steady(chat, embedder, probe)
            checks.append({"probe": probe, "kind": "consistency",
                           "held": steady, "said": said[:80]})

    held = sum(1 for c in checks if c["held"])
    # Which probe ran, because they are not worth the same. Arithmetic is graded
    # against the exact evaluator and settles the matter; consistency is the model
    # agreeing with itself, which a confidently memorised wrong answer also does.
    kind = "arithmetic" if any(c["kind"] == "arithmetic" for c in checks) else (
        "consistency" if checks else "none")
    return {
        "checks": checks,
        "kind": kind,
        "decisive": kind == "arithmetic",
        "held": held,
        "asked": len(checks),
        # No checks is not confidence. An unprobed answer scores zero, so a
        # caller cannot mistake "nothing was asked" for "everything held".
        "support": (held / len(checks)) if checks else 0.0,
        "at_edge": bool(checks) and held * 2 <= len(checks),
    }
