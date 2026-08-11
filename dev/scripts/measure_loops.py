#!/usr/bin/env python3
"""Is repetition actually the problem, or does the model drift instead?

This exists to decide whether a loop detector is worth building at all. A
near-identical study elsewhere measured 2 literal repeats in 41 steps -- 4.9 % --
and found the dominant failure was *drift*: new words, no progress, which a loop
detector cannot see by construction. Our own evidence is one pasted transcript,
which is n=1.

Three failure modes are counted separately because they need different fixes:

    REPEAT   the step says something an earlier step already said
    DRIFT    the step says something new-sounding that adds no new content
    NEITHER  a step that actually moves

The first thing this script does is check *itself*. In that other study the first
run reported a 58 % loop rate, and every one of those repeats was the step
extractor mistaking a code fence for a step and then seeing it again. A
measurement of the harness nearly became a finding about the model, so the
normalisation is audited before any of its numbers are believed.

    .venv/bin/python scripts/measure_loops.py [--questions N] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mpe_lkg import backends  # noqa: E402
from mpe_lkg.reasoning import reason  # noqa: E402

# Four kinds, because a loop rate averaged over one kind says nothing about the
# others. The malformed group is where the reported transcript came from.
QUESTIONS = {
    "simple": [
        "What is the capital of France?",
        "How many days are in a leap year?",
        "What is the chemical symbol for gold?",
        "Who wrote the play Hamlet?",
        "What is the largest ocean on Earth?",
        "In what year did the Berlin Wall fall?",
    ],
    "multi_step": [
        "If a train travels 240 km in 3 hours, how long does it take to travel 400 km?",
        "What is 17 percent of 250?",
        "A rectangle is twice as long as it is wide and has a perimeter of 36 cm. What are its sides?",
        "Which is larger, the area of a circle of radius 3 or a square of side 5?",
        "If I buy 3 items at 4.50 each and pay with a 20 note, what is my change?",
        "How many minutes are there in a fortnight?",
    ],
    "malformed": [
        "What is the capital of Oslo?",
        "When did Napoleon invade Australia?",
        "What is the population of the Atlantic Ocean?",
        "Who was the first person to walk on the Sun?",
        "What language do they speak in Antarctica's capital?",
        "How tall is the colour blue?",
    ],
    "ambiguous": [
        "What is the best programming language?",
        "Is a hot dog a sandwich?",
        "What is the meaning of the word 'set'?",
        "How long is a piece of string?",
        "Which came first, the chicken or the egg?",
        "What is the biggest city?",
    ],
}

# noqa: SIM905 -- a 70-word list is readable as prose and unreadable as a literal.
STOPWORDS = frozenset(
    """a an and are as at be but by can could do does for from had has have how i if in into is it
    its may might must not of on or should so than that the their them then there these they this
    to was we were what when where which who why will with would you your about also just more most
    other some such only own same very don now""".split()  # noqa: SIM905
)


def normalise(text: str) -> str:
    """Canonical form for exact-duplicate detection: whitespace and case only.

    Deliberately conservative. Every extra transformation is another way to make
    two distinct steps look identical, which is the failure this script audits for.
    """
    return re.sub(r"\s+", "", text).lower()


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def audit_normalisation(steps: list[dict]) -> dict:
    """Does the normalisation itself manufacture duplicates?

    Compares duplicate counts on the raw text against the normalised text. A large
    gap means the canonical form is collapsing things that differ, and every number
    downstream of it is a measurement of this script rather than of the model.
    """
    raw = [s["content"] for s in steps]
    norm = [normalise(s["content"]) for s in steps]
    raw_dupes = len(raw) - len(set(raw))
    norm_dupes = len(norm) - len(set(norm))
    manufactured = norm_dupes - raw_dupes

    empties = sum(1 for n in norm if not n)
    shortest = min((len(n) for n in norm), default=0)
    return {
        "raw_duplicates": raw_dupes,
        "normalised_duplicates": norm_dupes,
        "manufactured_by_normalisation": manufactured,
        "empty_after_normalisation": empties,
        "shortest_normalised_length": shortest,
        # If normalisation invents duplicates, or empties steps out, the rest of
        # this file is about the harness and not about the model.
        "trustworthy": manufactured == 0 and empties == 0 and shortest > 20,
    }


# Set from the measured distribution over 24 runs, not guessed. The median step
# already sits at 0.846 cosine to some earlier step and the 75th percentile at
# 0.900, so a 0.90 bar flags the more-similar quarter of ordinary steps rather
# than duplicates -- inspection confirmed it: "we need to calculate a percentage"
# followed by "the formula is (17/100)*250" scored 0.9013 and is plain progress.
# At 0.96 the flags are restatements on inspection. The title bar is insensitive:
# exactly the same 3 steps fire anywhere from 0.95 to 0.999, so nothing borders it.
RESTATEMENT_THRESHOLD = 0.96
REPEATED_MOVE_THRESHOLD = 0.95


def classify(steps: list[dict], title_vecs, content_vecs, *,
             move_thr=REPEATED_MOVE_THRESHOLD, restate_thr=RESTATEMENT_THRESHOLD) -> list[dict]:
    """Label every step REPEAT / DRIFT / NEITHER, with the reason."""
    seen_norm: dict[str, int] = {}
    verdicts = []

    for i, step in enumerate(steps):
        reasons = []
        key = normalise(step["content"])
        if key in seen_norm:
            reasons.append(("exact", seen_norm[key], 1.0))
        seen_norm.setdefault(key, i)

        if i:
            tsim = title_vecs[i] @ title_vecs[:i].T
            csim = content_vecs[i] @ content_vecs[:i].T
            if float(tsim.max()) >= move_thr:
                reasons.append(("move", int(tsim.argmax()), float(tsim.max())))
            if float(csim.max()) >= restate_thr:
                reasons.append(("restatement", int(csim.argmax()), float(csim.max())))

        # Drift: new words, no new subject matter. Measured against everything said
        # before, including the question.
        earlier = set().union(*(content_words(s["content"]) for s in steps[:i])) if i else set()
        earlier |= content_words(step.get("question", ""))
        mine = content_words(step["content"])
        novel = mine - earlier
        novel_ratio = len(novel) / max(len(mine), 1)

        kind = "repeat" if reasons else ("drift" if novel_ratio <= 0.15 else "neither")
        verdicts.append({
            "index": i,
            "kind": kind,
            "reasons": reasons,
            "novel_ratio": round(novel_ratio, 4),
            "novel_words": len(novel),
        })
    return verdicts


def mutual_knn_components(vectors: np.ndarray, k: int) -> dict:
    """Component sizes of the mutual-kNN graph.

    If this percolates -- one component swallowing nearly everything as k rises --
    then "the same" is not a threshold that can be chosen, and the detector has to
    be an exact canonical form instead of a similarity.
    """
    n = len(vectors)
    if n <= k:
        return {"k": k, "n": n, "components": 0, "largest": n, "note": "too few points"}

    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    neighbours = [set(np.argsort(-sims[i])[:k].tolist()) for i in range(n)]

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in neighbours[i]:
            if i in neighbours[j]:          # mutual, not one-way
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    sizes: dict[int, int] = {}
    for i in range(n):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    counts = sorted(sizes.values(), reverse=True)
    return {
        "k": k,
        "n": n,
        "components": len(counts),
        "largest": counts[0],
        "largest_share": round(counts[0] / n, 4),
        "singletons": sum(1 for c in counts if c == 1),
    }


def run_one(question: str, chat, embedder) -> dict:
    started = time.time()
    steps, error = [], None
    for event in reason(question, chat=chat, embedder=embedder):
        if event["type"] == "step":
            steps.append({"title": event["title"], "content": event["content"], "question": question})
        elif event["type"] == "final":
            steps.append({"title": "Final Answer", "content": event["content"], "question": question})
        elif event["type"] == "error":
            error = event["message"]
    return {"question": question, "steps": steps, "error": error, "seconds": round(time.time() - started, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=int, default=0, help="cap the number of questions (0 = all)")
    parser.add_argument("--out", default="docs/claims/loops.json")
    parser.add_argument(
        "--reanalyse", action="store_true",
        help="recompute from the saved transcripts instead of asking the model again",
    )
    args = parser.parse_args()

    out = ROOT / args.out
    transcripts = out.with_name("loops_transcripts.json")
    embedder = backends.OllamaEmbedding("")

    if args.reanalyse:
        # The raw steps are committed, so a threshold can be revisited without
        # spending another twenty minutes of model time -- and so anyone can check
        # these numbers rather than taking them.
        if not transcripts.exists():
            print(f"{transcripts} does not exist; run without --reanalyse first.")
            return 1
        runs = json.loads(transcripts.read_text())
        chat_model = "(from saved transcripts)"
        print(f"re-analysing {len(runs)} saved runs at "
              f"content>={RESTATEMENT_THRESHOLD} title>={REPEATED_MOVE_THRESHOLD}\n")
    else:
        chat_model = backends.pick_chat_model()
        if not backends.chat_models():
            print("No chat model installed. Try:  ollama pull llama3.2:3b")
            return 1

        chat = backends.OllamaChat(chat_model)
        flat = [(group, q) for group, qs in QUESTIONS.items() for q in qs]
        if args.questions:
            flat = flat[: args.questions]

        print(f"chat={chat_model}  embeddings={embedder.model}  questions={len(flat)}\n")
        runs = []
        for n, (group, question) in enumerate(flat, 1):
            print(f"  [{n}/{len(flat)}] {group:<11} {question[:56]}", flush=True)
            run = run_one(question, chat, embedder)
            run["group"] = group
            runs.append(run)
            if run["error"]:
                print(f"        error: {run['error'][:70]}")

    all_steps = [s for r in runs for s in r["steps"]]
    if not all_steps:
        print("No steps produced; nothing to measure.")
        return 1

    print("\naudit: is the normalisation itself creating duplicates?")
    audit = audit_normalisation(all_steps)
    for key, value in audit.items():
        print(f"  {key:<34} {value}")
    if not audit["trustworthy"]:
        print("\n  THE NORMALISATION IS SUSPECT. Every number below measures this script,")
        print("  not the model. Fix the extraction before believing any of it.")

    title_vecs = embedder.embed([s["title"] for s in all_steps])
    content_vecs = embedder.embed([s["content"] for s in all_steps])

    verdicts, offset = [], 0
    for run in runs:
        n = len(run["steps"])
        if n:
            verdicts.extend(classify(
                run["steps"], title_vecs[offset:offset + n], content_vecs[offset:offset + n]
            ))
        offset += n

    total = len(verdicts)
    repeats = [v for v in verdicts if v["kind"] == "repeat"]

    # How often a *question* loops, which is what a user experiences. Repeats
    # cluster -- one run here produced four -- so the per-step rate understates it.
    runs_with_repeat, cursor = 0, 0
    for run in runs:
        n = len(run["steps"])
        runs_with_repeat += any(v["kind"] == "repeat" for v in verdicts[cursor:cursor + n])
        cursor += n
    drifts = [v for v in verdicts if v["kind"] == "drift"]
    by_reason: dict[str, int] = {}
    for v in repeats:
        for kind, _, _ in v["reasons"]:
            by_reason[kind] = by_reason.get(kind, 0) + 1

    print(f"\nfailure modes over {total} steps in {len(runs)} runs")
    print(f"  repeat   {len(repeats):>4}  ({len(repeats)/total:.1%})   by detector: {by_reason}")
    print(f"  drift    {len(drifts):>4}  ({len(drifts)/total:.1%})")
    print(f"  neither  {total-len(repeats)-len(drifts):>4}  ({(total-len(repeats)-len(drifts))/total:.1%})")

    print("\nper group")
    per_group = {}
    idx = 0
    for run in runs:
        n = len(run["steps"])
        group_v = verdicts[idx:idx + n]
        bucket = per_group.setdefault(run["group"], {"steps": 0, "repeat": 0, "drift": 0, "runs": 0})
        bucket["runs"] += 1
        bucket["steps"] += n
        bucket["repeat"] += sum(1 for v in group_v if v["kind"] == "repeat")
        bucket["drift"] += sum(1 for v in group_v if v["kind"] == "drift")
        idx += n
    for group, b in per_group.items():
        avg = b["steps"] / max(b["runs"], 1)
        print(f"  {group:<11} {b['steps']:>3} steps ({avg:.1f}/run)  "
              f"repeat {b['repeat']:>3}  drift {b['drift']:>3}")

    print("\nmutual-kNN over all steps: does 'the same' percolate?")
    knn = [mutual_knn_components(content_vecs, k) for k in (2, 3, 4, 6)]
    for row in knn:
        print(f"  k={row['k']}  components {row['components']:>4}  "
              f"largest {row['largest']:>4} ({row.get('largest_share', 0):.1%})")

    payload = {
        "provenance": "measured",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_model": chat_model,
        "embedding_model": embedder.model,
        "runs": len(runs),
        "steps": total,
        "harness_audit": audit,
        "repeat_steps": len(repeats),
        "repeat_rate": round(len(repeats) / total, 4),
        "repeat_by_detector": by_reason,
        "drift_steps": len(drifts),
        "drift_rate": round(len(drifts) / total, 4),
        "clean_rate": round((total - len(repeats) - len(drifts)) / total, 4),
        "per_group": per_group,
        "mutual_knn": knn,
        "mean_steps_per_run": round(total / len(runs), 2),
        "runs_with_a_repeat": runs_with_repeat,
        "runs_with_a_repeat_rate": round(runs_with_repeat / len(runs), 4),
        "thresholds": {"restatement": RESTATEMENT_THRESHOLD, "repeated_move": REPEATED_MOVE_THRESHOLD},
        "errors": sum(1 for r in runs if r["error"]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")

    if not args.reanalyse:
        transcripts.write_text(json.dumps(runs, indent=2) + "\n")
        print(f"wrote {transcripts.relative_to(ROOT)} (the raw steps, so this is re-checkable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
