"""Run the arithmetic battery through the model, with the gate on and off.

The eval in eval.py measures reasoning across mixed question types, where
arithmetic is one thread among several and the noise floor swamps it -- two runs
of an identical configuration there differed by 28 points. This measures one
thing only, on numbers no model has memorised, which is the setting where the
gate either does something or does not.

Usage:
    python dev/scripts/battery_bench.py --per-group 2
    python dev/scripts/battery_bench.py --per-group 2 --seed 99  # a fresh battery
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from battery import SEED, build  # noqa: E402

from mpe_lkg import backends  # noqa: E402
from mpe_lkg.reasoning import reason  # noqa: E402

# Per seed, so a replication on a fresh battery cannot silently overwrite the
# result it is meant to be checked against.
OUT_DIR = pathlib.Path("docs/claims")


def run_arm(questions, *, check_arithmetic: bool, decompose: int, budget: float,
            select: bool = False) -> dict:
    chat = backends.OllamaChat(backends.pick_chat_model())
    embedder = backends.OllamaEmbedding("")
    rows = []
    for index, question in enumerate(questions, 1):
        started = time.time()
        answer, calcs, steps, error = "", [], 0, ""
        try:
            for event in reason(
                question.text,
                chat=chat,
                embedder=embedder,
                decompose=decompose,
                max_steps=14,
                check_arithmetic=check_arithmetic,
                select_answer=select,
                time_budget=budget,
            ):
                if event["type"] == "calc":
                    calcs.append(f"{event['expression']} = {event['value']}")
                elif event["type"] == "step":
                    steps += 1
                elif event["type"] == "final":
                    answer = event["content"]
                elif event["type"] == "error":
                    error = event.get("message", "error")
        except Exception as exc:  # noqa: BLE001 -- a crashed run is a result, not a stop
            error = f"{type(exc).__name__}: {exc}"

        correct = bool(answer) and question.matches(answer)
        # Did the exact value ever get computed, whether or not it was then used?
        # The distinction matters: those are two different failures with two
        # different fixes.
        settled = any(str(question.answer.numerator) in c.replace(",", "") for c in calcs)
        rows.append({
            "group": question.group,
            "question": question.text,
            "expected": str(question.answer),
            "answer": answer,
            "correct": correct,
            "settled_exactly": settled,
            "calcs": calcs,
            "steps": steps,
            "seconds": round(time.time() - started, 1),
            "error": error,
        })
        mark = "." if correct else "x"
        extra = " (computed but not used)" if settled and not correct else ""
        print(f"  [{index:2d}/{len(questions)}] {mark}  {question.group:22s} "
              f"{question.text[:44]}{extra}", flush=True)
    return {"rows": rows}


def summarise(rows: list[dict]) -> dict:
    by_group: dict[str, dict] = {}
    for row in rows:
        bucket = by_group.setdefault(row["group"], {"n": 0, "correct": 0, "settled": 0})
        bucket["n"] += 1
        bucket["correct"] += int(row["correct"])
        bucket["settled"] += int(row["settled_exactly"])
    n = len(rows)
    return {
        "n": n,
        "correct": sum(r["correct"] for r in rows),
        "correct_rate": (sum(r["correct"] for r in rows) / n) if n else 0.0,
        "settled_exactly": sum(r["settled_exactly"] for r in rows),
        "computed_but_unused": sum(
            1 for r in rows if r["settled_exactly"] and not r["correct"]),
        "steps_mean": (sum(r["steps"] for r in rows) / n) if n else 0.0,
        "errors": sum(1 for r in rows if r["error"]),
        "per_group": by_group,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--group", default="", help="only this group")
    parser.add_argument("--select", action="store_true",
                        help="answer by choosing a computed fact instead of writing one")
    args = parser.parse_args()

    questions = build(args.seed, args.per_group)
    if args.group:
        questions = [q for q in questions if q.group == args.group]
        if not questions:
            parser.error(f"no group {args.group!r}")
    print(f"battery: {len(questions)} questions, seed {args.seed}\n")

    results = {}
    for label, gate in (("gate_on", True), ("gate_off", False)):
        print(f"--- {label} ---")
        rows = run_arm(questions, check_arithmetic=gate, decompose=args.decompose,
                       budget=args.budget, select=args.select)["rows"]
        results[label] = summarise(rows)
        results[label]["rows"] = rows
        print()

    on, off = results["gate_on"], results["gate_off"]
    print(f"  gate on : {on['correct']}/{on['n']} ({on['correct_rate']*100:.0f}%)  "
          f"settled exactly {on['settled_exactly']}/{on['n']}  "
          f"computed-but-unused {on['computed_but_unused']}")
    print(f"  gate off: {off['correct']}/{off['n']} ({off['correct_rate']*100:.0f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"battery_{args.seed}.json"
    out.write_text(json.dumps({"seed": args.seed, **results}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
