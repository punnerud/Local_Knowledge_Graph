"""Run the battery against a model, and grade each mode with its own authority.

Exact and text questions are marked correct or not. Consistency questions are
NEVER counted as correct or incorrect: they are graded by whether the model
agrees with itself across repeated askings, reported under their own key with
the caveat attached, because a consistently wrong answer is invisible to that
grading and a reader must not be allowed to forget it.
"""

from __future__ import annotations

import time

from . import Question, build


def run(
    questions: list[Question],
    *,
    chat,
    embedder,
    check_arithmetic: bool = True,
    decompose: int = 8,
    budget: float = 120.0,
    select: bool = False,
    progress=print,
) -> dict:
    """Every question through the reasoning loop, one row each."""
    from ..reasoning import _steady, reason

    rows = []
    for index, question in enumerate(questions, 1):
        started = time.time()

        if question.mode == "consistency":
            # No reasoning run: the measurement is self-agreement at sampling
            # temperature, and a full run would launder it into an "answer".
            steady, said = _steady(chat, embedder, question.text)
            rows.append({
                "group": question.group, "domain": question.domain,
                "question": question.text, "mode": "consistency",
                "steady": steady, "said": said[:100],
                "seconds": round(time.time() - started, 1),
            })
            progress(f"  [{index:2d}/{len(questions)}] "
                     f"{'=' if steady else '~'}  {question.group:22s} "
                     f"{question.text[:44]}")
            continue

        answer, calcs, steps, error = "", [], 0, ""
        try:
            for event in reason(
                question.text, chat=chat, embedder=embedder,
                decompose=decompose, max_steps=14,
                check_arithmetic=check_arithmetic, time_budget=budget,
                select_answer=select,
            ):
                if event["type"] == "calc":
                    calcs.append(f"{event['expression']} = {event['value']}")
                elif event["type"] == "step":
                    steps += 1
                elif event["type"] == "final":
                    answer = event["content"]
                elif event["type"] == "error":
                    error = event.get("message", "error")
        except Exception as exc:  # noqa: BLE001 -- a crashed run is a result
            error = f"{type(exc).__name__}: {exc}"

        correct = bool(answer) and bool(question.matches(answer))
        settled = (question.answer is not None and question.answer.denominator == 1
                   and any(str(question.answer.numerator) in c.replace(",", "")
                           for c in calcs))
        rows.append({
            "group": question.group, "domain": question.domain,
            "question": question.text, "mode": question.mode,
            "expected": str(question.answer) if question.answer is not None
                        else "|".join(question.expect),
            "answer": answer, "correct": correct,
            "settled_exactly": settled, "calcs": calcs, "steps": steps,
            "seconds": round(time.time() - started, 1), "error": error,
        })
        progress(f"  [{index:2d}/{len(questions)}] "
                 f"{'.' if correct else 'x'}  {question.group:22s} "
                 f"{question.text[:44]}")
    return {"rows": rows}


def summarise(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["mode"] != "consistency"]
    steadiness = [r for r in rows if r["mode"] == "consistency"]
    by_group: dict[str, dict] = {}
    for row in graded:
        bucket = by_group.setdefault(row["group"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(row["correct"])

    out = {
        "n": len(graded),
        "correct": sum(r["correct"] for r in graded),
        "correct_rate": (sum(r["correct"] for r in graded) / len(graded))
                        if graded else 0.0,
        "settled_exactly": sum(1 for r in graded if r.get("settled_exactly")),
        "computed_but_unused": sum(
            1 for r in graded if r.get("settled_exactly") and not r["correct"]),
        "steps_mean": (sum(r["steps"] for r in graded) / len(graded)) if graded else 0.0,
        "errors": sum(1 for r in graded if r.get("error")),
        "per_group": by_group,
    }
    if steadiness:
        out["consistency"] = {
            "n": len(steadiness),
            "steady": sum(1 for r in steadiness if r["steady"]),
            # Attached to the data, not left to a README: this grading cannot
            # see a consistently wrong answer, so it must never be read as
            # correctness.
            "note": "self-agreement only; a consistently wrong answer is "
                    "invisible to this grading",
            "per_group": {
                g: {"n": sum(1 for r in steadiness if r["group"] == g),
                    "steady": sum(1 for r in steadiness
                                  if r["group"] == g and r["steady"])}
                for g in sorted({r["group"] for r in steadiness})
            },
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """The ``mpe-lkg battery`` command."""
    import argparse
    import json
    import pathlib

    from .. import backends

    parser = argparse.ArgumentParser(prog="mpe-lkg battery")
    parser.add_argument("--seed", type=int, default=None,
                        help="battery seed (default: the pinned one)")
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--domains", default="",
                        help="comma-separated; default all registered")
    parser.add_argument("--model", default="", help="chat model; default installed")
    parser.add_argument("--select", action="store_true",
                        help="answer by choosing a computed fact")
    parser.add_argument("--out", default="", help="output JSON path")
    args = parser.parse_args(argv)

    from . import SEED
    seed = args.seed if args.seed is not None else SEED
    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None
    questions = build(seed, args.per_group, domains)
    print(f"battery: {len(questions)} questions, seed {seed}, "
          f"domains {domains or sorted({q.domain for q in questions})}")

    chat = backends.OllamaChat(args.model or backends.pick_chat_model())
    embedder = backends.OllamaEmbedding("")
    rows = run(questions, chat=chat, embedder=embedder, select=args.select)["rows"]
    result = summarise(rows)
    result["rows"] = rows
    result["seed"] = seed
    result["model"] = args.model or "installed default"

    print(f"\n  correct {result['correct']}/{result['n']} "
          f"({result['correct_rate'] * 100:.0f}%)")
    if "consistency" in result:
        c = result["consistency"]
        print(f"  consistency (not correctness): steady {c['steady']}/{c['n']}")

    # The arm and seed in the name, so a run can never overwrite the baseline
    # it is to be compared against. That has nearly happened once.
    arm = "_select" if args.select else ""
    out = pathlib.Path(args.out or f"battery_{seed}{arm}.json")
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")
    return 0
