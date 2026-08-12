"""Run the battery with the arithmetic gate on and off, and write both arms.

The runner and the grading live in ``mpe_lkg.battery.bench`` now; this script is
the two-arm comparison harness around them. The eval in eval.py measures
reasoning across mixed question types, where the noise floor is wide; this
measures one thing at a time on numbers no model has memorised.

Usage:
    python dev/scripts/battery_bench.py --per-group 2
    python dev/scripts/battery_bench.py --per-group 2 --seed 99 --select
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from battery import SEED, build  # noqa: E402

from mpe_lkg import backends  # noqa: E402
from mpe_lkg.battery.bench import run, summarise  # noqa: E402

OUT_DIR = pathlib.Path("docs/claims")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--group", default="", help="only this group")
    parser.add_argument("--model", default="",
                        help="chat model to measure; defaults to whatever is installed")
    parser.add_argument("--select", action="store_true",
                        help="answer by choosing a computed fact instead of writing one")
    args = parser.parse_args()

    questions = build(args.seed, args.per_group)
    if args.group:
        questions = [q for q in questions if q.group == args.group]
        if not questions:
            parser.error(f"no group {args.group!r}")
    print(f"battery: {len(questions)} questions, seed {args.seed}\n")

    chat = backends.OllamaChat(args.model or backends.pick_chat_model())
    embedder = backends.OllamaEmbedding("")

    results = {"model": args.model or "installed default"}
    for label, gate in (("gate_on", True), ("gate_off", False)):
        print(f"--- {label} ---")
        rows = run(questions, chat=chat, embedder=embedder,
                   check_arithmetic=gate, decompose=args.decompose,
                   budget=args.budget, select=args.select)["rows"]
        results[label] = summarise(rows)
        results[label]["rows"] = rows
        print()

    on, off = results["gate_on"], results["gate_off"]
    print(f"  gate on : {on['correct']}/{on['n']} ({on['correct_rate'] * 100:.0f}%)  "
          f"settled exactly {on['settled_exactly']}/{on['n']}  "
          f"computed-but-unused {on['computed_but_unused']}")
    print(f"  gate off: {off['correct']}/{off['n']} ({off['correct_rate'] * 100:.0f}%)")

    # The arm goes in the filename: a --select run must never overwrite the
    # baseline it is to be compared against. That nearly happened once.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.model.split(":")[0].replace("/", "-") if args.model else "default"
    arm = "_select" if args.select else ""
    out = OUT_DIR / f"battery_{args.seed}_{tag}{arm}.json"
    out.write_text(json.dumps({"seed": args.seed, **results}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
