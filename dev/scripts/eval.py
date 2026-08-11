#!/usr/bin/env python3
"""A baseline, so that "we saved context" can be checked rather than asserted.

The next change alters how the prompt is built. That can save tokens by making the
model worse, and without a before-and-after there is no way to tell the two apart.
This measures four things per question: whether the answer survived, how many steps
it took, how many prompt tokens it cost, and how often it repeated itself.

The grader is deliberately blunt. It looks for required substrings and forbidden
ones -- enough to catch "the answer disappeared", not enough to score prose. A
grader that needs a model of its own is a second thing that can be wrong, and it
would be measured by nothing.

Token counts come from Ollama's own ``prompt_eval_count`` rather than from dividing
characters by four, so the saving is exact.

    .venv/bin/python scripts/eval.py [--repeats N] [--questions N] [--label NAME]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mpe_lkg import backends  # noqa: E402
from mpe_lkg.reasoning import SHORT_SYSTEM_PROMPT, reason  # noqa: E402

# expect: any one of these substrings must appear in the final answer.
# reject:  none of these may appear.
# The malformed group is graded on whether the answer questions the premise at all,
# which is the only defensible thing to ask of a question with no answer.
QUESTIONS = [
    # -- simple: a wrong answer here is unambiguous, and padding shows as step count
    {"group": "simple", "q": "What is the capital of France?", "expect": ["paris"]},
    {"group": "simple", "q": "What is the chemical symbol for gold?", "expect": ["au"]},
    {"group": "simple", "q": "Who wrote the play Hamlet?", "expect": ["shakespeare"]},
    {"group": "simple", "q": "What is the largest ocean on Earth?", "expect": ["pacific"]},
    {"group": "simple", "q": "In what year did the Berlin Wall fall?", "expect": ["1989"]},
    {"group": "simple", "q": "How many days are in a leap year?", "expect": ["366"]},

    # -- multi-step: these are where cutting context too hard should show up first
    {"group": "multi_step", "q": "What is 17 percent of 250?", "expect": ["42.5", "42,5"]},
    {"group": "multi_step", "q": "If a train travels 240 km in 3 hours at a constant speed, "
                                 "how many hours does it take to travel 400 km?",
     "expect": ["5"]},
    {"group": "multi_step", "q": "How many minutes are there in a fortnight?", "expect": ["20160", "20,160"]},
    {"group": "multi_step", "q": "If I buy 3 items at 4.50 each and pay with a 20 note, "
                                 "what is my change?",
     "expect": ["6.50", "6,50", "6.5"]},
    {"group": "multi_step", "q": "A rectangle is twice as long as it is wide and has a perimeter "
                                 "of 36 cm. What is its width?",
     "expect": ["6"]},
    # pi*3^2 = 28.27 against 5^2 = 25, so the CIRCLE is larger. This expectation was
    # written backwards at first, and the arms that answered correctly were marked
    # wrong for it. Every arithmetic answer here has since been checked by hand.
    {"group": "multi_step", "q": "Which is larger, the area of a circle of radius 3 or a square "
                                 "of side 5?",
     "expect": ["circle"], "reject": ["square is larger", "square has a larger"]},

    # -- malformed: the only right answer is to challenge the question
    {"group": "malformed", "q": "What is the capital of Oslo?",
     "expect": ["not have a capital", "is the capital", "is a city", "itself the capital",
                "does not have", "misunderstand", "premise"]},
    {"group": "malformed", "q": "When did Napoleon invade Australia?",
     "expect": ["never", "did not", "didn't", "no record", "no evidence"]},
    {"group": "malformed", "q": "Who was the first person to walk on the Sun?",
     "expect": ["no one", "nobody", "impossible", "cannot", "has not", "never"]},
    {"group": "malformed", "q": "What is the population of the Atlantic Ocean?",
     "expect": ["no population", "not have a population", "ocean", "does not have", "not applicable"]},
    {"group": "malformed", "q": "How tall is the colour blue?",
     "expect": ["not have a height", "does not have", "colour", "color", "not a physical",
                "cannot", "misunderstand"]},

    # -- hard: several angles genuinely required, and the obvious answer is often
    #    wrong. The simple group is saturated at 24/24, and a saturated group can
    #    only show a fall, never an improvement. Every expectation below is worked
    #    out by hand in its comment -- an inverted one slipped through last time.
    {"group": "hard",
     # 14:35 + 2h50m = 17:25 arrival. Leaving 25 min before 14:35 = 14:10.
     "q": "A train departs at 14:35 and the journey takes 2 hours 50 minutes. I need 25 "
          "minutes to reach the station. What time must I leave home, and what time do I arrive?",
     "expect": ["14:10", "2:10"], "reject": []},
    {"group": "hard",
     # Three colours, so by the pigeonhole principle four draws guarantee a pair.
     "q": "A drawer holds 3 red socks, 5 blue socks and 2 green socks. How many socks must I "
          "take out in the dark to be certain of having a matching pair?",
     "expect": ["4", "four"]},
    {"group": "hard",
     # The classic trap: they weigh the same. A kilogram is a kilogram.
     "q": "Which weighs more, a kilogram of feathers or a kilogram of steel?",
     "expect": ["same", "equal", "neither", "both weigh"],
     "reject": ["steel weighs more", "feathers weigh more"]},
    {"group": "hard",
     # b + (b+1.00) = 1.10 -> b = 0.05. The intuitive 0.10 is wrong.
     "q": "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. "
          "How much does the ball cost?",
     "expect": ["0.05", ".05", "5 cent", "5p"], "reject": ["0.10 ", "10 cent"]},
    {"group": "hard",
     # 5 machines 5 minutes 5 widgets -> 1 machine makes 1 widget in 5 minutes ->
     # 100 machines make 100 widgets in 5 minutes.
     "q": "If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take "
          "to make 100 widgets?",
     "expect": ["5 minute", "five minute"], "reject": ["100 minute", "hundred minute"]},
    {"group": "hard",
     # Doubling daily, full on day 48, so half-covered the day before: day 47.
     "q": "Lily pads double in area every day and cover the whole lake on day 48. "
          "On which day is the lake half covered?",
     "expect": ["47", "forty-seven"]},
    {"group": "hard",
     # 1.5 km at 5 km/h = 18 min walking; the bus takes 8 min but leaves in 12,
     # arriving at 20 min. Walking arrives first.
     "q": "The shop is 1.5 km away. I walk at 5 km/h. A bus leaves in 12 minutes and takes "
          "8 minutes. Do I get there sooner walking or waiting for the bus?",
     "expect": ["walk"], "reject": ["bus is faster", "wait for the bus", "take the bus"]},
    {"group": "hard",
     # Two hops: Norway's capital is Oslo; Oslo is on the Oslofjord.
     "q": "Which body of water is the capital city of Norway situated on?",
     "expect": ["oslofjord", "oslo fjord", "fjord"]},

    # -- ambiguous: KEPT, BUT NOT GRADED. Substring matching cannot score these, and
    #    pretending otherwise produced a number that moved for the wrong reason.
    #    "A hot dog is a sandwich." is a perfectly good answer and matches nothing in
    #    any keyword list worth writing; meanwhile a rambling hedge scores well by
    #    accident because it happens to contain the word "depends". Measured: the
    #    baseline beat the synthesis arm here 67 % to 33 % on exactly that artefact.
    #    They are still run, so the answers can be read, and reported separately.
    {"group": "ambiguous", "q": "Is a hot dog a sandwich?", "expect": [], "gradeable": False},
    {"group": "ambiguous", "q": "Which came first, the chicken or the egg?",
     "expect": [], "gradeable": False},
    {"group": "ambiguous", "q": "What is the meaning of the word 'set'?",
     "expect": [], "gradeable": False},
]


def grade(answer: str, spec: dict) -> bool | None:
    """True, False, or None when this question is not gradeable this way."""
    if not spec.get("gradeable", True):
        return None
    text = re.sub(r"\s+", " ", answer.lower())
    if any(bad.lower() in text for bad in spec.get("reject", [])):
        return False
    return any(good.lower() in text for good in spec["expect"])


def run_once(spec: dict, chat, embedder, *, synthesise=True, budget=120.0, detect=True,
             system_prompt="", min_steps=0, decompose=0, check_arith=True) -> dict:
    chat.reset_usage()
    started = time.time()
    steps, answer, error = [], "", None

    repeats, sums_fixed = 0, 0
    for event in reason(spec["q"], chat=chat, embedder=embedder, synthesise=synthesise,
                        time_budget=budget, detect_repeats=detect, system_prompt=system_prompt,
                        min_steps=min_steps, decompose=decompose,
                        check_arithmetic=check_arith):
        if event["type"] == "step":
            steps.append(event["content"])
        elif event["type"] == "repeat":
            repeats += 1
        elif event["type"] == "arithmetic":
            sums_fixed += 1
        elif event["type"] == "final":
            answer = event["content"]
        elif event["type"] == "error":
            error = event["message"]

    usage = chat.describe()
    return {
        "question": spec["q"],
        "group": spec["group"],
        "answer": answer,
        "correct": (grade(answer, spec) and bool(answer) and not error) if spec.get("gradeable", True) else None,
        "gradeable": spec.get("gradeable", True),
        "steps": len(steps) + (1 if answer else 0),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "calls": usage["calls"],
        "seconds": round(time.time() - started, 2),
        "repeats_caught": repeats,
        "sums_fixed": sums_fixed,
        "error": error,
    }


def summarise(results: list[dict]) -> dict:
    def stat(key):
        values = [r[key] for r in results]
        return {
            "mean": round(statistics.fmean(values), 2),
            "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        }

    graded = [r for r in results if r["gradeable"]]
    correct = sum(bool(r["correct"]) for r in graded)
    return {
        "n": len(results),
        "graded": len(graded),
        "correct": correct,
        # Over the gradeable questions only. Averaging in a group the grader cannot
        # score would make the headline move for reasons that are not about quality.
        "correct_rate": round(correct / len(graded), 4) if graded else None,
        "steps": stat("steps"),
        "prompt_tokens": stat("prompt_tokens"),
        "completion_tokens": stat("completion_tokens"),
        "calls": stat("calls"),
        "seconds": stat("seconds"),
        "repeats_caught": sum(r.get("repeats_caught", 0) for r in results),
        "sums_fixed": sum(r.get("sums_fixed", 0) for r in results),
        "errors": sum(1 for r in results if r["error"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3,
                        help="runs per question; one has no error bar (default 3)")
    parser.add_argument("--questions", type=int, default=0, help="cap the question count (0 = all)")
    # Measuring one group is how a targeted change gets checked without paying for
    # the 28 questions it cannot affect.
    parser.add_argument("--group", default="", help="only this group (simple, multi_step, hard, ...)")
    parser.add_argument("--label", default="baseline", help="what this arm is called in the output")
    parser.add_argument("--out", default="docs/claims/eval.json")
    parser.add_argument("--no-synthesis", action="store_true",
                        help="take the last step as the answer, as the baseline did")
    parser.add_argument("--budget", type=float, default=120.0, help="seconds per run")
    parser.add_argument("--no-arithmetic-check", action="store_true",
                        help="do not verify the model's sums with mpeqs")
    parser.add_argument("--decompose", type=int, default=0,
                        help="plan N angles first, then take one step per angle")
    parser.add_argument("--min-steps", type=int, default=0,
                        help="floor on reasoning steps; 0 lets novelty decide")
    parser.add_argument("--short-prompt", action="store_true",
                        help="the terser prompt that measured 9 points worse")
    parser.add_argument("--no-repeat-detection", action="store_true",
                        help="keep every step, however much it repeats")
    args = parser.parse_args()

    if not backends.chat_models():
        print("No chat model installed. Try:  ollama pull llama3.2:3b")
        return 1

    chat = backends.OllamaChat(backends.pick_chat_model())
    embedder = backends.OllamaEmbedding("")
    specs = [q for q in QUESTIONS if q["group"] == args.group] if args.group else QUESTIONS
    if args.group and not specs:
        parser.error(f"no group {args.group!r}; have "
                     + ", ".join(sorted({q['group'] for q in QUESTIONS})))
    specs = specs[: args.questions] if args.questions else specs

    print(f"arm={args.label}  chat={chat.model}  embeddings={embedder.model}")
    print(f"{len(specs)} questions x {args.repeats} runs = {len(specs) * args.repeats} runs\n")

    results = []
    for n, spec in enumerate(specs, 1):
        marks = []
        for _ in range(args.repeats):
            result = run_once(spec, chat, embedder, synthesise=not args.no_synthesis,
                              budget=args.budget, detect=not args.no_repeat_detection,
                              system_prompt=SHORT_SYSTEM_PROMPT if args.short_prompt else "",
                              min_steps=args.min_steps, decompose=args.decompose,
                              check_arith=not args.no_arithmetic_check)
            results.append(result)
            marks.append("-" if result["correct"] is None else ("." if result["correct"] else "x"))
        print(f"  [{n:>2}/{len(specs)}] {''.join(marks)}  {spec['group']:<11} {spec['q'][:52]}", flush=True)

    overall = summarise(results)
    per_group = {
        group: summarise([r for r in results if r["group"] == group])
        for group in dict.fromkeys(s["group"] for s in specs)
    }

    print(f"\n{args.label}")
    print(f"  correct        {overall['correct']}/{overall['graded']}  "
          f"({overall['correct_rate']:.1%})   [{overall['n'] - overall['graded']} runs not gradeable]")
    print(f"  steps per run  {overall['steps']['mean']} +/- {overall['steps']['stdev']}")
    print(f"  prompt tokens  {overall['prompt_tokens']['mean']} +/- {overall['prompt_tokens']['stdev']}")
    print(f"  model calls    {overall['calls']['mean']} +/- {overall['calls']['stdev']}")
    print(f"  seconds        {overall['seconds']['mean']} +/- {overall['seconds']['stdev']}")

    print("\nper group")
    for group, s in per_group.items():
        rate = f"{s['correct_rate']:.0%}" if s["correct_rate"] is not None else "  --"
        print(f"  {group:<11} {s['correct']:>2}/{s['graded']:<2} ({rate:>4})  "
              f"steps {s['steps']['mean']:>4}  prompt tokens {s['prompt_tokens']['mean']:>7}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update({
        "provenance": "measured",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_model": chat.model,
        "embedding_model": embedder.model,
        "repeats": args.repeats,
        args.label: {"overall": overall, "per_group": per_group},
    })
    out.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)} (arm '{args.label}')")

    raw = out.with_name(f"eval_runs_{args.label}.json")
    raw.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {raw.relative_to(ROOT)} (every answer, so the grading is checkable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
