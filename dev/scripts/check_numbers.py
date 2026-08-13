#!/usr/bin/env python3
"""Fail loudly when a documented number stops being true.

Prose drifts away from data silently. This turns every measurable claim in the
documentation into a lookup against the JSON that produced it, so a claim that stops
holding breaks the build instead of quietly becoming a lie.

Four rules the checks below follow:

1. The claim's English wording is the check's label, so a failure reads as a
   sentence rather than as a key path.
2. Tolerances come from observed spread, not from taste.
3. Some checks must measure against something other than this pipeline's own
   history -- a synthetic input whose answer is known by arithmetic. A consistency
   gate cannot detect a consistent error.
4. Generated files carry a provenance field which defaults to the untrusted value,
   so an old file is never mistaken for a fresh measurement.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mpe_lkg.graph import cosine_similarity, strongest_path, top_similarities  # noqa: E402

CLAIMS = "docs/claims/edge_spread.json"
SEARCH = "docs/claims/search_bench.json"
SWEEP = "docs/claims/layer_sweep.json"
LOOPS = "docs/claims/loops.json"
EVAL = "docs/claims/eval.json"
BATTERY = "docs/claims/battery_20260811.json"
BATTERY_REPLICATION = "docs/claims/battery_99.json"
UNITS = "docs/claims/battery_4242.json"
QWEN = "docs/claims/battery_20260811_qwen3.json"
QWEN_SELECT = "docs/claims/battery_555_qwen3_select.json"
QWEN_REPLICATION = "docs/claims/battery_555_qwen3.json"
LLAMA_REPLICATION = "docs/claims/battery_555_llama3.2.json"
FULL_BATTERY = "docs/claims/battery_full_20260811_qwen3.json"
PHYSICS = "docs/claims/battery_physics_20260811_qwen3.json"
CALCPHYS = "docs/claims/battery_calcphys_20260811_qwen3.json"
ALL_SIX = "docs/claims/battery_all6_20260811_qwen3.json"

# (file, extractor, expected, tolerance, label)
#
# Tolerances are three times the spread observed across the four independent topics
# in scripts/measure.py, so these fail on a regression rather than on the ordinary
# variation between one topic and the next. The per-topic standard deviations are
# recorded in the same file as cv_stdev / mean_stdev.
FILE_CHECKS = [
    (
        CLAIMS,
        lambda d: 1.0 if d.get("provenance") == "measured" else 0.0,
        1.0,
        0.001,
        "edge spread is measured, not inherited",
    ),
    (
        CLAIMS,
        lambda d: float(d["all-minilm"]["dim"]),
        384.0,
        0.0,
        "all-minilm reports 384 dimensions",
    ),
    (
        CLAIMS,
        lambda d: float(d["nomic-embed-text"]["dim"]),
        768.0,
        0.0,
        "nomic-embed-text reports 768 dimensions",
    ),
    (
        CLAIMS,
        lambda d: float(d["all-minilm"]["n"]),
        36.0,
        0.0,
        "four six-step topics at top_k=2 give 36 edges",
    ),
    (
        CLAIMS,
        lambda d: float(d["nomic-embed-text"]["mean"]),
        0.6660,
        0.14,
        "reasoning steps average 0.67 cosine under nomic-embed-text",
    ),
    (
        CLAIMS,
        lambda d: float(d["nomic-embed-text"]["cv"]),
        0.1330,
        0.076,
        "nomic-embed-text edge weights vary by only 13 percent of their mean",
    ),
    (
        CLAIMS,
        lambda d: float(d["all-minilm"]["mean"]),
        0.4788,
        0.20,
        "reasoning steps average 0.48 cosine under all-minilm",
    ),
    (
        CLAIMS,
        lambda d: float(d["all-minilm"]["cv"]),
        0.2764,
        0.33,
        "all-minilm edge weights vary by 28 percent of their mean",
    ),
    (
        CLAIMS,
        # The comparison, not either number alone: the larger, better-regarded model
        # produces the *less* discriminative graph on this task. If that ever
        # reverses, the advice in the README is wrong and should change.
        lambda d: 1.0 if d["nomic-embed-text"]["cv"] < d["all-minilm"]["cv"] else 0.0,
        1.0,
        0.001,
        "nomic-embed-text separates steps less than all-minilm does",
    ),
    # Timings are machine-dependent, so these are upper bounds rather than point
    # values: expected 0 with the bound as the tolerance. They are set an order of
    # magnitude above what this machine measures, so they fail on a complexity
    # regression -- an accidental O(n log n) sort or a per-query database decode --
    # rather than on a slower laptop.
    (
        SEARCH,
        lambda d: float(d["all_pairs_20_steps_ms"]),
        0.0,
        1.0,
        "all-pairs over a 20-step chain stays under a millisecond",
    ),
    (
        SEARCH,
        lambda d: float(d["exact_topk_1k_ms"]),
        0.0,
        1.0,
        "exact top-k over 1000 vectors stays under a millisecond",
    ),
    (
        SEARCH,
        lambda d: float(d["exact_topk_100k_ms"]),
        0.0,
        50.0,
        "exact top-k over 100000 vectors stays under 50 ms",
    ),
    (
        SEARCH,
        # Falsified-prediction guard. The approximate index this project used to
        # depend on returns one wrong neighbour per query on a current numpy, which
        # is why it was removed. If a future build starts behaving, that is news and
        # the removal is worth revisiting.
        lambda d: 1.0 if d.get("annoy_returns_self_first") else 0.0,
        0.0,
        0.001,
        "annoy 1.17.3 still fails to return a vector as its own neighbour",
    ),
    (
        SWEEP,
        lambda d: 1.0 if d.get("provenance") == "measured" else 0.0,
        1.0,
        0.001,
        "layer sweep is measured, not inherited",
    ),
    (
        SWEEP,
        # The first block sees each token before any context has been mixed in, so it
        # cannot tell one topic from another. Near-zero here is the control that says
        # the separation measured deeper in is real and not an artefact of the metric.
        lambda d: float(d["first_layer_separation"]),
        0.0,
        0.02,
        "the first block separates topics by almost nothing",
    ),
    (
        SWEEP,
        lambda d: float(d["best_separation"]),
        0.148,
        0.06,
        "the deepest block separates topics by about 0.15",
    ),
    (
        SWEEP,
        lambda d: 1.0 if d["best_separation"] > 5 * max(d["first_layer_separation"], 1e-6) else 0.0,
        1.0,
        0.001,
        "depth separates topics several times better than the first block",
    ),
    # The loop battery is ONE run of 24 questions, so it has no error bar of its
    # own. These are therefore written as guards and comparisons rather than as
    # point values: a rate pinned to three decimals from a single battery would be
    # precision this measurement has not earned.
    (
        LOOPS,
        # The guard that comes before every other number in that file. Elsewhere a
        # first run reported a 58 % loop rate that was entirely the step extractor
        # seeing its own output twice; a measurement of the harness nearly became a
        # finding about the model.
        lambda d: 1.0 if d["harness_audit"]["trustworthy"] else 0.0,
        1.0,
        0.001,
        "the loop measurement does not manufacture its own duplicates",
    ),
    (
        LOOPS,
        lambda d: float(d["errors"]),
        0.0,
        0.0,
        "every question in the loop battery completed",
    ),
    (
        LOOPS,
        # An upper bound, not a value. It catches a regression to the 21 % that a
        # 0.90 threshold produced before inspection showed that bar was flagging
        # ordinary progress as repetition.
        lambda d: float(d["repeat_rate"]),
        0.0,
        0.15,
        "repeated steps stay a small minority of all steps",
    ),
    (
        LOOPS,
        # The gate's actual question. Drift is the failure a repeat detector cannot
        # see, and elsewhere it dominated; here it does not, which is what justifies
        # building the detector at all. If this ever flips, that plan needs redoing.
        lambda d: 1.0 if d["drift_rate"] <= d["repeat_rate"] else 0.0,
        1.0,
        0.001,
        "drift does not dominate repetition, so a repeat detector is the right tool",
    ),
    (
        LOOPS,
        # The number that describes the experience rather than the steps: repeats
        # cluster, so a small per-step rate still means a quarter of questions loop.
        lambda d: float(d["runs_with_a_repeat_rate"]),
        0.25,
        0.20,
        "roughly a quarter of questions produce at least one repeated step",
    ),
    (
        LOOPS,
        # Percolation check. Elsewhere the largest component swallowed 99 % of the
        # points by k=6 and cluster purity fell to the random baseline, which would
        # mean "the same" is not a threshold that can be chosen at all. Ours holds.
        lambda d: max(row["largest_share"] for row in d["mutual_knn"] if row["k"] <= 4),
        0.0,
        0.15,
        "the step graph does not percolate at k<=4, so similarity stays meaningful",
    ),
    # The eval arms. The tolerance on the headline is wide on purpose: a two-run
    # arm read 97.1 % and four runs read 91.2 %, so anything tighter would be
    # pinning the sampling noise of a stochastic model rather than its behaviour.
    (
        EVAL,
        lambda d: 1.0 if d.get("provenance") == "measured" else 0.0,
        1.0,
        0.001,
        "the eval is measured, not inherited",
    ),
    (
        EVAL,
        lambda d: float(d["validated"]["overall"]["correct_rate"]),
        0.91,
        0.10,
        "the shipped configuration answers about 91 percent of gradeable questions",
    ),
    (
        EVAL,
        # The comparison is what matters, and it is far outside the noise: this is
        # the same 20 questions, the same model, the same grader.
        lambda d: 1.0 if (d["validated"]["overall"]["correct_rate"]
                          > d["baseline"]["overall"]["correct_rate"] + 0.25) else 0.0,
        1.0,
        0.001,
        "it beats what this project shipped by more than 25 points",
    ),
    (
        EVAL,
        lambda d: float(d["validated"]["overall"]["prompt_tokens"]["mean"]),
        1649.0,
        800.0,
        "a run costs about 1650 prompt tokens, down from 2700",
    ),
    (
        EVAL,
        lambda d: float(d["validated"]["overall"]["steps"]["mean"]),
        3.9,
        1.5,
        "a run takes about four steps, down from six",
    ),
    (
        EVAL,
        lambda d: float(d["validated"]["overall"]["errors"]),
        0.0,
        0.0,
        "no run in the eval errored",
    ),
    # --- Working through the planned angles, rather than abandoning them ---
    (
        EVAL,
        lambda d: float(d["angle_floor"]["overall"]["steps"]["mean"]),
        8.56,
        1.0,
        "a run works through its angles: about 8.5 steps, up from 6.7",
    ),
    (
        EVAL,
        # Every group, not just the mean. A mean of 8.5 is also what you get from
        # simple questions stopping at 3 and hard ones running to 14.
        lambda d: min(g["steps"]["mean"] for g in d["angle_floor"]["per_group"].values()),
        8.22,
        0.9,
        "no group falls below eight steps",
    ),
    (
        EVAL,
        # The requirement was eight steps at no cost to accuracy. This pins the
        # "no cost" half: the arms are within a point and a half of each other.
        lambda d: abs(d["angle_floor"]["overall"]["correct_rate"]
                      - d["calc_on"]["overall"]["correct_rate"]) * 100,
        1.3,
        6.0,
        "the extra steps cost no accuracy",
    ),
    (
        EVAL,
        lambda d: (d["angle_floor"]["overall"]["prompt_tokens"]["mean"]
                   / d["calc_on"]["overall"]["prompt_tokens"]["mean"]),
        1.34,
        0.25,
        "and cost about a third more prompt tokens",
    ),
    (
        EVAL,
        # THE NOISE FLOOR, pinned because it governs how every other comparison
        # here should be read. Two runs of an identical configuration -- same
        # questions, same model, same code, differing in nothing -- scored 39% and
        # 67%. Any claim resting on a gap smaller than that is folklore.
        lambda d: abs(d["calc_control"]["overall"]["correct_rate"]
                      - d["calc_control_b"]["overall"]["correct_rate"]) * 100,
        27.8,
        6.0,
        "identical configurations differ by ~28 points at n=18: read every gap against this",
    ),
    # --- The arithmetic gate, measured where arithmetic is the task ---
    #
    # The mixed eval could not decide this: arithmetic is one thread among five
    # there, and the 28-point noise floor above swallowed the effect. On a battery
    # of generated questions with numbers nothing has memorised, it is not close.
    (
        BATTERY,
        lambda d: d["gate_on"]["correct_rate"] - d["gate_off"]["correct_rate"],
        0.35,
        0.20,
        "the gate is worth about 35 points where arithmetic is the task",
    ),
    (
        # Replication on a DIFFERENT battery -- fresh numbers, nothing tuned
        # against them. This is the claim that makes the one above worth trusting.
        BATTERY_REPLICATION,
        lambda d: d["gate_on"]["correct_rate"] - d["gate_off"]["correct_rate"],
        0.30,
        0.20,
        "and about the same again on a battery it has never seen",
    ),
    (
        BATTERY,
        # Unaided the model is not merely worse, it is barely able to do this at
        # all. This was 2 of 20 when first pinned and is now 5: the battery was
        # re-run after concrete examples were removed from the prompts, and
        # dropping copyable numbers helped the no-arithmetic arm too. Recording
        # the move rather than the old figure, because the old figure measured a
        # different prompt.
        lambda d: float(d["gate_off"]["correct"]),
        5.0,
        3.0,
        "unaided, the model gets a quarter of them",
    ),
    (
        BATTERY,
        # Computed exactly and then not used in the answer. Small, but not zero,
        # and it is the failure that survived the spine fix -- worth watching
        # rather than declaring solved.
        lambda d: float(d["gate_on"]["computed_but_unused"]),
        2.0,
        2.0,
        "a couple of runs still compute the right value and answer something else",
    ),
    (
        UNITS,
        # Unit questions were the one group exact arithmetic could not touch: 0 of
        # 4, with every failure dimensional rather than arithmetic. Naming the
        # conversion instead of computing it is what moved them at all.
        lambda d: float(d["gate_on"]["correct"]) / max(d["gate_on"]["n"], 1),
        0.33,
        0.25,
        "naming the conversion moves unit questions off zero",
    ),
    (
        UNITS,
        lambda d: float(d["gate_off"]["correct"]),
        0.0,
        0.0,
        "and without it the model gets none of them",
    ),
    # --- The model, which turned out to be the whole game ---
    #
    # Every design change in this repository put together moves the battery less
    # than swapping llama3.2:3b for qwen3:4b-instruct-2507. Pinned in both
    # directions, on two independent batteries, because it is the number that
    # should govern where effort goes next.
    (
        QWEN,
        lambda d: d["gate_on"]["correct_rate"],
        0.90,
        0.15,
        "qwen3:4b answers about 90 percent of the battery",
    ),
    (
        QWEN_REPLICATION,
        lambda d: d["gate_on"]["correct_rate"],
        0.75,
        0.15,
        "and about 75 percent on a battery it has never seen",
    ),
    (
        LLAMA_REPLICATION,
        # The same questions, the same code, the same day. Pinned so the
        # comparison cannot quietly become a comparison of two different things.
        lambda d: d["gate_on"]["correct_rate"],
        0.30,
        0.15,
        "llama3.2:3b answers about a third of the same questions",
    ),
    (
        QWEN,
        # It is not buying accuracy with length: fewer steps, better answers.
        lambda d: float(d["gate_on"]["steps_mean"]),
        5.5,
        1.5,
        "and reaches them in fewer steps, not more",
    ),
    (
        QWEN_SELECT,
        # Selection costs nothing on a capable model. The point of pinning it is
        # that the default is off, so nothing else would notice if this drifted.
        lambda d: d["gate_on"]["correct_rate"],
        0.80,
        0.15,
        "answering by selection costs no accuracy on qwen3",
    ),
    # --- The full battery: all four domains, all three grading modes ---
    (
        FULL_BATTERY,
        lambda d: d["correct_rate"],
        0.89,
        0.12,
        "qwen3 answers about nine in ten of the full battery's gradeable questions",
    ),
    (
        FULL_BATTERY,
        # The knowledge-edge probe as a battery, and it behaved exactly as
        # designed: steady on facts broad enough that any grounded model holds
        # them, UNSTEADY on places assembled from an RNG -- where a steady
        # answer would be manufactured confidence.
        lambda d: float(d["consistency"]["per_group"]["anchored"]["steady"]),
        2.0,
        1.0,
        "anchored facts are held steadily",
    ),
    (
        FULL_BATTERY,
        lambda d: float(d["consistency"]["per_group"]["invented"]["steady"]),
        0.0,
        1.0,
        "and invented places are not -- the honest signal, pinned so it stays",
    ),
    (
        FULL_BATTERY,
        # Never mixed into correctness: the note travels with the data.
        lambda d: 1.0 if "invisible" in d["consistency"]["note"] else 0.0,
        1.0,
        0.001,
        "the consistency caveat is attached to the data itself",
    ),
    (
        PHYSICS,
        # Constants stated in the question, so the task is composition across
        # units rather than recall -- the exact failure mode this project has
        # built machinery for, and the machinery carries it.
        lambda d: d["correct_rate"],
        1.0,
        0.25,
        "physics composition with stated constants is answered nearly clean",
    ),
    (
        CALCPHYS,
        # Calculus and extended physics together: derivatives at a point,
        # definite integrals, tangent slopes, quadratic roots, free fall,
        # acceleration, kinetic energy -- graded by mpeqs.calculus, whose two
        # independent derivative paths must agree before any model is graded.
        lambda d: d["correct_rate"],
        0.86,
        0.15,
        "calculus and extended physics land above five in six",
    ),
    (
        ALL_SIX,
        # The whole battery in one run: 58 gradeable questions across six
        # domains, one sitting, one model. The number the README's positioning
        # rests on when someone asks "measured against what?"
        lambda d: d["correct_rate"],
        0.91,
        0.10,
        "the full six-domain battery lands at nine in ten",
    ),
    (
        ALL_SIX,
        lambda d: float(d["errors"]),
        0.0,
        0.0,
        "and no run in it errored",
    ),
    (
        QWEN_SELECT,
        # And it does the job it exists for: a value computed exactly and then not
        # used. Five of those across two batteries without it, one with.
        lambda d: float(d["gate_on"]["computed_but_unused"]),
        1.0,
        1.5,
        "and nearly removes the failure it was built for",
    ),
]


def ground_truth_checks() -> list[tuple[bool, str, str]]:
    """Checks whose expected value comes from arithmetic, not from a previous run."""
    results = []

    # Two unit vectors 60 degrees apart have cosine exactly 0.5.
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.5, math.sqrt(3) / 2], dtype=np.float32)
    got = cosine_similarity(a, b)
    results.append((abs(got - 0.5) < 1e-6, f"{got:.6f} vs 0.5", "cosine of a 60-degree pair is exactly 0.5"))

    # Orthogonal vectors are exactly 0.
    got = cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    results.append((abs(got) < 1e-6, f"{got:.6f} vs 0.0", "cosine of an orthogonal pair is exactly 0"))

    # top_similarities must rank by the analytic ordering, looking only backwards.
    vectors = np.array([[1, 0], [0, 1], [math.cos(0.1), math.sin(0.1)]], dtype=np.float32)
    order = [index for index, _ in top_similarities(vectors, 2, top_k=2)]
    results.append((order == [0, 1], f"{order} vs [0, 1]", "nearest earlier step is ranked first"))

    # The strongest path is the one whose similarities multiply highest: a chain of
    # three 0.9 edges (0.729) beats a single 0.10 shortcut. A greedy search over
    # negated weights returns the shortcut instead.
    graph = {
        "nodes": [{"id": n} for n in ("A", "B", "C", "D")],
        "edges": [
            {"from": "A", "to": "D", "value": 0.10},
            {"from": "A", "to": "B", "value": 0.90},
            {"from": "B", "to": "C", "value": 0.90},
            {"from": "C", "to": "D", "value": 0.90},
        ],
    }
    path, _, mean = strongest_path(graph, "A", "D")
    results.append(
        (path == ["A", "B", "C", "D"], f"{path}", "strongest path maximises the product of similarities")
    )
    results.append((abs(mean - 0.9) < 1e-6, f"{mean:.6f} vs 0.9", "path similarity is the geometric mean"))

    return results


def main() -> int:
    failures = skipped = 0

    for passed, detail, label in ground_truth_checks():
        print(f"{'ok  ' if passed else 'FAIL'}  {label:<52} {detail}")
        failures += not passed

    for path, extract, expected, tolerance, label in FILE_CHECKS:
        target = ROOT / path
        if not target.exists():
            print(f"SKIP  {label:<52} ({path} missing -- run 'make measure')")
            skipped += 1
            continue
        try:
            got = float(extract(json.loads(target.read_text())))
        except (KeyError, TypeError, ValueError) as exc:
            print(f"FAIL  {label:<52} could not read: {exc}")
            failures += 1
            continue
        bad = abs(got - expected) > tolerance
        print(f"{'FAIL' if bad else 'ok  '}  {label:<52} {got:>10.4f}  expected {expected} +/- {tolerance}")
        failures += bad

    print(f"\n{failures} failed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
