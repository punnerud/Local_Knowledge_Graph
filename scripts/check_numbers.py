#!/usr/bin/env python3
"""Fail loudly when a documented number stops being true.

Prose drifts away from data silently. This turns every measurable claim in the
README into a lookup against the JSON that produced it, so a claim that stops
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mpe_lkg.graph import cosine_similarity, strongest_path, top_similarities  # noqa: E402

CLAIMS = "docs/claims/edge_spread.json"
SEARCH = "docs/claims/search_bench.json"
SWEEP = "docs/claims/layer_sweep.json"

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
