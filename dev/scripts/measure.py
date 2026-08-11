#!/usr/bin/env python3
"""Measure how much the drawn edge weights actually vary, per embedding model.

The graph draws an edge for every pair of reasoning steps it thinks are related,
and labels it with a cosine similarity. If those similarities all land in a narrow
band the picture looks informative while carrying almost nothing, so this measures
the spread rather than assuming it.

Deliberately not driven by a live language model: the same fixed step texts go
through every embedding model, so the number describes the embedder and not the
sampling noise of whatever wrote the steps. Several independent topics are measured
so the observed spread across them can set the tolerance in check_numbers.py --
a single run would give a number with no error bar.

Writes docs/claims/edge_spread.json. Run with: make measure
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mpe_lkg import backends  # noqa: E402
from mpe_lkg.graph import build_graph, edge_weight_spread  # noqa: E402

# Four unrelated topics, each a plausible chain of reasoning steps.
TOPICS = {
    "cities": [
        "First I need to decide whether the question means city proper or metropolitan area.",
        "Using city proper populations, the largest are Tokyo, Delhi, Shanghai, Dhaka and Sao Paulo.",
        "Metropolitan definitions change the ranking, because they absorb surrounding municipalities.",
        "I should check whether the figures are recent, since urban populations move quickly.",
        "Sources disagree by several million for Delhi depending on the boundary used.",
        "Taking city proper and recent UN figures, Tokyo remains the largest.",
    ],
    "arithmetic": [
        "The problem asks for the product of two three-digit numbers.",
        "I will decompose 347 times 216 into partial products to reduce mistakes.",
        "347 times 200 is 69400, and 347 times 16 is 5552.",
        "Adding the partial products gives 74952.",
        "Checking by estimation, 350 times 216 is about 75600, which is close.",
        "The product is 74952.",
    ],
    "biology": [
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "The light dependent reactions happen in the thylakoid membrane and produce ATP and NADPH.",
        "The Calvin cycle then fixes carbon dioxide using that ATP and NADPH.",
        "An alternative framing would separate the oxygen evolving complex as its own stage.",
        "Temperature and light intensity both limit the overall rate.",
        "So photosynthesis is a two stage process linked by ATP and NADPH.",
    ],
    "logic": [
        "The statement is a conditional, so its truth depends only on the case where the premise holds.",
        "I should test whether the converse is being assumed anywhere in the argument.",
        "Affirming the consequent would be a fallacy here, and the argument appears to do that.",
        "Trying a counterexample: the premise can be false while the conclusion is true.",
        "That counterexample shows the inference is not valid.",
        "The argument is invalid because it affirms the consequent.",
    ],
}


def measure(backend) -> dict:
    per_topic = []
    for name, steps in TOPICS.items():
        vectors = backend.embed(steps)
        ids = [f"Step{i + 1}" for i in range(len(steps))]
        graph = build_graph(ids, ids, vectors, top_k=2)
        spread = edge_weight_spread(graph)
        spread["topic"] = name
        per_topic.append(spread)

    cvs = [t["cv"] for t in per_topic]
    means = [t["mean"] for t in per_topic]
    return {
        "model": backend.model,
        "dim": backend.dim,
        "topics": per_topic,
        "n": sum(t["n"] for t in per_topic),
        # Averaged over four independent topics, with the observed spread reported
        # so a tolerance can be set from it rather than guessed.
        "cv": statistics.fmean(cvs),
        "cv_stdev": statistics.stdev(cvs),
        "mean": statistics.fmean(means),
        "mean_stdev": statistics.stdev(means),
    }


def main() -> int:
    models = [m for m in backends.list_models() if m["is_embedding"]]
    if not models:
        print("No embedding model installed. Try:  ollama pull nomic-embed-text")
        return 1

    results = {
        "provenance": "measured",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topics": sorted(TOPICS),
        "steps_per_topic": len(next(iter(TOPICS.values()))),
    }

    for entry in models:
        name = entry["name"]
        print(f"measuring {name} ...", flush=True)
        try:
            result = measure(backends.OllamaEmbedding(name))
        except backends.BackendError as exc:
            print(f"  skipped: {exc}")
            continue
        key = name.split(":")[0]
        results[key] = result
        print(
            f"  dim={result['dim']}  mean={result['mean']:.4f}+/-{result['mean_stdev']:.4f}  "
            f"cv={result['cv']:.4f}+/-{result['cv_stdev']:.4f}  edges={result['n']}"
        )

    out = ROOT / "docs" / "claims" / "edge_spread.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
