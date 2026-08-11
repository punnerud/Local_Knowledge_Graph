#!/usr/bin/env python3
"""How well does each layer of a model separate one topic from another?

The point of reading inside a model is that "the embedding" is not one thing. This
sweeps a set of layers over the same fixed reasoning-step corpus and reports, per
layer, how far apart it puts steps from the same topic versus steps from different
topics. That separation is the quantity that decides whether a graph drawn from
that layer means anything.

The corpus is the one in scripts/measure.py: four unrelated topics of six steps
each. Same texts through every layer, so the number describes the layer.

    .venv/bin/python scripts/layer_sweep.py [model] [layer ...]
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

import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from measure import TOPICS  # noqa: E402

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def separation(vectors: np.ndarray, labels: list[str]) -> dict:
    """Mean within-topic similarity minus mean across-topic similarity.

    A layer that scores near zero is not distinguishing the topics at all, whatever
    its absolute similarities look like.
    """
    similarity = vectors @ vectors.T
    same, different = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (same if labels[i] == labels[j] else different).append(float(similarity[i, j]))

    within = statistics.fmean(same)
    across = statistics.fmean(different)
    return {
        "within_topic": within,
        "across_topic": across,
        "separation": within - across,
        "mean_abs": float(np.abs(similarity[np.triu_indices(len(labels), 1)]).mean()),
    }


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    requested = sys.argv[2:]

    try:
        from mpe_lkg.layers import MultiLayerProbe, describe_layers
    except ImportError as exc:
        print(exc)
        return 1

    texts, labels = [], []
    for topic, steps in TOPICS.items():
        texts.extend(steps)
        labels.extend([topic] * len(steps))

    if not requested:
        from mpe_lkg.layers import HiddenStateEmbedding

        probe = HiddenStateEmbedding(model_name, layer="blocks.-1")
        n = describe_layers(probe.model)["n_blocks"]
        # Evenly spaced through the stack, always including the first and the last.
        requested = [f"blocks.{i}" for i in sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})]
        del probe

    print(f"model={model_name}  texts={len(texts)}  topics={len(TOPICS)}")
    multi = MultiLayerProbe(model_name, requested)
    info = describe_layers(multi.probes[requested[0]].model)
    print(f"block stack: {info['block_stack']} ({info['n_blocks']} x {info['block_type']}), "
          f"final norm applied: {info['has_final_norm']}\n")

    results = {}
    print(f"{'layer':>12}  {'within':>8}  {'across':>8}  {'separation':>11}")
    for layer, vectors in multi.embed(texts).items():
        stats = separation(vectors, labels)
        results[layer] = stats
        print(f"{layer:>12}  {stats['within_topic']:>8.4f}  {stats['across_topic']:>8.4f}  "
              f"{stats['separation']:>11.4f}")

    best = max(results, key=lambda k: results[k]["separation"])
    print(f"\nbest separation: {best} ({results[best]['separation']:.4f})")

    payload = {
        "provenance": "measured",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "n_blocks": info["n_blocks"],
        "dim": int(multi.probes[requested[0]].dim),
        "pooling": multi.probes[requested[0]].pooling,
        "final_norm_applied": info["has_final_norm"],
        "layers": results,
        "best_layer": best,
        "best_separation": results[best]["separation"],
        "first_layer_separation": results[requested[0]]["separation"],
    }
    out = ROOT / "docs" / "claims" / "layer_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
