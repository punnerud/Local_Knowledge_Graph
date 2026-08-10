#!/usr/bin/env python3
"""Where does an approximate index start beating an exact scan, here?

Two different shapes of work happen in this app and they do not have the same
answer, so both are measured:

* all-pairs inside one reasoning chain -- every step against every earlier step,
  which is what draws the edges. N is the number of steps, so tens.
* top-k against the whole store -- one query vector against everything ever
  saved, which is what fills the "Related Questions" panel. N grows without
  bound if the store is not wiped between questions.

Also measures recall, because an approximate index that is faster and wrong is
not faster.

    .venv/bin/python scripts/bench_search.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

try:
    from annoy import AnnoyIndex
except ImportError:  # pragma: no cover - optional benchmark dependency
    AnnoyIndex = None

DIM = 768
SIZES = [100, 1_000, 10_000, 100_000]
TOP_K = 5
N_TREES = 10


def timed(fn, repeats: int = 5) -> float:
    """Best-of-N wall clock in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000


def make_vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, DIM)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def numpy_topk(store: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    scores = store @ query
    # argpartition is O(n); a full sort would be O(n log n) for no reason.
    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    return top[np.argsort(-scores[top])]


def bench_all_pairs() -> None:
    print("\nAll-pairs similarity inside one reasoning chain")
    print(f"{'steps':>8}  {'numpy full matrix':>20}")
    for n in (10, 20, 50, 100):
        vectors = make_vectors(n)
        ms = timed(lambda v=vectors: v @ v.T)
        print(f"{n:>8}  {ms:>17.3f} ms")


def bench_topk() -> None:
    print("\nTop-k against the whole store")
    header = f"{'vectors':>9}  {'numpy exact':>12}  {'annoy build':>12}  {'annoy query':>12}  {'annoy recall':>12}"
    print(header)

    for n in SIZES:
        store = make_vectors(n)
        query = make_vectors(1, seed=99)[0]

        numpy_ms = timed(lambda s=store, q=query: numpy_topk(s, q, TOP_K))
        exact = set(numpy_topk(store, query, TOP_K).tolist())

        if AnnoyIndex is None:
            print(f"{n:>9}  {numpy_ms:>9.3f} ms  {'(annoy not installed)':>40}")
            continue

        def build(s=store):
            index = AnnoyIndex(DIM, "angular")
            for i, row in enumerate(s):
                index.add_item(i, row)
            index.build(N_TREES)
            return index

        build_ms = timed(build, repeats=1)
        index = build()
        query_ms = timed(lambda idx=index, q=query: idx.get_nns_by_vector(q, TOP_K))
        approx = set(index.get_nns_by_vector(query, TOP_K))
        recall = len(exact & approx) / len(exact)

        print(
            f"{n:>9}  {numpy_ms:>9.3f} ms  {build_ms:>9.1f} ms  {query_ms:>9.3f} ms  {recall:>11.0%}"
        )


def bench_amortised() -> None:
    """The index has to be rebuilt whenever a vector is added.

    This app appends after every reasoning step, so the build cost is not paid
    once -- it is paid again on every insert unless the index is kept incremental,
    which Annoy cannot do: an Annoy index is immutable once built.
    """
    if AnnoyIndex is None:
        return
    print("\nCost of one insert followed by one query (Annoy must rebuild)")
    print(f"{'vectors':>9}  {'numpy exact':>12}  {'annoy rebuild+query':>21}")
    for n in SIZES:
        store = make_vectors(n)
        query = make_vectors(1, seed=99)[0]
        numpy_ms = timed(lambda s=store, q=query: numpy_topk(s, q, TOP_K))

        def rebuild_and_query(s=store, q=query):
            index = AnnoyIndex(DIM, "angular")
            for i, row in enumerate(s):
                index.add_item(i, row)
            index.build(N_TREES)
            return index.get_nns_by_vector(q, TOP_K)

        annoy_ms = timed(rebuild_and_query, repeats=1)
        print(f"{n:>9}  {numpy_ms:>9.3f} ms  {annoy_ms:>18.1f} ms")


def write_claims() -> None:
    """Record the numbers the README quotes, so they can be checked rather than trusted."""
    import json
    import pathlib
    import platform

    store_1k = make_vectors(1_000)
    store_100k = make_vectors(100_000)
    query = make_vectors(1, seed=99)[0]
    chain = make_vectors(20)

    claims = {
        "provenance": "measured",
        "machine": f"{platform.system()} {platform.machine()} python{platform.python_version()}",
        "dim": DIM,
        "top_k": TOP_K,
        "all_pairs_20_steps_ms": timed(lambda: chain @ chain.T),
        "exact_topk_1k_ms": timed(lambda: numpy_topk(store_1k, query, TOP_K)),
        "exact_topk_100k_ms": timed(lambda: numpy_topk(store_100k, query, TOP_K)),
    }

    if AnnoyIndex is not None:
        # Does the approximate index return the right answer at all? A vector's
        # nearest neighbour is itself, at distance zero. On this platform the
        # installed build fails that, which is why it is not a dependency.
        probe = make_vectors(100)
        index = AnnoyIndex(DIM, "angular")
        for i, row in enumerate(probe):
            index.add_item(i, row)
        index.build(N_TREES)
        neighbours = index.get_nns_by_item(7, TOP_K)
        claims["annoy_returns_self_first"] = bool(neighbours and neighbours[0] == 7)
        claims["annoy_returns_k_results"] = len(neighbours)
        claims["annoy_version"] = "1.17.3"

    out = pathlib.Path(__file__).resolve().parent.parent / "docs" / "claims" / "search_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(claims, indent=2) + "\n")
    print(f"\nwrote {out.name}")


def main() -> int:
    print(f"dim={DIM}  top_k={TOP_K}  n_trees={N_TREES}  numpy={np.__version__}")
    bench_all_pairs()
    bench_topk()
    bench_amortised()
    write_claims()
    return 0


if __name__ == "__main__":
    sys.exit(main())
