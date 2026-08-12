# How it works

## The graph is load-bearing

Every claim below is a measurement pinned in `docs/claims/` and re-checked by CI on every
push — the numbers cannot drift from the code.

| what the graph does | measured |
|---|---|
| **writes the answer**: the strongest path through the steps is what the final answer is synthesised from | 54.9% → **91.2%** correct on the eval that motivated it |
| **steers exploration**: sub-questions that drift from the parent, or repeat one already asked, are dropped by embedding — decisions the model cannot make about itself | drift and repeats filtered in `explore`/`settle`, tested |
| **gates the answer type**: whether a numeric answer is even on the table is an embedding-relevance decision made by code, not by the model | the model chose a number for "capital of France" 3/3 without it |
| **hands arithmetic to [mpeqs](https://github.com/punnerud/MPEqs)**: expressions and unit conversions settle exactly, off the graph's facts | **+32.5 points** where arithmetic is the task, 95% CI [+15.3, +49.7], replicated |
| **accumulates knowledge**: every run lands in per-session SQL tables and RDF, queryable across time | sessions isolated, hints opt-in, `GET /sessions` |

Not a general knowledge-graph-builder for documents or RAG: the graph here is built *from
the model's own reasoning* and fed back into it.

| File | Responsibility |
|---|---|
| `src/mpe_lkg/app.py` | Flask routes and server-sent-event framing |
| `src/mpe_lkg/backends.py` | Chat and embedding backends, model discovery, health checks |
| `src/mpe_lkg/reasoning.py` | The step-by-step loop |
| `src/mpe_lkg/graph.py` | Similarity, graph construction, strongest path |
| `src/mpe_lkg/store.py` | SQLite storage and exact nearest-neighbour search |
| `src/mpe_lkg/layers.py` | Embeddings read from inside a model |

The strongest path maximises the product of the similarities along it, which is the same as
minimising a sum of `-log(similarity)`. Those costs are non-negative, so Dijkstra gives the
exactly optimal path, and the number reported is the geometric mean of the edges on it.

## Why the similarity search has no approximate index

The store keeps growing — it is no longer wiped between questions, so "Related Questions and
Answers" can actually surface earlier ones — which makes it fair to ask whether it needs an
ANN index. Measured with `scripts/bench_search.py` at 768 dimensions:

| Vectors | Exact scan (numpy) | Annoy query | Annoy build, per insert |
|---|---|---|---|
| 100 | 0.007 ms | 0.031 ms | 3.5 ms |
| 1 000 | 0.017 ms | 0.032 ms | 36 ms |
| 10 000 | 0.30 ms | 0.031 ms | 366 ms |
| 100 000 | 3.3 ms | 0.032 ms | 4 020 ms |

Three things follow.

**The exact scan is already fast enough at any plausible size.** Hundreds of vectors cost
about 0.02 ms, against an LLM call that takes seconds. Even a hundred thousand costs 3 ms.

**An Annoy index cannot be appended to.** It is immutable once built, and this app inserts
after every reasoning step, so the whole index has to be rebuilt on each one. That is the
last column, and it is worse than the exact scan at every size measured.

**On a current numpy the index returns wrong answers.** With `annoy` 1.17.3 and numpy 2.5.2 on
Python 3.12, `get_nns_by_item(7, 5)` returns `[1]` — one result instead of five, and not the
vector itself, which must always be its own nearest neighbour at distance zero. That is
reproducible in a clean environment built from the old `requirements.txt`, which means the
"Related Questions" panel was silently returning a single arbitrary row.

The last point is pinned as a check that fails if a future build ever starts behaving, so the
decision can be revisited rather than inherited. `tests/test_store.py` asserts exactness
directly: a vector is its own nearest neighbour, and the ranking matches a full brute-force
sort.

If the store ever does grow past a few hundred thousand vectors, the argument that changes
first is memory, not speed — 100 000 × 768 × 4 bytes is about 300 MB held in RAM — and the
answer then is a memory-mapped index, not a faster query.
