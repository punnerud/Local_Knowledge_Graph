# Development

```bash
make venv          # uv-based environment, including a headless browser for the render tests
make all           # ruff + pytest + the documented numbers, in one gate
make test          # unit, stream and browser tests; no model needed
make test-ollama   # the tests that need a live Ollama
make measure       # re-measure the embedding-model table into docs/claims/
make sweep         # re-measure the per-layer separation table (needs torch)
make bench         # exact scan vs an approximate index, at several store sizes
make loops         # is repetition actually the problem, or does the model drift?
```

`make all` runs `scripts/check_numbers.py`, which resolves every measurable claim in these
documents to a value in `docs/claims/`. If a number stops being true the build fails, instead
of the documentation quietly becoming wrong. Some of its checks are ground truths computed
from arithmetic rather than from a previous run, because a consistency gate cannot detect a
consistent error.

Every measurement script writes both its summary and the raw material it was computed from, so
a number can be re-derived rather than taken on trust.

## Does the model actually loop?

`make loops` runs 24 questions across four kinds — simple, multi-step, malformed, ambiguous —
and separates three failure modes, because they need different fixes:

| | over 153 steps in 24 runs |
|---|---|
| **repeat** — says something an earlier step already said | 4.6 % |
| **drift** — new words, no new subject matter | 3.9 % |
| neither | 91.5 % |

Per step that looks small. Per *question* it is not: repeats cluster, so **a quarter of
questions produce at least one**, and one run produced four.

Two things this measurement had to get right before any of it could be believed.

**It audits itself first.** A near-identical study elsewhere reported a 58 % loop rate on its
first run, and every one of those repeats was the step extractor mistaking a code fence for a
step and then seeing it again — a measurement of the harness that nearly became a finding
about the model. So the normalisation is checked for manufacturing duplicates before its
numbers are used, and that check is pinned in `check_numbers.py`.

**The threshold came from the distribution, not from taste.** The first attempt used cosine
≥ 0.90 and reported 21 %. But the median step already sits at 0.846 similarity to some earlier
step and the 75th percentile at 0.900, so that bar flags the more-similar quarter of *ordinary*
steps. Inspection confirmed it: "we need to calculate a percentage" followed by "the formula is
(17/100) × 250" scored 0.9013, and is plain progress. At 0.96 the flags are restatements —
including two steps that were character-identical to an earlier one. The title bar is
insensitive by comparison: exactly the same three steps fire anywhere from 0.95 to 0.999.

`docs/claims/loops_transcripts.json` holds every raw step, and `make loops --reanalyse`
recomputes from it, so a threshold can be revisited without spending model time and anyone can
check these numbers instead of taking them.
