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

## Does the thinking improve the answer?

`make eval` runs 20 questions with known answers and reports whether the answer survived, how
many steps it took, and how many prompt tokens Ollama actually tokenised. Every answer is
saved, so the grading can be checked and old arms regraded without re-running the model.

Measured against `llama3.2:3b`, gradeable groups only:

| arm | correct | steps | prompt tokens | seconds |
|---|---|---|---|---|
| what this project shipped | 54.9 % | 6.2 | 2704 | 7.5 |
| **now, validated over 80 runs** | **91.2 %** | **3.9** | **1649** | **5.2** |

Two changes account for it, and neither is the one that was planned.

**The answer is written from the graph.** It used to be the text of whatever step came last,
so the answer to "What is the capital of France?" did not contain the word Paris — it was a
footnote about regional capitals, because a prompt that rewards exploring alternatives ends on
a caveat. Now the strongest path through the graph is extracted and one call turns that thread
into an answer. This is the graph earning its place rather than being drawn and ignored.

**The step floor was removed.** `min_steps = 5` forced every question to five steps whether or
not it had five steps' worth of thinking in it. Removing it halved both the step count and the
token cost with no loss of accuracy — and it also removed most of the repetition the loop
detector was built for, because those repeats lived in steps 5 to 8. We had been detecting a
problem we were causing.

Three things were tried and **measured worse**, and are recorded here so they are not
rediscovered:

- **A shorter system prompt.** The 233-token original reads like shouting and is resent every
  call. Three terser rewrites all lost, by 9 to 15 points. A control that put a step floor back
  showed it is the wording and not the amount of reasoning: at 3.7 steps against the long
  prompt's 3.5, the short version still lost. What it buys is arithmetic.
- **The repeat detector, as a contributor.** It has never fired in an eval run. It stays as
  insurance for the pathological case, tested against a model that repeats one move forever.
- **Grading ambiguous questions by substring.** "A hot dog is a sandwich." is a good answer and
  matches no keyword list worth writing, while a rambling hedge scores well by accident. Those
  questions still run; they are reported separately rather than averaged into a headline they
  would move for the wrong reason.

A number that is only measured once is not measured. A two-run arm read 97.1 % where four runs
read 91.2 %, which is why the shipped figure is the four-run one.


## How much does a run vary, run to run?

Two arms in this file turned out to be the same configuration measured twice —
`decompose8` was run before the arithmetic gate existed, and `arith_off` runs the same
settings with the gate explicitly disabled. They scored **84 % and 74 %**.

That is the noise floor: **about ten points at two runs per question**, on 25 gradeable
questions against a 3B model at temperature 0.2. It was not designed as a replication and is
more useful for having been an accident.

Read every comparison here against it. The large results survive — 55 % to 91 % is five times
the noise — but a four-point difference between two arms is not a result, and this document
does not treat one as such.

## Checking the model's arithmetic: correct, and nearly inert

`mpeqs` evaluates a sum exactly and refuses rather than guessing, so a step claiming
`14 * 24 * 60 = 10080` can be contradicted without asking the model to check itself. That is
wired in, tested, and it works when it fires.

It hardly ever fires. Measured: **one correction across 50 runs**, and a scan of 153 saved
reasoning steps found **11 checkable sums and none of them wrong**.

The reason is worth recording, because it points at what would work. This model writes its
arithmetic as prose — "a fortnight is 14 days, and each day has 1440 minutes, so 20160" —
rather than as `14 * 1440 = 20160`. An extractor keyed on an explicit expression sees almost
nothing, and the errors that did occur were stated in the *answer* as a bare number with no
working to check.

So the gate stays, described as what it is: correct when it fires, and not a measurable
contributor on this model's output style. Getting real value out of an exact solver means
asking the model for a **spec** it can evaluate — `{"solver": "arith", "answer": "(17/100)*250"}` —
rather than scraping prose for something that looks like a sum. That is a different experiment.
