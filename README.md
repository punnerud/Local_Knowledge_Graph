# Local Knowledge Graph

[![Linux](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/linux.yml/badge.svg?branch=main)](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/linux.yml)
[![macOS](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/macos.yml/badge.svg?branch=main)](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/macos.yml)
[![Windows](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/windows.yml/badge.svg?branch=main)](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/windows.yml)
[![PyPI](https://img.shields.io/pypi/v/mpe-lkg.svg)](https://pypi.org/project/mpe-lkg/)
[![Python](https://img.shields.io/pypi/pyversions/mpe-lkg.svg)](https://pypi.org/project/mpe-lkg/)

<!-- Absolute, not docs/example.png: PyPI renders this README standalone on
     pypi.org, where a relative path resolves against pypi.org and 404s. -->
![Example](https://raw.githubusercontent.com/punnerud/Local_Knowledge_Graph/main/docs/example.png)

Ask a local model a question, watch it reason step by step — and see the reasoning as a
knowledge graph that is **used, not just drawn**. **Blue** edges are embedding similarity —
association. **Green** is what an exact evaluator settled — sums and unit conversions, in
fractions, via [mpeqs](https://github.com/punnerud/MPEqs). Everything runs on your machine.

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

## Run it

```bash
pip install mpe-lkg
mpe-lkg
```

Open <http://localhost:5100>. It talks to a local model through
[Ollama](https://ollama.com) and will tell you what is missing and the one command that
fixes it. The model matters more than anything else here — measured, `qwen3:4b-instruct-2507`
answers 82.5% of the arithmetic battery against `llama3.2:3b`'s 40%:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Ask harder questions with `mode=explore` (each sub-question answered by its own run) or
`mode=settle` (explored twice, finished only when two independent runs agree), and take any
run as RDF:

```bash
curl -sX POST localhost:5100/jobs -H 'content-type: application/json' \
     -d '{"query":"How many seconds are there in 23 weeks?","mode":"settle"}'
curl -s localhost:5100/jobs/<id>/rdf
```

## More

| | |
|---|---|
| [Modes, jobs and RDF](docs/modes.md) | reason / explore / settle, the vote, the probes, the vocabulary |
| [Models and configuration](docs/models.md) | Choosing models, every environment variable, troubleshooting |
| [Embeddings from inside a model](docs/internal-layers.md) | Reading a chosen layer instead of an embedding endpoint |
| [How it works](docs/design.md) | The modules, the strongest-path search, why there is no ANN index |
| [Development](docs/development.md) | Tests, and the gate that checks this documentation against measured data |

## Licence

[The mpedb License 1.0](LICENSE) — the same licence as
[mpedb](https://github.com/punnerud/mpedb) and [MPEqs](https://github.com/punnerud/MPEqs),
byte for byte.

Free of charge for every person and every organization, with one exception: a group whose
revenue or valuation exceeds five billion dollars owes a one-time fee of seven US cents per
device. Not an OSI-approved licence.

---

<sub>Published to PyPI as `mpe-lkg` — **M**orten **P**unnerud-**E**ngelstad **L**ocal
**K**nowledge **G**raph.</sub>
