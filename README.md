# Local Knowledge Graph

[![Linux](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/linux.yml/badge.svg?branch=main)](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/linux.yml)
[![macOS](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/macos.yml/badge.svg?branch=main)](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/macos.yml)
[![Windows](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/windows.yml/badge.svg?branch=main)](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/windows.yml)
[![PyPI](https://img.shields.io/pypi/v/mpe-lkg.svg)](https://pypi.org/project/mpe-lkg/)
[![Python](https://img.shields.io/pypi/pyversions/mpe-lkg.svg)](https://pypi.org/project/mpe-lkg/)

<!-- Absolute, not docs/example.png: PyPI renders this README standalone on
     pypi.org, where a relative path resolves against pypi.org and 404s. -->
![Example](https://raw.githubusercontent.com/punnerud/Local_Knowledge_Graph/main/docs/example.png)

Ask a local model a question, watch it reason step by step, and see the steps drawn as a graph
where the edges are how similar the steps are to each other.

The run starts by breaking the question into angles worth checking, then works through them —
about eight steps, each with a named job rather than a quota to fill. Any arithmetic a step
relies on is handed over as an expression and evaluated exactly, in fractions, by
[mpeqs](https://github.com/punnerud/MPEqs). Those sums are shown above the graph: they are the
one part of a run the model did not decide, and you can check `20-13.5 = 6.5` at a glance in a
way you cannot check a paragraph of reasoning.

The graph draws in two colours, because it holds two kinds of claim. **Blue** links steps by
how similar their embeddings are — an association, with no truth value, and the thing that
makes indirect knowledge visible. **Green** is what an exact evaluator settled: a sum, or a
conversion between two units, derived from exact ratios. A reader should never have to guess
which is which.

Everything runs on your machine. Nothing is uploaded anywhere.

## Headless, and as RDF

A run can be started without a browser, polled, and taken as RDF — the graph as text, for
anything that would rather query it than look at it.

```bash
ID=$(curl -sX POST localhost:5100/jobs -H 'content-type: application/json' \
       -d '{"query":"How many seconds are there in 23 weeks?"}' | jq -r .id)

curl -s localhost:5100/jobs/$ID              # {"state":"running","steps":3,...}
curl -s localhost:5100/jobs/$ID/stream       # N-Triples, live, one triple per line
curl -s localhost:5100/jobs/$ID/rdf          # Turtle, once it has finished
curl -sX DELETE localhost:5100/jobs/$ID      # stop it
```

Two formats for two purposes. **N-Triples** streams: each line is a complete document, so a
consumer can parse what has arrived without waiting for the end. **Turtle** is prefixed and
readable, and needs the whole document, so it is what a finished run serialises to.

The `lkg:basis` predicate carries the same distinction as the colours, so a consumer can take
only the part it can rely on:

```turtle
<run/a1b2/link/Step1-Step2>  lkg:similarity "0.8371"^^xsd:decimal ;
                             lkg:basis      lkg:Embedding .     # measured association

<run/a1b2/conversion/1>      lkg:statement  "23 week = 13910400 second" ;
                             lkg:from       <unit/week> ;
                             lkg:to         <unit/second> ;
                             lkg:basis      lkg:Exact .         # derived, reproducible
```

## Run it

```bash
pip install mpe-lkg
mpe-lkg
```

Then open <http://localhost:5100>.

**Which model matters more than anything else here.** Measured on the same 40 generated
arithmetic questions, `qwen3:4b-instruct-2507` answers **82.5%** against `llama3.2:3b`'s
**40%** — +42.5 points, 95% CI [+23.3, +61.7], replicated on a second battery — and does it in
fewer steps, not more. That is a larger gain than every prompt and design change in this
repository put together, so it is worth spending 2.5 GB on before spending an evening on
prompts:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

It needs a local model, which it reaches through [Ollama](https://ollama.com). **You do not
need to work that out from here** — start it and it will tell you what it found, what is
missing, and the one command that fixes it. `mpe-lkg doctor` reports the same thing without
starting the server, and exits non-zero, so it works in a script.

Python 3.10 or newer. The wheel is `py3-none-any`, so nothing is compiled and the same
artefact serves Linux, macOS and Windows — all three tested on every push.

<details>
<summary>From a clone, or from Python</summary>

```bash
git clone https://github.com/punnerud/Local_Knowledge_Graph
cd Local_Knowledge_Graph
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/mpe-lkg
```

`python app.py` still works from a clone as it always has.

```python
from mpe_lkg import create_app, health

print(health())
create_app().run(port=5100)
```

</details>

## More

| | |
|---|---|
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
