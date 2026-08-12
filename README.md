# Local Knowledge Graph

[![Linux](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/linux.yml/badge.svg?branch=main)](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/linux.yml)
[![macOS](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/macos.yml/badge.svg?branch=main)](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/macos.yml)
[![Windows](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/windows.yml/badge.svg?branch=main)](https://github.com/punnerud/Local_Knowledge_Graph/actions/workflows/windows.yml)
[![PyPI](https://img.shields.io/pypi/v/mpe-lkg.svg)](https://pypi.org/project/mpe-lkg/)
[![Python](https://img.shields.io/pypi/pyversions/mpe-lkg.svg)](https://pypi.org/project/mpe-lkg/)

<!-- Absolute, not docs/example.png: PyPI renders this README standalone on
     pypi.org, where a relative path resolves against pypi.org and 404s. -->
![Example](https://raw.githubusercontent.com/punnerud/Local_Knowledge_Graph/main/docs/example.png)

A local model reasons step by step, and the reasoning becomes a knowledge graph that is
[**used, not just drawn**](docs/design.md#the-graph-is-load-bearing). Everything runs on your machine.

## Run it

```bash
pip install mpe-lkg
mpe-lkg
```

Open <http://localhost:5100>. It talks to a local model through
[Ollama](https://ollama.com) and will tell you what is missing and the one command that
fixes it. The model matters more than anything else — measured, `qwen3:4b-instruct-2507`
answers 82.5% of the arithmetic battery against `llama3.2:3b`'s 40%:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Harder questions: `mode=explore` answers each sub-question with its own run; `mode=settle`
finishes only when two independent runs agree. Any run can be taken as RDF:

```bash
curl -sX POST localhost:5100/jobs -H 'content-type: application/json' \
     -d '{"query":"How many seconds are there in 23 weeks?","mode":"settle"}'
curl -s localhost:5100/jobs/<id>/rdf
```

## More

| | |
|---|---|
| [How it works, and what the graph is for](docs/design.md) | the load-bearing measurements, the modules, the strongest-path search |
| [Modes, jobs and RDF](docs/modes.md) | reason / explore / settle, the vote, the probes, the vocabulary |
| [Models and configuration](docs/models.md) | choosing models, every environment variable, troubleshooting |
| [Embeddings from inside a model](docs/internal-layers.md) | reading a chosen layer instead of an embedding endpoint |
| [Development](docs/development.md) | tests, the battery, and the gate that checks this documentation against measured data |

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
