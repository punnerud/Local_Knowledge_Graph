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

Everything runs on your machine. Nothing is uploaded anywhere.

## Run it

```bash
pip install mpe-lkg
mpe-lkg
```

Then open <http://localhost:5100>.

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
