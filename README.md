# Local Knowledge Graph

![Example](example.png)

Ask a local Llama model a question, watch it reason step by step, and see the steps drawn as a
knowledge graph where the edges are the semantic similarity between them.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) running locally
- One chat model and, ideally, one embedding model

```bash
ollama pull llama3.2:3b        # or llama3.1:8b, or any chat model you already have
ollama pull nomic-embed-text   # optional but recommended, see "Which embedding model" below
```

## Installation

```bash
git clone https://github.com/punnerud/Local_Knowledge_Graph
cd Local_Knowledge_Graph
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Then open <http://localhost:5100>.

If the app cannot find what it needs it says so on startup and at
<http://localhost:5100/health>, naming the exact `ollama pull` command that fixes it.

## Configuration

Everything is an environment variable, and the defaults work unchanged.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is listening |
| `LKG_CHAT_MODEL` | `llama3.1:8b` | The model that does the reasoning |
| `LKG_EMBED_MODEL` | *(auto)* | Embedding model. Empty means: use an installed embedding model if there is one, otherwise fall back to the chat model |
| `LKG_HOST` / `LKG_PORT` | `127.0.0.1` / `5100` | Where the app listens |
| `LKG_DEBUG` | off | Set to `1` for the Flask debugger. Do not do this on a shared network |

## Which embedding model, and why it matters

The edges in the graph are cosine similarities, so how much they vary decides whether the
picture tells you anything. Measured over four unrelated six-step reasoning chains
(`make measure`, recorded in `data/claims/edge_spread.json`):

| Model | Dimensions | Mean edge weight | Coefficient of variation |
|---|---|---|---|
| `all-minilm` | 384 | 0.48 ± 0.07 | **0.28 ± 0.11** |
| `nomic-embed-text` | 768 | 0.67 ± 0.05 | **0.13 ± 0.03** |

The uncertainties are the spread across the four topics. The larger model produces the
*less* discriminative graph here: under `nomic-embed-text` almost every pair of reasoning
steps scores around 0.67, so the edge labels stop distinguishing anything. This is ordinary
distance concentration, and it is a good reason to look at the spread rather than trusting
that a better retrieval model draws a better graph. `all-minilm` is the better default for
the *drawing* even though it is the weaker retriever.

Both work. Any embedding size works — nothing in the code assumes a dimension.

## Development

```bash
make venv          # uv-based environment, including a headless browser for the render tests
make all           # ruff + pytest + the documented numbers, in one gate
make test          # unit, stream and browser tests; no model needed
make test-ollama   # the tests that need a live Ollama
make measure       # re-measure the table above into data/claims/
```

`make all` runs `scripts/check_numbers.py`, which resolves every measurable claim in this
README to a value in `data/claims/`. If a number here stops being true, the build fails
instead of the README quietly becoming wrong. Some of its checks are ground truths computed
from arithmetic rather than from a previous run, because a consistency gate cannot detect a
consistent error.

## Troubleshooting

**The page stays blank when I submit.**
Open <http://localhost:5100/health>. It reports whether Ollama answered, which models are
installed, and what to pull. Errors are now shown in the page itself rather than only in the
browser console.

**It says a model is not found.**
The default chat model is `llama3.1:8b`. If you have a different one, either pull that, or
set `LKG_CHAT_MODEL` to a model you already have.

**Ollama runs in Docker or on another machine.**
Set `OLLAMA_URL`, and make sure Ollama binds beyond localhost (`OLLAMA_HOST=0.0.0.0`).

**It seemed to hang and never printed anything.**
That was a real bug: two retry paths could loop forever without ever sending anything to the
browser. Both are bounded now, and the stream sends a heartbeat while the model is thinking.

## How it works

| File | Responsibility |
|---|---|
| `app.py` | Flask routes and server-sent-event framing |
| `backends.py` | Chat and embedding backends, model discovery, health checks |
| `reasoning.py` | The step-by-step loop |
| `graph.py` | Similarity, graph construction, strongest path |
| `store.py` | SQLite storage and exact nearest-neighbour search |

The strongest path maximises the product of the similarities along it, which is the same as
minimising a sum of `-log(similarity)`. Those costs are non-negative, so Dijkstra gives the
exactly optimal path, and the number reported is the geometric mean of the edges on it.

Similarity search is a full scan in numpy. An approximate index only starts paying for itself
somewhere around a hundred thousand vectors, and a single query here produces about ten.
