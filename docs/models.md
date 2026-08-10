# Models and configuration

The app picks whichever chat and embedding models Ollama reports, and the page has a
dropdown for each. These variables override that; the defaults work unchanged.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is listening |
| `LKG_CHAT_MODEL` | *(auto)* | The model that does the reasoning. Empty means: use an installed chat model. This is an override, not a default |
| `LKG_EMBED_MODEL` | *(auto)* | Embedding model. Empty means: use an installed embedding model if there is one, otherwise fall back to the chat model |
| `LKG_HOST` / `LKG_PORT` | `127.0.0.1` / `5100` | Where the app listens |
| `LKG_DEBUG` | off | Set to `1` for the Flask debugger. Do not do this on a shared network |

## Which embedding model, and why it matters

The edges in the graph are cosine similarities, so how much they vary decides whether the
picture tells you anything. Measured over four unrelated six-step reasoning chains
(`make measure`, recorded in `docs/claims/edge_spread.json`):

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

## Troubleshooting

**The page stays blank when I submit.**
Open <http://localhost:5100/health>. It reports whether Ollama answered, which models are
installed, and what to pull. Errors are now shown in the page itself rather than only in the
browser console.

**It says a model is not found.**
It should not: the app picks whichever chat and embedding models Ollama actually reports, and
the page has a dropdown for each. If Ollama has no chat model at all, the page lists a few with
their download sizes and can fetch one for you.

**Ollama runs in Docker or on another machine.**
Set `OLLAMA_URL`, and make sure Ollama binds beyond localhost (`OLLAMA_HOST=0.0.0.0`).

**It seemed to hang and never printed anything.**
That was a real bug: two retry paths could loop forever without ever sending anything to the
browser. Both are bounded now, and the stream sends a heartbeat while the model is thinking.
