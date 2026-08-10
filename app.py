"""Local Knowledge Graph -- Flask entry point.

Thin by design: routing, server-sent-event framing, and process configuration. The
model backends live in ``backends.py``, the graph maths in ``graph.py``, the storage
in ``store.py``, and the reasoning loop in ``reasoning.py``.
"""

from __future__ import annotations

import json
import os
import queue
import threading

from flask import Flask, Response, jsonify, render_template, request

import backends
from reasoning import reason
from store import EmbeddingStore

app = Flask(__name__)

DB_PATH = os.environ.get("LKG_DB", "embeddings.db")
# How long to wait for the next event before telling the browser we are still alive.
HEARTBEAT_SECONDS = float(os.environ.get("LKG_HEARTBEAT", "5"))


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def make_backends() -> tuple:
    """Build the default backends. Tests replace this.

    Set LKG_EMBED_BACKEND=hf to embed from inside a local model instead of from an
    embedding endpoint, which is what makes a model with no embedding API usable and
    lets you point at a specific depth:

        LKG_EMBED_BACKEND=hf LKG_HF_MODEL=HuggingFaceTB/SmolLM2-135M \
        LKG_HF_LAYER=blocks.-1 python app.py
    """
    chat = backends.OllamaChat(backends.DEFAULT_CHAT_MODEL)

    if os.environ.get("LKG_EMBED_BACKEND") == "hf":
        from layers import HiddenStateEmbedding

        return chat, HiddenStateEmbedding(
            os.environ.get("LKG_HF_MODEL", "HuggingFaceTB/SmolLM2-135M"),
            layer=os.environ.get("LKG_HF_LAYER", "blocks.-1"),
            pooling=os.environ.get("LKG_HF_POOLING", "last"),
        )

    return chat, backends.OllamaEmbedding(backends.DEFAULT_EMBED_MODEL)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Why nothing is happening, in terms a user can act on."""
    return jsonify(backends.health(backends.DEFAULT_BASE_URL))


@app.route("/query", methods=["GET", "POST"])
def query():
    if request.method == "POST":
        user_query = (request.json or {}).get("query", "")
    else:
        user_query = request.args.get("query", "")
    user_query = user_query.strip()

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    chat, embedder = app.config.get("BACKENDS_FACTORY", make_backends)()

    def generate():
        """Stream events, and keep talking even while the model is thinking.

        The reasoning loop runs on a worker thread and hands events over a queue, so
        this generator can emit a heartbeat during the long silence before a slow
        model produces its first token. Without it a browser sees an open connection
        and no data, which is indistinguishable from the app being broken.
        """
        store = EmbeddingStore(app.config.get("DB_PATH", DB_PATH))
        events: queue.Queue = queue.Queue()
        sentinel = object()

        def worker():
            try:
                # The store deliberately accumulates across questions. It used to be
                # wiped at the start of every request, which made "Related Questions
                # and Answers" structurally unable to show anything but the current
                # run's own steps. Set LKG_RESET_DB=1 to go back to a clean slate.
                if os.environ.get("LKG_RESET_DB") == "1":
                    store.clear()
                query_vector = embedder.embed([user_query])[0]
                query_id = store.add(
                    user_query,
                    query_vector,
                    is_question=True,
                    model=embedder.describe().get("model", ""),
                )
                for event in reason(user_query, chat=chat, embedder=embedder, store=store):
                    events.put(event)
                    if event["type"] == "error":
                        return
                similar = store.find_similar(
                    query_vector,
                    top_k=5,
                    model=embedder.describe().get("model", ""),
                    exclude_ids={query_id},
                )
                events.put({"type": "similar", "items": similar, "store_size": store.count()})
            except backends.BackendError as exc:
                events.put({"type": "error", "message": str(exc), "hint": exc.hint})
            except Exception as exc:  # noqa: BLE001
                events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}", "hint": ""})
            finally:
                events.put(sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            while True:
                try:
                    event = events.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    yield ": heartbeat\n\n"
                    continue
                if event is sentinel:
                    break
                yield _sse(event)
            yield _sse({"type": "done_stream"})
        finally:
            thread.join(timeout=1.0)
            store.close()

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    status = backends.health(backends.DEFAULT_BASE_URL)
    if not status["ok"]:
        print(f"\n  {status['problem']}\n  {status['hint']}\n")
    else:
        print(f"\n  Ollama at {status['base_url']} — models: {', '.join(status['models'])}\n")

    # Bound to localhost with the debugger off by default. The previous default of
    # debug=True on 0.0.0.0 exposed the Werkzeug console to the whole network.
    app.run(
        host=os.environ.get("LKG_HOST", "127.0.0.1"),
        port=int(os.environ.get("LKG_PORT", "5100")),
        debug=os.environ.get("LKG_DEBUG", "") == "1",
        threaded=True,
    )
