"""Local Knowledge Graph -- Flask entry point.

Thin by design: routing, server-sent-event framing, and process configuration. The
model backends live in ``backends.py``, the graph maths in ``graph.py``, the storage
in ``store.py``, and the reasoning loop in ``reasoning.py``.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time

from flask import Flask, Response, jsonify, render_template, request

from . import backends, rdf
from .jobs import Job, Registry
from .reasoning import explore, reason, settle
from .store import EmbeddingStore

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
    chat = backends.OllamaChat(_selected_chat())

    if os.environ.get("LKG_EMBED_BACKEND") == "hf":
        from .layers import HiddenStateEmbedding

        return chat, HiddenStateEmbedding(
            os.environ.get("LKG_HF_MODEL", "HuggingFaceTB/SmolLM2-135M"),
            layer=os.environ.get("LKG_HF_LAYER", "blocks.-1"),
            pooling=os.environ.get("LKG_HF_POOLING", "last"),
        )

    return chat, backends.OllamaEmbedding(_selected_embedding())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Why nothing is happening, in terms a user can act on."""
    return jsonify(backends.health(backends.DEFAULT_BASE_URL))


def _selected_chat() -> str:
    """The chat model in use: an explicit choice, else whatever is installed."""
    return app.config.get("CHAT_MODEL") or backends.pick_chat_model(
        backends.DEFAULT_BASE_URL, backends.DEFAULT_CHAT_MODEL
    )


def _selected_embedding() -> str:
    return app.config.get("EMBED_MODEL", backends.DEFAULT_EMBED_MODEL)


@app.route("/models", methods=["GET", "POST"])
def models():
    """List what Ollama has, and let the page choose among it.

    Being told to pull a model you do not need, while three usable ones sit
    installed, is the worst version of this app's first-run experience.
    """
    if request.method == "POST":
        wanted = request.json or {}
        installed = {m["name"] for m in backends.list_models(backends.DEFAULT_BASE_URL)}

        for key, config_key in (("chat", "CHAT_MODEL"), ("embedding", "EMBED_MODEL")):
            name = (wanted.get(key) or "").strip()
            if not name:
                continue
            # Only ever select something Ollama actually reports. This value is
            # sent straight to the model API, so it is not a free-text field.
            if name not in installed:
                return jsonify({"error": f"'{name}' is not installed"}), 400
            app.config[config_key] = name

    installed = backends.list_models(backends.DEFAULT_BASE_URL)
    return jsonify({
        "chat": [m["name"] for m in installed if not m["is_embedding"]],
        "embedding": [m["name"] for m in installed if m["is_embedding"]],
        "selected": {"chat": _selected_chat(), "embedding": _selected_embedding()},
        "suggested": backends.SUGGESTED,
        "ollama_url": backends.DEFAULT_BASE_URL,
    })


def _same_origin() -> bool:
    """Refuse cross-site requests to the state-changing routes.

    A page on the internet can POST to a service on your loopback address. Starting
    a multi-gigabyte download has to be something *this* page asked for.
    """
    site = request.headers.get("Sec-Fetch-Site")
    if site:
        return site in ("same-origin", "none")
    origin = request.headers.get("Origin") or request.headers.get("Referer") or ""
    return not origin or origin.startswith(request.host_url.rstrip("/"))


@app.route("/pull", methods=["POST"])
def pull():
    """Download a model, streaming Ollama's progress to the page."""
    if not _same_origin():
        return jsonify({"error": "cross-origin request refused"}), 403

    name = ((request.json or {}).get("model") or "").strip()
    # An allowlist, not free text: this route causes a multi-gigabyte download, and
    # the set of things a first-run user needs is small and known.
    if name not in {entry["name"] for entry in backends.SUGGESTED}:
        return jsonify({"error": f"'{name}' is not one of the offered models"}), 400

    def generate():
        try:
            for chunk in backends.pull_model(name, backends.DEFAULT_BASE_URL):
                yield _sse({"type": "pull", **chunk})
            yield _sse({"type": "pull_done", "model": name})
        except backends.BackendError as exc:
            yield _sse({"type": "error", "message": str(exc), "hint": exc.hint})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/favicon.ico")
def favicon():
    """A real answer, so the browser stops logging a 404 on every page load."""
    dot = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="7" fill="#2a78d6"/>'
        '<circle cx="10" cy="11" r="3.4" fill="#fff"/><circle cx="22" cy="9" r="2.6" fill="#fff"/>'
        '<circle cx="16" cy="23" r="3" fill="#fff"/>'
        '<path d="M10 11 L22 9 M10 11 L16 23 M22 9 L16 23" stroke="#fff" stroke-width="1.6" fill="none"/>'
        "</svg>"
    )
    return Response(dot, mimetype="image/svg+xml", headers={"Cache-Control": "max-age=86400"})


JOBS = Registry()


def _job_events(user_query: str, store_path: str, mode: str = "reason"):
    """The reasoning events for a headless run, with its own backends and store."""
    chat, embedder = app.config.get("BACKENDS_FACTORY", make_backends)()
    store = EmbeddingStore(store_path)
    decompose = int(os.environ.get("LKG_DECOMPOSE", "8"))
    try:
        if mode == "settle":
            # Explored, then explored again, and finished only when two
            # independent runs agree. No single call decides.
            yield from settle(user_query, chat=chat, embedder=embedder,
                              store=store, decompose=decompose)
        elif mode == "explore":
            yield from explore(user_query, chat=chat, embedder=embedder,
                               store=store, decompose=decompose)
        else:
            yield from reason(user_query, chat=chat, embedder=embedder, store=store,
                              decompose=decompose)
    finally:
        store.close()


@app.route("/jobs", methods=["GET", "POST"])
def jobs():
    """POST a question to start a headless run; GET lists what is known."""
    if request.method == "GET":
        return jsonify({"jobs": [j.status() for j in JOBS.all()]})

    payload = request.json if request.is_json else request.form
    user_query = str((payload or {}).get("query", "")).strip()
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # reason: one run. explore: each sub-question answered by its own run.
    # settle: explored twice, finished only on agreement.
    mode = str((payload or {}).get("mode", "reason")).strip().lower()
    if mode not in {"reason", "explore", "settle"}:
        return jsonify({"error": f"unknown mode {mode!r}",
                        "modes": ["reason", "explore", "settle"]}), 400

    job = Job(user_query)
    job.mode = mode
    JOBS.add(job)
    job.start(_job_events(user_query, app.config.get("DB_PATH", DB_PATH), mode))
    # 202: accepted and still running. The Location header is where to look.
    return jsonify(job.status()), 202, {"Location": f"/jobs/{job.id}"}


@app.route("/jobs/<job_id>", methods=["GET", "DELETE"])
def job_detail(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "no such job"}), 404
    if request.method == "DELETE":
        # Idempotent: killing a finished job is not an error, it is a no-op with
        # the same visible outcome, and a retrying client should not see a 409.
        job.kill()
        return jsonify(job.status())
    return jsonify(job.status())


@app.route("/jobs/<job_id>/rdf")
def job_rdf(job_id: str):
    """The finished graph as Turtle, or as N-Triples with ?format=nt."""
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "no such job"}), 404

    wants_nt = request.args.get("format", "turtle").lower() in {"nt", "ntriples", "n-triples"}
    render = rdf.to_ntriples if wants_nt else rdf.to_turtle
    body = render(
        job.id, job.question,
        graph=job.graph(), answer=job.answer,
        steps=job.of_type("step"),
        conversions=[e["result"] for e in job.of_type("convert")],
        sums=[f"{e['expression']} = {e['value']}" for e in job.of_type("calc")],
        # Only top-level findings: a sub-run's own findings describe its own
        # question, and flattening every level into one list loses which is which.
        findings=[e for e in job.of_type("finding") if not e.get("level")],
        votes=job.of_type("vote"),
        agreement=next(iter(job.of_type("agreed")), None),
    )
    kind = "application/n-triples" if wants_nt else "text/turtle"
    return Response(body, mimetype=f"{kind}; charset=utf-8")


@app.route("/jobs/<job_id>/stream")
def job_stream(job_id: str):
    """N-Triples as the run produces them, one complete triple per line.

    N-Triples rather than Turtle because a line is valid on its own: there are no
    prefixes to declare and no state to carry, so a consumer can parse what has
    arrived without waiting for the end. Turtle cannot be streamed this way, which
    is why the finished document is the one that gets prefixes.
    """
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "no such job"}), 404

    def generate():
        sent = 0
        seen: set[str] = set()
        while True:
            events = job.snapshot()
            new, sent = events[sent:], len(events)
            for line in _triples_for(job, new):
                if line not in seen:
                    seen.add(line)
                    yield line + "\n"
            if job.done.is_set() and sent >= len(job.snapshot()):
                return
            time.sleep(0.1)

    return Response(generate(), mimetype="application/n-triples; charset=utf-8")


def _triples_for(job, events: list[dict]):
    """Triples for a slice of new events, using the same vocabulary as the file."""
    steps = [e for e in events if e.get("type") == "step"]
    converts = [e["result"] for e in events if e.get("type") == "convert"]
    calcs = [f"{e['expression']} = {e['value']}" for e in events if e.get("type") == "calc"]
    graph = next((e["graph"] for e in reversed(events) if e.get("graph")), None)
    answer = next((e.get("content", "") for e in events if e.get("type") == "final"), "")
    return rdf.run_triples(job.id, job.question, graph=graph, answer=answer,
                           steps=steps, conversions=converts, sums=calcs)


@app.route("/query", methods=["GET", "POST"])
def query():
    if request.method == "POST":
        payload = request.json or {}
        user_query = payload.get("query", "")
        mode = str(payload.get("mode", "reason"))
    else:
        user_query = request.args.get("query", "")
        mode = request.args.get("mode", "reason")
    user_query = user_query.strip()
    mode = mode.strip().lower()
    if mode not in {"reason", "explore", "settle"}:
        mode = "reason"

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
                runner = {"explore": explore, "settle": settle}.get(mode, reason)
                for event in runner(user_query, chat=chat, embedder=embedder, store=store):
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


PORT_SEARCH_WINDOW = 20


def find_free_port(host: str, first: int, window: int = PORT_SEARCH_WINDOW) -> int:
    """The first free port at or above ``first``.

    Binds to test rather than asking whether the port is free: a "is it available"
    check followed by a separate bind has a race between the two, and the common
    case here -- you already have the app running -- is exactly when that race
    matters. Binding and catching OSError leaves no gap.
    """
    for candidate in range(first, first + window):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise OSError(
        f"No free port between {first} and {first + window - 1} on {host}. "
        f"Pass --port to choose another range."
    )


def banner(status: dict) -> str:
    """What the app found, and what to do about it.

    Written so the README does not have to explain the setup: the program that
    knows what is installed is the one best placed to say what is missing.
    """
    lines = ["", "  Local Knowledge Graph"]

    if not status["models"]:
        lines += [
            "",
            f"  Ollama did not answer at {status['base_url']}.",
            "",
            "  This app runs a language model on your own machine through Ollama.",
            "  1. Install it from https://ollama.com",
            "  2. Pull a model:   ollama pull llama3.2:3b",
            "  3. Start this again.",
            "",
            "  If Ollama runs elsewhere, set OLLAMA_URL to point at it.",
        ]
        return "\n".join(lines) + "\n"

    if not status["ok"]:
        lines += [
            "",
            f"  Ollama is running at {status['base_url']}, with: {', '.join(status['models'])}",
            f"  {status['problem']}",
            "",
            "  The page will offer to download one, or:",
            "    ollama pull llama3.2:3b",
        ]
        return "\n".join(lines) + "\n"

    chat = status.get("chat_model") or "?"
    embed = status.get("embedding_model") or f"{chat} (no embedding model installed)"
    lines += [
        "",
        f"  chat model    {chat}",
        f"  embeddings    {embed}",
        f"  ollama        {status['base_url']}",
        "",
        "  Both are changeable in the page. 'mpe-lkg doctor' reports this without starting.",
    ]
    return "\n".join(lines) + "\n"


def run(host: str | None = None, port: int | None = None, debug: bool | None = None) -> None:
    """Start the server, after saying whether the model backend is actually there."""
    print(banner(backends.health(backends.DEFAULT_BASE_URL)))

    host = host or os.environ.get("LKG_HOST", "127.0.0.1")
    wanted = port or int(os.environ.get("LKG_PORT", "5100"))
    chosen = find_free_port(host, wanted)
    if chosen != wanted:
        # Said out loud even though it is a convenience: silently landing on a
        # different port than the one you asked for is worse than an error, because
        # you go looking at the wrong URL.
        print(f"  {wanted} is in use — serving on http://{host}:{chosen} instead\n")

    # Bound to localhost with the debugger off by default. The previous default of
    # debug=True on 0.0.0.0 exposed the Werkzeug console to the whole network.
    app.run(
        host=host,
        port=chosen,
        debug=os.environ.get("LKG_DEBUG", "") == "1" if debug is None else debug,
        threaded=True,
    )


if __name__ == "__main__":
    run()
