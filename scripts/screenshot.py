#!/usr/bin/env python3
"""Drive the real app in a real browser and save a screenshot.

Used to regenerate example.png, and as a manual end-to-end check against a live
model rather than the fakes the test suite uses.

    .venv/bin/python scripts/screenshot.py "What is the capital of France?" out.png
"""

from __future__ import annotations

import pathlib
import socket
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

import app as app_module  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "Can you give the 5 biggest cities in population size in order?"
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "example.png")

    app_module.app.config["DB_PATH"] = str(ROOT / "embeddings.db")
    port = free_port()
    server = make_server("127.0.0.1", port, app_module.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_context(viewport={"width": 1600, "height": 1000}).new_page()
            page.goto(f"http://127.0.0.1:{port}")
            page.fill("#query", question)
            page.click("#submit")
            print("waiting for the model ...", flush=True)
            page.wait_for_function(
                "() => document.querySelector('#submit').disabled === false", timeout=600_000
            )
            page.wait_for_timeout(2500)  # let the graph physics settle

            errors = page.locator(".error-box")
            if errors.count():
                print(f"error shown in page: {errors.first.inner_text()}")
            print(f"status: {page.locator('#status').inner_text()}")
            print(f"steps: {page.locator('.step').count()}, final: {page.locator('.final-answer').count()}")
            print(f"nodes: {page.evaluate('() => nodes.get().length')}, "
                  f"edges: {page.evaluate('() => edges.get().length')}")

            page.screenshot(path=str(out), full_page=True)
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)

    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
