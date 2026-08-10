"""Does the graph actually get drawn?

Everything else in this suite asserts on the data sent to the browser. This file
runs a real browser against a real server and reads the state back out of vis.js, so
a change that serialises perfectly but never renders is still caught.

Vendored copies of vis-network and marked are used, so these tests do not depend on
a CDN being reachable or on upstream not changing under us.
"""

import socket
import threading

import pytest
from conftest import normal_script, step
from werkzeug.serving import make_server

from backends import DeterministicEmbedding, ScriptedChat

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LiveServer:
    def __init__(self, script, tmp_path, *, repeat_last=False, dim=48, delay=0.0):
        import app as app_module

        chat = ScriptedChat(script, repeat_last=repeat_last, delay=delay)
        app_module.app.config["BACKENDS_FACTORY"] = lambda: (chat, DeterministicEmbedding(dim))
        app_module.app.config["DB_PATH"] = str(tmp_path / "render.db")
        self.port = free_port()
        self._server = make_server("127.0.0.1", self.port, app_module.app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._thread.join(timeout=5)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    yield page
    context.close()
    assert not errors, f"browser reported errors: {errors}"


def run_query(page, server, text="What is the capital of France?"):
    page.goto(server.url)
    page.fill("#query", text)
    page.click("#submit")
    page.wait_for_function("() => document.querySelector('#submit').disabled === false", timeout=30_000)


def graph_state(page) -> dict:
    return page.evaluate("() => ({nodes: nodes.get(), edges: edges.get()})")


class TestRendering:
    def test_nodes_and_edges_reach_visjs(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            state = graph_state(page)

        assert len(state["nodes"]) == 6
        assert state["edges"], "similarity edges must be drawn"
        known = {n["id"] for n in state["nodes"]}
        for edge in state["edges"]:
            assert edge["from"] in known and edge["to"] in known

    def test_edges_carry_a_similarity_label_and_length(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            state = graph_state(page)

        for edge in state["edges"]:
            assert edge["label"], "the similarity value is drawn on the edge"
            assert edge["length"] > 0, "the spring length must reach vis.js"

    def test_the_graph_paints_pixels(self, page, tmp_path):
        """A canvas that stays blank would pass every data-level assertion."""
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            page.wait_for_timeout(1200)  # let the physics settle
            painted = page.evaluate(
                """() => {
                    const c = document.querySelector('#graph canvas');
                    const ctx = c.getContext('2d');
                    const {data} = ctx.getImageData(0, 0, c.width, c.height);
                    let n = 0;
                    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) n++;
                    return n;
                }"""
            )
        assert painted > 1000, "the graph canvas is essentially empty"

    def test_steps_and_final_answer_are_shown(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            assert page.locator(".step").count() == 5
            assert page.locator(".final-answer").count() == 1
            assert "Paris" in page.locator(".final-answer").inner_text()

    def test_screenshot_artifact(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            page.wait_for_timeout(1200)
            out = tmp_path / "graph.png"
            page.screenshot(path=str(out), full_page=True)
        assert out.stat().st_size > 10_000


class TestErrorsAreVisible:
    def test_backend_failure_is_shown_on_the_page(self, page, tmp_path):
        """The reported symptom was a blank page with the error only in the console."""
        with LiveServer([], tmp_path) as server:  # a chat backend with nothing to say
            run_query(page, server)
            assert page.locator(".error-box").count() >= 1
            assert page.locator(".error-box").first.inner_text().strip()

    def test_submit_is_re_enabled_after_an_error(self, page, tmp_path):
        with LiveServer([], tmp_path) as server:
            run_query(page, server)
            assert page.locator("#submit").is_enabled()


class TestInteraction:
    def test_enter_key_submits(self, page, tmp_path):
        """There was no keydown handler at all; pressing Enter did nothing."""
        with LiveServer(normal_script(), tmp_path) as server:
            page.goto(server.url)
            page.fill("#query", "capital of France")
            page.press("#query", "Enter")
            page.wait_for_selector(".step", timeout=30_000)
            page.wait_for_function("() => !document.querySelector('#submit').disabled", timeout=30_000)
            assert page.locator(".step").count() >= 1

    def test_submit_is_disabled_while_streaming(self, page, tmp_path):
        # A backend that answers instantly makes the in-flight state unobservable,
        # so the fake is slowed down enough for the browser to be caught mid-run.
        with LiveServer(normal_script(), tmp_path, delay=0.4) as server:
            page.goto(server.url)
            page.fill("#query", "q")
            page.click("#submit")
            page.wait_for_function("() => document.querySelector('#submit').disabled === true", timeout=5_000)
            page.wait_for_function("() => !document.querySelector('#submit').disabled", timeout=30_000)

    def test_second_run_replaces_the_first(self, page, tmp_path):
        """The old page never closed the previous EventSource, so runs interleaved."""
        with LiveServer(normal_script() + normal_script(), tmp_path) as server:
            run_query(page, server, "first question")
            run_query(page, server, "second question")
            assert page.locator(".step").count() == 5
            assert page.evaluate("() => eventSource === null")

    def test_png_export_produces_a_file(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            page.wait_for_timeout(1000)
            with page.expect_download(timeout=15_000) as download:
                page.click("#download-img")
            saved = tmp_path / "export.png"
            download.value.save_as(str(saved))

        assert saved.stat().st_size > 5_000
        assert saved.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestTruncationNotice:
    def test_shortened_step_is_marked_in_the_ui(self, page, tmp_path):
        with LiveServer([step("Long", "y" * 900)], tmp_path, repeat_last=True) as server:
            run_query(page, server)
            assert page.locator(".notice").count() >= 1
