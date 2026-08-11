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

from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LiveServer:
    def __init__(self, script, tmp_path, *, repeat_last=False, dim=48, delay=0.0):
        import mpe_lkg.app as app_module

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

        # Six reasoning steps plus the answer. The answer is its own node because it
        # is written from the graph's strongest thread rather than lifted from the
        # last step -- when it *is* the last step's text, that node is relabelled
        # instead, which test_stream covers.
        assert len(state["nodes"]) == 7
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
            # state="attached", not visible: once the run finishes the steps fold
            # away behind the summary, so they exist without being on screen.
            page.wait_for_selector(".step", state="attached", timeout=30_000)
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


class TestProgressAndCollapse:
    """Fase D: progress while it runs, and a folded summary once it is done."""

    def test_progress_bar_advances_and_finishes(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path, delay=0.3) as server:
            page.goto(server.url)
            page.fill("#query", "q")
            page.click("#submit")
            page.wait_for_function(
                "() => document.querySelector('#progress').dataset.state === 'running'", timeout=10_000
            )
            page.wait_for_function(
                "() => document.querySelector('#progress').dataset.state === 'done'", timeout=30_000
            )
            width = page.evaluate("() => document.querySelector('#progress-bar span').style.width")
        assert width == "100%"

    def test_a_failed_run_does_not_report_success(self, page, tmp_path):
        with LiveServer([], tmp_path) as server:   # backend with nothing to say
            run_query(page, server)
            assert page.evaluate("() => document.querySelector('#progress').dataset.state") == "error"

    def test_thinking_folds_into_one_line(self, page, tmp_path):
        """Question and answer stay in view; the reasoning collapses behind a count."""
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)

            toggle = page.locator(".thinking-toggle")
            assert toggle.count() == 1
            assert "5 thinking steps" in toggle.inner_text()
            assert page.locator("#steps").is_hidden()
            # The answer is never hidden -- that is the point of folding.
            assert page.locator(".final-answer").is_visible()

    def test_the_summary_expands_and_folds_again(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            page.click(".thinking-toggle")
            assert page.locator("#steps").is_visible()
            page.click(".thinking-toggle")
            assert page.locator("#steps").is_hidden()


class TestGraphToLog:
    def test_clicking_a_node_opens_that_step(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            page.wait_for_timeout(800)
            # Click through vis.js's own event, which is what a real click triggers.
            page.evaluate("() => network.emit('click', {nodes: ['Step3'], edges: []})")

            assert page.locator("#steps").is_visible(), "the log must open to show the step"
            assert page.locator("#step-3[data-focus='true']").count() == 1

    def test_the_close_button_folds_the_log_again(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            page.wait_for_timeout(800)
            page.evaluate("() => network.emit('click', {nodes: ['Step2'], edges: []})")
            page.click("#step-2 .step-close")

            assert page.locator("#steps").is_hidden()
            assert page.locator("#step-2[data-focus='true']").count() == 0

    def test_hovering_a_node_highlights_the_path_that_reached_it(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            page.wait_for_timeout(800)
            page.evaluate("() => network.emit('hoverNode', {node: 'Step4'})")
            highlighted = page.evaluate(
                "() => nodes.get().filter(n => n.borderWidth === 4).map(n => n.id)"
            )
            page.evaluate("() => network.emit('blurNode', {node: 'Step4'})")
            cleared = page.evaluate("() => nodes.get().filter(n => n.borderWidth === 4).length")

        assert "Step4" in highlighted, "the hovered node is on its own path"
        assert len(highlighted) > 1, "the path back to the start should light up too"
        assert cleared == 0, "leaving the node must clear the highlight"

    def test_steps_carry_the_id_the_graph_looks_them_up_by(self, page, tmp_path):
        with LiveServer(normal_script(), tmp_path) as server:
            run_query(page, server)
            ids = page.evaluate("() => [...document.querySelectorAll('#steps .step')].map(e => e.id)")
        assert ids == [f"step-{i}" for i in range(1, len(ids) + 1)]

    def test_an_exactly_evaluated_sum_is_shown(self, page, tmp_path):
        """The one part of a run the model did not decide, so it is worth showing.

        A reader can check "20-13.5 = 6.5" at a glance in a way they cannot check a
        paragraph of reasoning.
        """
        pytest.importorskip("mpeqs", reason="the arithmetic gate is an optional extra")
        script = [
            step("Total", "Three items at 4.50 each.", calc="3*4.5"),
            step("Change", "Subtract from the note.", "final_answer", calc="20-13.5"),
            "Your change is 6.50.",
        ]
        with LiveServer(script, tmp_path) as server:
            run_query(page, server, "What is my change?")
            chips = page.locator("#sums span")
            assert chips.count() == 2
            shown = [chips.nth(i).inner_text() for i in range(2)]
            assert "3*4.5 = 13.5" in shown
            assert "20-13.5 = 6.5" in shown

    def test_sums_from_a_previous_run_are_cleared(self, page, tmp_path):
        # A stale "= 6.5" left over from the last question is worse than none.
        pytest.importorskip("mpeqs", reason="the arithmetic gate is an optional extra")
        script = [
            step("A", "x", "final_answer", calc="2+2"), "Four.",
            step("B", "y", "final_answer", calc="3+3"), "Six.",
        ]
        with LiveServer(script, tmp_path) as server:
            run_query(page, server, "What is two plus two?")
            assert [c.inner_text() for c in page.locator("#sums span").all()] == ["2+2 = 4"]
            run_query(page, server, "What is three plus three?")
            # Not ["2+2 = 4", "3+3 = 6"]: the first run's sum belongs to the first
            # question, and leaving it up would attribute it to this one.
            assert [c.inner_text() for c in page.locator("#sums span").all()] == ["3+3 = 6"]
