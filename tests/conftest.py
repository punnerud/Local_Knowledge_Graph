import json
import pathlib
import sys

import pytest

# Works whether or not the package is installed, and identically on Windows.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mpe_lkg.backends import DeterministicEmbedding, ScriptedChat  # noqa: E402


def step(title: str, content: str, next_action: str = "continue") -> str:
    return json.dumps({"title": title, "content": content, "next_action": next_action})


def normal_script(n_steps: int = 6) -> list[str]:
    """A well-behaved model: distinct steps, then a final answer."""
    script = [
        step(f"Title {i}", f"Reasoning about part {i} of the problem, considering alternatives.")
        for i in range(1, n_steps)
    ]
    script.append(step("Conclusion", "The capital of France is Paris.", "final_answer"))
    return script


@pytest.fixture
def embedder():
    return DeterministicEmbedding(dim=48)


@pytest.fixture
def flask_client(tmp_path, embedder):
    """A Flask test client wired to fakes, with a per-test database."""
    import mpe_lkg.app as app_module

    def make_client(script: list[str], *, repeat_last: bool = False, embed=None):
        chat = ScriptedChat(script, repeat_last=repeat_last)
        used_embedder = embed or embedder
        app_module.app.config["BACKENDS_FACTORY"] = lambda: (chat, used_embedder)
        app_module.app.config["DB_PATH"] = str(tmp_path / "test.db")
        app_module.app.config["TESTING"] = True
        return app_module.app.test_client(), chat

    return make_client


def read_events(response) -> list[dict]:
    """Parse an SSE response body into event dicts, ignoring heartbeat comments."""
    events = []
    for block in response.get_data(as_text=True).split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events
