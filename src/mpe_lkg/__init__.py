"""Local Knowledge Graph — a local LLM reasons step by step, and the steps become a graph.

    pip install mpe-lkg
    mpe-lkg

Or from Python:

    from mpe_lkg import create_app
    create_app().run(port=5100)
"""

from __future__ import annotations

__version__ = "0.5.0"

__all__ = ["__version__", "create_app", "main", "health"]


def create_app():
    """The Flask application. Imported lazily so `import mpe_lkg` stays cheap."""
    from .app import app

    return app


def main(argv: list[str] | None = None) -> int:
    from .cli import main as _main

    return _main(argv)


def health(base_url: str | None = None) -> dict:
    """Is Ollama reachable, and does it have what this needs?"""
    from . import backends

    return backends.health(base_url or backends.DEFAULT_BASE_URL)
