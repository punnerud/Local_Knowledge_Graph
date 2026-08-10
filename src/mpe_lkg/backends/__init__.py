"""Model backends.

The rest of the application never talks to Ollama directly. It asks for an
``EmbeddingBackend`` and a ``ChatBackend`` and uses those, which is what makes the
reasoning loop and the graph testable without a model running anywhere.

Two rules are load-bearing here and are easy to lose in a refactor:

* An embedding backend reports its own dimension. Nothing downstream may assume a
  size. The original code hardcoded 4096 in one place and passed it as a parameter
  in another, so swapping the embedding model broke similarity search silently.
* A backend never swallows an error. A failure is raised as ``BackendError`` with a
  message meant for a human, because the symptom users reported was a blank page
  with nothing in the terminal.

Split three ways: what every backend shares, the real one, and the fakes the tests
run against. Everything is re-exported here, so ``from mpe_lkg import backends``
still reaches all of it.
"""

from ._shared import (
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    REQUEST_TIMEOUT,
    STEP_SCHEMA,
    BackendError,
    ChatBackend,
    EmbeddingBackend,
)
from .fakes import DeterministicEmbedding, ScriptedChat
from .ollama import (
    CHAT_PREFERENCE,
    SUGGESTED,
    OllamaChat,
    OllamaEmbedding,
    chat_models,
    embedding_models,
    health,
    list_models,
    pick_chat_model,
    pull_model,
)

__all__ = [
    "BackendError", "ChatBackend", "EmbeddingBackend", "STEP_SCHEMA",
    "DEFAULT_BASE_URL", "DEFAULT_CHAT_MODEL", "DEFAULT_EMBED_MODEL", "REQUEST_TIMEOUT",
    "OllamaChat", "OllamaEmbedding", "list_models", "chat_models", "embedding_models",
    "pick_chat_model", "pull_model", "health", "SUGGESTED", "CHAT_PREFERENCE",
    "ScriptedChat", "DeterministicEmbedding",
]
