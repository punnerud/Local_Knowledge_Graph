"""SQLite-backed embedding store with brute-force similarity search.

This replaces the Annoy index the project used to carry. An approximate
nearest-neighbour index earns its keep somewhere around a hundred thousand vectors;
this application holds roughly ten per query. Below that crossover a full scan in
numpy is both faster and exact, and dropping the dependency also removes the one
package in the requirements that does not ship a wheel for current Python -- which
is why ``pip install -r requirements.txt`` failed outright on a modern interpreter.

It also fixes a correctness bug that came with the index: Annoy's angular distance
is ``sqrt(2 * (1 - cos))`` over the range [0, 2], and the old code reported
``1 - distance`` as though it were a cosine similarity. That number could go
negative and was displayed to four decimal places as if it meant something.
"""

from __future__ import annotations

import sqlite3

import numpy as np

DEFAULT_PATH = "embeddings.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    is_question INTEGER NOT NULL DEFAULT 0,
    dim INTEGER NOT NULL,
    model TEXT NOT NULL DEFAULT ''
)
"""


class EmbeddingStore:
    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path
        # The rows are produced inside a streaming response, which Flask may run on
        # a different thread than the one that opened the connection.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add the columns that older databases from this project lack.

        Without ``dim`` and ``model`` there is no way to tell a 4096-dimensional
        vector from a 384-dimensional one, so switching embedding model silently
        corrupted every search against the old rows.
        """
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(embeddings)")}
        if "dim" not in existing:
            self.conn.execute("ALTER TABLE embeddings ADD COLUMN dim INTEGER NOT NULL DEFAULT 0")
        if "model" not in existing:
            self.conn.execute("ALTER TABLE embeddings ADD COLUMN model TEXT NOT NULL DEFAULT ''")

    def close(self) -> None:
        self.conn.close()

    def clear(self) -> None:
        self.conn.execute("DELETE FROM embeddings")
        self.conn.commit()

    def add(self, text: str, embedding: np.ndarray, *, is_question: bool = False, model: str = "") -> int:
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        cursor = self.conn.execute(
            "INSERT INTO embeddings (text, embedding, is_question, dim, model) VALUES (?, ?, ?, ?, ?)",
            (text, sqlite3.Binary(vector.tobytes()), int(is_question), int(vector.size), model),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def find_similar(
        self,
        query: np.ndarray,
        *,
        top_k: int = 5,
        model: str = "",
        exclude_ids: set[int] | None = None,
    ) -> list[dict]:
        """Exact cosine nearest neighbours, restricted to compatible vectors."""
        vector = np.asarray(query, dtype=np.float32).ravel()
        dim = int(vector.size)
        exclude_ids = exclude_ids or set()

        params: list = [dim]
        sql = "SELECT id, text, embedding, is_question FROM embeddings WHERE dim = ?"
        if model:
            sql += " AND model = ?"
            params.append(model)

        rows = list(self.conn.execute(sql, params))
        rows = [r for r in rows if r[0] not in exclude_ids]
        if not rows:
            return []

        matrix = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(matrix, axis=1)
        query_norm = float(np.linalg.norm(vector))
        if query_norm == 0.0:
            return []
        scores = np.divide(
            matrix @ vector,
            norms * query_norm,
            out=np.zeros(len(rows), dtype=np.float32),
            where=norms > 0,
        )

        order = np.argsort(-scores)[:top_k]
        return [
            {
                "id": int(rows[i][0]),
                "text": rows[i][1],
                "similarity": float(scores[i]),
                "is_question": bool(rows[i][3]),
            }
            for i in order
        ]
