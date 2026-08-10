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
        # (dim, model) -> (matrix, metadata). Dropped on write.
        self._cache: dict[tuple[int, str], tuple[np.ndarray, list]] = {}

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
        self._cache.clear()

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def add(self, text: str, embedding: np.ndarray, *, is_question: bool = False, model: str = "") -> int:
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        cursor = self.conn.execute(
            "INSERT INTO embeddings (text, embedding, is_question, dim, model) VALUES (?, ?, ?, ?, ?)",
            (text, sqlite3.Binary(vector.tobytes()), int(is_question), int(vector.size), model),
        )
        self.conn.commit()
        self._cache.pop((int(vector.size), model), None)
        return int(cursor.lastrowid)

    def _matrix(self, dim: int, model: str) -> tuple[np.ndarray, list]:
        """Rows of the given shape as one contiguous matrix, cached in memory.

        Reading and unpacking every blob out of SQLite on each query is what makes a
        growing store slow -- not the arithmetic. The matrix product over a hundred
        thousand vectors takes about three milliseconds; decoding them from the
        database each time does not.
        """
        key = (dim, model)
        if key in self._cache:
            return self._cache[key]

        params: list = [dim]
        sql = "SELECT id, text, embedding, is_question FROM embeddings WHERE dim = ?"
        if model:
            sql += " AND model = ?"
            params.append(model)
        rows = list(self.conn.execute(sql, params))

        if rows:
            matrix = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        else:
            matrix = np.zeros((0, dim), dtype=np.float32)

        meta = [(int(r[0]), r[1], bool(r[3])) for r in rows]
        self._cache[key] = (matrix, meta)
        return matrix, meta

    def find_similar(
        self,
        query: np.ndarray,
        *,
        top_k: int = 5,
        model: str = "",
        exclude_ids: set[int] | None = None,
    ) -> list[dict]:
        """Exact cosine nearest neighbours, restricted to compatible vectors.

        Exact, not approximate, and deliberately so. Measured on this machine at 768
        dimensions (``scripts/bench_search.py``), one query costs 0.017 ms over a
        thousand vectors and 3.3 ms over a hundred thousand -- against an LLM call
        that takes seconds. An approximate index answers in 0.03 ms flat, which buys
        nothing here, and the one this project used to depend on cannot be appended
        to: it has to be rebuilt from scratch on every insert, which costs 3.5 ms at
        a hundred vectors and four seconds at a hundred thousand.
        """
        vector = np.asarray(query, dtype=np.float32).ravel()
        query_norm = float(np.linalg.norm(vector))
        if query_norm == 0.0:
            return []
        vector = vector / query_norm
        exclude_ids = exclude_ids or set()

        matrix, meta = self._matrix(int(vector.size), model)
        if not len(matrix):
            return []

        scores = matrix @ vector
        if exclude_ids:
            keep = np.array([row_id not in exclude_ids for row_id, _, _ in meta])
            scores = np.where(keep, scores, -np.inf)
            available = int(keep.sum())
        else:
            available = len(scores)
        if available == 0:
            return []

        k = min(top_k, available)
        # argpartition is linear; a full sort would be O(n log n) for no reason.
        candidates = np.argpartition(-scores, k - 1)[:k]
        order = candidates[np.argsort(-scores[candidates])]

        return [
            {
                "id": meta[i][0],
                "text": meta[i][1],
                "similarity": float(scores[i]),
                "is_question": meta[i][2],
            }
            for i in order
        ]
