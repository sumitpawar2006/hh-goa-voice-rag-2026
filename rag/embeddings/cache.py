from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from threading import RLock

import numpy as np


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = RLock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()

    def get(self, model: str, text: str) -> np.ndarray | None:
        cache_key = self.key(model, text)
        with self._lock:
            row = self._connection.execute(
                "SELECT dimensions, vector FROM embeddings WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        dimensions, blob = row
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        return vector if len(vector) == dimensions else None

    def set(self, model: str, text: str, vector: np.ndarray) -> None:
        self.set_many(model, [(text, vector)])

    def set_many(self, model: str, items: Sequence[tuple[str, np.ndarray]]) -> None:
        rows = []
        for text, vector in items:
            normalized = np.asarray(vector, dtype=np.float32)
            rows.append((self.key(model, text), model, len(normalized), normalized.tobytes()))
        if not rows:
            return
        with self._lock, self._connection:
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings(cache_key, model, dimensions, vector)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(row[0]) if row else 0

    def delete_many(self, model: str, texts: Sequence[str]) -> int:
        keys = [(self.key(model, text),) for text in texts]
        if not keys:
            return 0
        with self._lock, self._connection:
            before = self._connection.total_changes
            self._connection.executemany("DELETE FROM embeddings WHERE cache_key = ?", keys)
            return self._connection.total_changes - before

    def close(self) -> None:
        with self._lock:
            self._connection.close()
