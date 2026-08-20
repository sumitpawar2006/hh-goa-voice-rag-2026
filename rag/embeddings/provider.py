from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from rag.embeddings.cache import EmbeddingCache


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...


class FastEmbedProvider:
    def __init__(
        self,
        model_name: str,
        cache_path: Path = Path("rag/data/cache/embeddings.sqlite3"),
        batch_size: int = 32,
        threads: int | None = 2,
    ) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_path)
        self._model = TextEmbedding(model_name=model_name, threads=threads, lazy_load=True)
        model_info = next(
            (item for item in TextEmbedding.list_supported_models() if item["model"] == model_name),
            None,
        )
        if model_info is None:
            raise ValueError(f"FastEmbed does not support {model_name}")
        self.dimension = int(model_info["dim"])

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[np.ndarray | None] = [self.cache.get(self.model_name, text) for text in texts]
        missing_indices = [index for index, vector in enumerate(results) if vector is None]
        if missing_indices:
            for start in range(0, len(missing_indices), self.batch_size):
                batch_indices = missing_indices[start : start + self.batch_size]
                batch_texts = [texts[index] for index in batch_indices]
                generated = list(self._model.embed(batch_texts, batch_size=self.batch_size))
                cache_items: list[tuple[str, np.ndarray]] = []
                for index, vector in zip(batch_indices, generated, strict=True):
                    normalized = np.asarray(vector, dtype=np.float32)
                    results[index] = normalized
                    cache_items.append((texts[index], normalized))
                self.cache.set_many(self.model_name, cache_items)
        return [np.asarray(vector, dtype=np.float32).tolist() for vector in results]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def warmup(self) -> None:
        next(iter(self._model.embed(["multilingual retrieval readiness"], batch_size=1)))

    def close(self) -> None:
        self.cache.close()


class HashEmbeddingProvider:
    """Deterministic feature-hashing embedder used only for tests and offline diagnostics."""

    model_name = "hash-char-trigram-v1"

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _one(self, text: str) -> list[float]:
        normalized = f"  {text.casefold().strip()}  "
        vector = np.zeros(self.dimension, dtype=np.float32)
        for index in range(max(0, len(normalized) - 2)):
            token = normalized[index : index + 3]
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1 if digest[4] & 1 else -1
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)

    def warmup(self) -> None:
        self._one("multilingual retrieval readiness")

    def close(self) -> None:
        return None


def build_embedder(
    provider: str,
    model_name: str,
    batch_size: int = 32,
    cache_path: Path = Path("rag/data/cache/embeddings.sqlite3"),
    threads: int | None = 2,
) -> EmbeddingProvider:
    if provider == "hash":
        return HashEmbeddingProvider()
    if provider == "fastembed":
        return FastEmbedProvider(model_name, cache_path, batch_size, threads)
    raise ValueError(f"Unknown embedding provider: {provider}")
