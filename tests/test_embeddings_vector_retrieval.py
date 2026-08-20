from __future__ import annotations

from pathlib import Path

import numpy as np

from rag.embeddings.cache import EmbeddingCache
from rag.embeddings.provider import HashEmbeddingProvider


def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider(dimension=64)
    first = provider.embed_query("multilingual retrieval")
    second = provider.embed_query("multilingual retrieval")
    assert first == second
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1) < 1e-5


def test_embedding_cache_persists_batches(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite3")
    vectors = [
        ("first", np.asarray([1.0, 0.0], dtype=np.float32)),
        ("second", np.asarray([0.0, 1.0], dtype=np.float32)),
    ]
    cache.set_many("test-model", vectors)
    assert cache.count() == 2
    np.testing.assert_array_equal(cache.get("test-model", "second"), vectors[1][1])
    assert cache.delete_many("test-model", ["first"]) == 1
    assert cache.count() == 1
    cache.close()


def test_vector_store_search_and_metadata_filter(indexed_pipeline: dict[str, object]) -> None:
    embedder = indexed_pipeline["embedder"]
    store = indexed_pipeline["store"]
    vector = embedder.embed_query("Manhattan Project atomic bombs")  # type: ignore[attr-defined]
    results = store.search(vector, 3, {"query_type": "DESCRIPTION"})  # type: ignore[attr-defined]
    assert results
    assert results[0].document_id == "doc_manhattan"
    assert all(item.metadata["query_type"] == "DESCRIPTION" for item in results)


def test_retrieval_returns_latency_and_rerank_score(indexed_pipeline: dict[str, object]) -> None:
    response = indexed_pipeline["retrieval"].retrieve(  # type: ignore[attr-defined]
        "What was the immediate impact of the Manhattan Project?"
    )
    assert response.results[0].document_id == "doc_manhattan"
    assert response.results[0].rerank_score is not None
    assert response.embedding_latency_ms >= 0
    assert response.retrieval_latency_ms >= response.embedding_latency_ms
