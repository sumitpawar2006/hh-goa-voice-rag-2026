from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from rag.embeddings.provider import EmbeddingProvider
from rag.reranking.service import Reranker
from rag.vector_store.base import SearchResult, VectorStore

_SPACE = re.compile(r"\s+")


class RetrievalResponse(BaseModel):
    query: str
    results: list[SearchResult]
    candidate_count: int
    retrieval_latency_ms: float
    embedding_latency_ms: float
    reranking_latency_ms: float
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        reranker: Reranker,
        top_k: int = 5,
        candidate_k: int = 12,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = max(candidate_k, top_k)

    @staticmethod
    def normalize_query(query: str) -> str:
        return _SPACE.sub(" ", query.strip())

    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        normalized = self.normalize_query(query)
        if not normalized:
            raise ValueError("Query cannot be empty")
        requested_k = top_k or self.top_k
        started = perf_counter()
        embed_started = perf_counter()
        vector = self.embedder.embed_query(normalized)
        embedding_ms = (perf_counter() - embed_started) * 1000
        candidates = self.store.search(vector, self.candidate_k, filters)
        rerank_started = perf_counter()
        results = self.reranker.rerank(normalized, candidates, requested_k)
        rerank_ms = (perf_counter() - rerank_started) * 1000
        return RetrievalResponse(
            query=normalized,
            results=results,
            candidate_count=len(candidates),
            retrieval_latency_ms=round((perf_counter() - started) * 1000, 3),
            embedding_latency_ms=round(embedding_ms, 3),
            reranking_latency_ms=round(rerank_ms, 3),
            filters=filters or {},
        )
