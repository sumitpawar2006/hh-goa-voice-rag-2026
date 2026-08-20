from __future__ import annotations

import uuid
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.generation.models import AnswerGenerator
from rag.guardrails.validator import GuardrailValidator
from rag.retrieval.service import RetrievalService
from rag.vector_store.base import SearchResult


class StageTrace(BaseModel):
    stage: str
    status: Literal["passed", "failed", "skipped"]
    latency_ms: float
    detail: str | None = None


class SourceReference(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    source: str
    similarity_score: float
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    strategy: str
    position: int

    @classmethod
    def from_result(cls, result: SearchResult) -> SourceReference:
        return cls(**result.model_dump())


class LatencyBreakdown(BaseModel):
    validation_ms: float = 0
    safety_ms: float = 0
    embedding_ms: float = 0
    vector_search_ms: float = 0
    reranking_ms: float = 0
    generation_ms: float = 0
    grounding_ms: float = 0
    total_ms: float = 0


class RAGResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    grounded: bool
    confidence: float = Field(ge=0, le=1)
    refusal_reason: str | None = None
    generator: str
    retrieval_count: int = 0
    chunking_strategy: str | None = None
    latency: LatencyBreakdown
    trace: list[StageTrace]


class RAGHarness:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: AnswerGenerator,
        guardrails: GuardrailValidator,
        max_query_chars: int = 1000,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator
        self.guardrails = guardrails
        self.max_query_chars = max_query_chars

    @staticmethod
    def _ms(started: float) -> float:
        return round((perf_counter() - started) * 1000, 3)

    def _refusal(
        self,
        *,
        request_id: str,
        reason: str,
        answer: str,
        started: float,
        latency: LatencyBreakdown,
        trace: list[StageTrace],
        sources: list[SearchResult] | None = None,
    ) -> RAGResponse:
        latency.total_ms = self._ms(started)
        return RAGResponse(
            request_id=request_id,
            answer=answer,
            sources=[SourceReference.from_result(item) for item in (sources or [])],
            grounded=False,
            confidence=0,
            refusal_reason=reason,
            generator=self.generator.provider_name,
            retrieval_count=len(sources or []),
            chunking_strategy=(sources[0].strategy if sources else None),
            latency=latency,
            trace=trace,
        )

    def run(
        self,
        question: str,
        *,
        filters: dict[str, Any] | None = None,
        request_id: str | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        overall_started = perf_counter()
        request_id = request_id or str(uuid.uuid4())
        latency = LatencyBreakdown()
        trace: list[StageTrace] = []

        stage_started = perf_counter()
        normalized = self.retrieval.normalize_query(question)
        if not normalized or len(normalized) > self.max_query_chars:
            latency.validation_ms = self._ms(stage_started)
            trace.append(
                StageTrace(stage="validation", status="failed", latency_ms=latency.validation_ms)
            )
            return self._refusal(
                request_id=request_id,
                reason="invalid_query",
                answer="Please enter a non-empty question within the allowed length.",
                started=overall_started,
                latency=latency,
                trace=trace,
            )
        latency.validation_ms = self._ms(stage_started)
        trace.append(
            StageTrace(stage="validation", status="passed", latency_ms=latency.validation_ms)
        )

        stage_started = perf_counter()
        safety = self.guardrails.validate_input(normalized)
        latency.safety_ms = self._ms(stage_started)
        trace.append(
            StageTrace(
                stage="safety",
                status="passed" if safety.allowed else "failed",
                latency_ms=latency.safety_ms,
                detail=safety.reason,
            )
        )
        if not safety.allowed:
            return self._refusal(
                request_id=request_id,
                reason=safety.reason or "unsafe_request",
                answer=(
                    "I can't help with that request. Ask a safe question about the knowledge base."
                ),
                started=overall_started,
                latency=latency,
                trace=trace,
            )

        retrieval_started = perf_counter()
        retrieved = self.retrieval.retrieve(normalized, filters=filters, top_k=top_k)
        retrieval_wall_ms = self._ms(retrieval_started)
        latency.embedding_ms = retrieved.embedding_latency_ms
        latency.reranking_ms = retrieved.reranking_latency_ms
        latency.vector_search_ms = round(
            max(0.0, retrieval_wall_ms - latency.embedding_ms - latency.reranking_ms), 3
        )
        trace.extend(
            [
                StageTrace(stage="embedding", status="passed", latency_ms=latency.embedding_ms),
                StageTrace(
                    stage="vector_search",
                    status="passed",
                    latency_ms=latency.vector_search_ms,
                    detail=f"{retrieved.candidate_count} candidates",
                ),
                StageTrace(stage="reranking", status="passed", latency_ms=latency.reranking_ms),
            ]
        )

        relevance = self.guardrails.validate_context(retrieved.results)
        trace.append(
            StageTrace(
                stage="context_validation",
                status="passed" if relevance.relevant else "failed",
                latency_ms=0,
                detail=relevance.reason or f"best_score={relevance.best_score:.3f}",
            )
        )
        if not relevance.relevant:
            return self._refusal(
                request_id=request_id,
                reason=relevance.reason or "no_context",
                answer="I couldn't find enough supporting information to answer that.",
                started=overall_started,
                latency=latency,
                trace=trace,
                sources=retrieved.results,
            )

        stage_started = perf_counter()
        generated = self.generator.generate(normalized, retrieved.results)
        latency.generation_ms = self._ms(stage_started)
        trace.append(
            StageTrace(stage="generation", status="passed", latency_ms=latency.generation_ms)
        )
        if generated.answer == "INSUFFICIENT_CONTEXT":
            return self._refusal(
                request_id=request_id,
                reason="model_insufficient_context",
                answer="I couldn't find enough supporting information to answer that.",
                started=overall_started,
                latency=latency,
                trace=trace,
                sources=retrieved.results,
            )

        stage_started = perf_counter()
        grounding = self.guardrails.validate_output(generated, retrieved.results)
        latency.grounding_ms = self._ms(stage_started)
        trace.append(
            StageTrace(
                stage="grounding",
                status="passed" if grounding.grounded else "failed",
                latency_ms=latency.grounding_ms,
                detail=grounding.reason or f"support={grounding.support_score:.3f}",
            )
        )
        if not grounding.grounded:
            return self._refusal(
                request_id=request_id,
                reason="hallucination_detected",
                answer="The generated answer could not be verified against the retrieved context.",
                started=overall_started,
                latency=latency,
                trace=trace,
                sources=retrieved.results,
            )

        cited = [item for item in retrieved.results if item.chunk_id in generated.source_chunk_ids]
        score_confidence = max(0.0, min(1.0, relevance.best_score))
        confidence = min(generated.confidence, grounding.support_score, score_confidence)
        latency.total_ms = self._ms(overall_started)
        return RAGResponse(
            request_id=request_id,
            answer=generated.answer,
            sources=[SourceReference.from_result(item) for item in cited],
            grounded=True,
            confidence=round(confidence, 3),
            refusal_reason=None,
            generator=self.generator.provider_name,
            retrieval_count=len(retrieved.results),
            chunking_strategy=retrieved.results[0].strategy if retrieved.results else None,
            latency=latency,
            trace=trace,
        )
