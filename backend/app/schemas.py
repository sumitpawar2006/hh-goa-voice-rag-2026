from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.orchestration.harness import RAGResponse


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=25)
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(QueryRequest):
    pass


class TranscriptResponse(BaseModel):
    text: str
    language_code: str | None
    language_probability: float | None
    latency_ms: float
    provider: Literal["elevenlabs"] = "elevenlabs"
    model: str


class VoiceQueryResponse(BaseModel):
    request_id: str
    transcript: TranscriptResponse
    result: RAGResponse
    total_latency_ms: float


class BenchmarkRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=20)
    iterations: int = Field(default=3, ge=1, le=10)


class Percentiles(BaseModel):
    p50: float
    p70: float
    p100: float
    sample_count: int


class BenchmarkResponse(BaseModel):
    total: Percentiles
    embedding: Percentiles
    vector_search: Percentiles
    reranking: Percentiles
    generation: Percentiles
    grounding: Percentiles
    target_ms: float = 200
    target_met: bool
    scope: str = "text-query RAG pipeline; excludes STT"


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    accepted: bool = True


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded"]
    version: str
    services: dict[str, Any]
