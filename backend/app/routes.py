from __future__ import annotations

import asyncio
import json
import math
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, cast

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from backend.app.logging import get_logger
from backend.app.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    Percentiles,
    QueryRequest,
    RetrieveRequest,
    TranscriptResponse,
    VoiceQueryResponse,
)
from backend.app.services import ServiceContainer
from backend.app.stt import (
    TranscriptionError,
    TranscriptionRateLimited,
    TranscriptionUnavailable,
)
from rag.orchestration.harness import RAGResponse
from rag.retrieval.service import RetrievalResponse

router = APIRouter()
logger = get_logger()
_feedback_lock = asyncio.Lock()
_ALLOWED_AUDIO = {
    "audio/webm",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/mp4",
    "audio/x-m4a",
}


def _services(request: Request) -> ServiceContainer:
    return cast(ServiceContainer, request.app.state.services)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


async def _read_audio(upload: UploadFile, maximum: int) -> tuple[bytes, str]:
    content_type = (upload.content_type or "").split(";", 1)[0].lower()
    if content_type not in _ALLOWED_AUDIO:
        raise HTTPException(status_code=415, detail="Unsupported audio format")
    data = await upload.read(maximum + 1)
    if not data:
        raise HTTPException(status_code=422, detail="Audio file is empty")
    if len(data) > maximum:
        raise HTTPException(status_code=413, detail="Audio file exceeds the configured limit")
    if len(data) < 128:
        raise HTTPException(status_code=422, detail="Audio recording is too short")
    return data, content_type


async def _transcribe(upload: UploadFile, services: ServiceContainer) -> TranscriptResponse:
    data, content_type = await _read_audio(upload, services.settings.max_audio_bytes)
    try:
        transcript = await services.stt.transcribe(
            data, upload.filename or "recording.webm", content_type
        )
    except TranscriptionRateLimited as exc:
        raise HTTPException(
            status_code=429, detail="Voice transcription is rate limited. Try again."
        ) from exc
    except TranscriptionUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice transcription is unavailable. Check the server configuration.",
        ) from exc
    except TranscriptionError as exc:
        raise HTTPException(
            status_code=502, detail="Voice transcription failed. Please try again."
        ) from exc
    return TranscriptResponse(**asdict(transcript))


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    services = _services(request)
    try:
        vector = await run_in_threadpool(services.store.health)
    except Exception:
        logger.exception("health_check_failed", stage="vector_store")
        vector = {"status": "error", "points": 0}
    stt_ready = bool(services.settings.elevenlabs_api_key)
    generator_ready = (
        services.settings.generator_provider == "extractive"
        or services.settings.llm_model_path.exists()
    )
    ready = vector.get("status") == "ready" and stt_ready and generator_ready
    return HealthResponse(
        status="ready" if ready else "degraded",
        version="0.1.0",
        services={
            "vector_store": vector,
            "embeddings": {
                "provider": services.settings.embedding_provider,
                "model": services.settings.embedding_model,
            },
            "generation": {
                "provider": services.settings.generator_provider,
                "ready": generator_ready,
            },
            "speech_to_text": {
                "provider": "elevenlabs",
                "model": services.settings.elevenlabs_stt_model,
                "ready": stt_ready,
            },
        },
    )


@router.post("/query", response_model=RAGResponse)
async def query(payload: QueryRequest, request: Request) -> RAGResponse:
    services = _services(request)
    try:
        result = await run_in_threadpool(
            services.harness.run,
            payload.question,
            filters=payload.filters,
            request_id=_request_id(request),
            top_k=payload.top_k,
        )
    except Exception as exc:
        logger.exception("pipeline_failed", request_id=_request_id(request), stage="query")
        message = str(exc).lower()
        if "collection" in message or "qdrant" in message:
            raise HTTPException(
                status_code=503, detail="Knowledge retrieval is temporarily unavailable."
            ) from exc
        raise HTTPException(
            status_code=502, detail="Answer generation failed. Please try again."
        ) from exc
    logger.info(
        "pipeline_complete",
        request_id=result.request_id,
        latency_ms=result.latency.total_ms,
        retrieval_count=result.retrieval_count,
        grounded=result.grounded,
        refusal_reason=result.refusal_reason,
    )
    return result


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(payload: RetrieveRequest, request: Request) -> RetrievalResponse:
    try:
        return await run_in_threadpool(
            _services(request).retrieval.retrieve,
            payload.question,
            filters=payload.filters,
            top_k=payload.top_k,
        )
    except Exception as exc:
        logger.exception("retrieval_failed", request_id=_request_id(request), stage="retrieve")
        raise HTTPException(
            status_code=503, detail="Knowledge retrieval is temporarily unavailable."
        ) from exc


@router.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(request: Request, audio: Annotated[UploadFile, File()]) -> TranscriptResponse:
    return await _transcribe(audio, _services(request))


@router.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(request: Request, audio: Annotated[UploadFile, File()]) -> VoiceQueryResponse:
    started = perf_counter()
    services = _services(request)
    transcript = await _transcribe(audio, services)
    try:
        result = await run_in_threadpool(
            services.harness.run,
            transcript.text,
            request_id=_request_id(request),
        )
    except Exception as exc:
        logger.exception("voice_pipeline_failed", request_id=_request_id(request))
        raise HTTPException(
            status_code=502, detail="Answer generation failed. Please try again."
        ) from exc
    return VoiceQueryResponse(
        request_id=result.request_id,
        transcript=transcript,
        result=result,
        total_latency_ms=round((perf_counter() - started) * 1000, 3),
    )


def _percentiles(values: list[float]) -> Percentiles:
    ordered = sorted(values)

    def nearest_rank(percentile: float) -> float:
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return round(ordered[index], 3)

    return Percentiles(
        p50=nearest_rank(0.5),
        p70=nearest_rank(0.7),
        p100=round(ordered[-1], 3),
        sample_count=len(ordered),
    )


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(payload: BenchmarkRequest, request: Request) -> BenchmarkResponse:
    samples: dict[str, list[float]] = {
        "total": [],
        "embedding": [],
        "vector_search": [],
        "reranking": [],
        "generation": [],
        "grounding": [],
    }
    services = _services(request)
    for _ in range(payload.iterations):
        for question in payload.queries:
            try:
                result = await run_in_threadpool(services.harness.run, question)
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="Benchmark pipeline is unavailable."
                ) from exc
            latency = result.latency
            samples["total"].append(latency.total_ms)
            samples["embedding"].append(latency.embedding_ms)
            samples["vector_search"].append(latency.vector_search_ms)
            samples["reranking"].append(latency.reranking_ms)
            samples["generation"].append(latency.generation_ms)
            samples["grounding"].append(latency.grounding_ms)
    total = _percentiles(samples["total"])
    return BenchmarkResponse(
        total=total,
        embedding=_percentiles(samples["embedding"]),
        vector_search=_percentiles(samples["vector_search"]),
        reranking=_percentiles(samples["reranking"]),
        generation=_percentiles(samples["generation"]),
        grounding=_percentiles(samples["grounding"]),
        target_met=total.p100 < 200,
    )


@router.post("/feedback", response_model=FeedbackResponse, status_code=202)
async def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    record = {
        **payload.model_dump(),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    path = Path("reports/feedback.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    async with _feedback_lock:
        await asyncio.to_thread(_append_jsonl, path, record)
    return FeedbackResponse()


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
