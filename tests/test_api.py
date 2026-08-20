from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services import ServiceContainer
from backend.app.stt import Transcript, TranscriptionUnavailable
from rag.generation.models import GeneratedAnswer
from rag.guardrails.validator import GuardrailValidator
from rag.orchestration.harness import RAGHarness


class FakeSTT:
    async def transcribe(self, data: bytes, filename: str, content_type: str) -> Transcript:
        return Transcript(
            text="What was the immediate impact of the Manhattan Project?",
            language_code="en",
            language_probability=0.99,
            latency_ms=12.5,
        )

    async def close(self) -> None:
        return None


class FailingSTT:
    async def transcribe(self, data: bytes, filename: str, content_type: str) -> Transcript:
        raise TranscriptionUnavailable("provider unavailable")

    async def close(self) -> None:
        return None


def build_client(indexed_pipeline: dict[str, object], tmp_path: Path, stt: Any) -> TestClient:
    settings = Settings(
        app_env="test",
        embedding_provider="hash",
        qdrant_path=tmp_path / "unused",
        min_relevance_score=-1,
        elevenlabs_api_key="configured",
    )
    services = ServiceContainer(settings)
    services.__dict__["store"] = indexed_pipeline["store"]
    services.__dict__["retrieval"] = indexed_pipeline["retrieval"]
    services.__dict__["harness"] = indexed_pipeline["harness"]
    services.__dict__["stt"] = stt
    return TestClient(create_app(settings, services))


def test_health_query_retrieve_feedback_and_voice(
    indexed_pipeline: dict[str, object], tmp_path: Path
) -> None:
    with build_client(indexed_pipeline, tmp_path, FakeSTT()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["services"]["vector_store"]["points"] > 0

        query = client.post(
            "/query", json={"question": "What was the impact of the Manhattan Project?"}
        )
        assert query.status_code == 200
        assert query.json()["grounded"] is True

        retrieve = client.post("/retrieve", json={"question": "Manhattan Project impact"})
        assert retrieve.status_code == 200
        assert retrieve.json()["results"]

        feedback = client.post(
            "/feedback", json={"request_id": query.json()["request_id"], "rating": 1}
        )
        assert feedback.status_code == 202

        voice = client.post(
            "/voice-query",
            files={"audio": ("voice.webm", b"0" * 512, "audio/webm")},
        )
        assert voice.status_code == 200
        assert voice.json()["transcript"]["provider"] == "elevenlabs"
        assert voice.json()["result"]["grounded"] is True


def test_api_rejects_empty_invalid_and_unsupported_audio(
    indexed_pipeline: dict[str, object], tmp_path: Path
) -> None:
    with build_client(indexed_pipeline, tmp_path, FakeSTT()) as client:
        assert client.post("/query", json={"question": ""}).status_code == 422
        empty = client.post("/transcribe", files={"audio": ("voice.webm", b"", "audio/webm")})
        invalid = client.post(
            "/transcribe", files={"audio": ("voice.txt", b"x" * 512, "text/plain")}
        )
        short = client.post("/transcribe", files={"audio": ("voice.webm", b"x" * 12, "audio/webm")})
        assert empty.status_code == 422
        assert invalid.status_code == 415
        assert short.status_code == 422


def test_api_recovers_from_stt_failure(indexed_pipeline: dict[str, object], tmp_path: Path) -> None:
    with build_client(indexed_pipeline, tmp_path, FailingSTT()) as client:
        response = client.post(
            "/transcribe", files={"audio": ("voice.webm", b"x" * 512, "audio/webm")}
        )
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()


def test_benchmark_reports_required_percentiles(
    indexed_pipeline: dict[str, object], tmp_path: Path
) -> None:
    with build_client(indexed_pipeline, tmp_path, FakeSTT()) as client:
        response = client.post(
            "/benchmark",
            json={"queries": ["Manhattan Project impact", "Where is Goa?"], "iterations": 2},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"]["sample_count"] == 4
        assert payload["total"]["p100"] >= payload["total"]["p70"] >= payload["total"]["p50"]


class FailingGenerator:
    provider_name = "failing-test-generator"

    def generate(self, question: str, contexts: list[object]) -> GeneratedAnswer:
        raise RuntimeError("model provider failed")


class MalformedGenerator:
    provider_name = "malformed-test-generator"

    def generate(self, question: str, contexts: list[object]) -> GeneratedAnswer:
        return {"unexpected": "shape"}  # type: ignore[return-value]


def test_api_recovers_from_llm_failure_and_malformed_output(
    indexed_pipeline: dict[str, object], tmp_path: Path
) -> None:
    client = build_client(indexed_pipeline, tmp_path, FakeSTT())
    with client:
        for generator in (FailingGenerator(), MalformedGenerator()):
            cast(FastAPI, client.app).state.services.__dict__["harness"] = RAGHarness(
                indexed_pipeline["retrieval"],  # type: ignore[arg-type]
                generator,  # type: ignore[arg-type]
                GuardrailValidator(-1),
            )
            response = client.post(
                "/query", json={"question": "What was the Manhattan Project impact?"}
            )
            assert response.status_code == 502
            assert "generation failed" in response.json()["detail"].lower()


def test_api_recovers_from_vector_database_failure(
    indexed_pipeline: dict[str, object], tmp_path: Path
) -> None:
    client = build_client(indexed_pipeline, tmp_path, FakeSTT())

    class BrokenRetrieval:
        def retrieve(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("qdrant connection failed")

    cast(FastAPI, client.app).state.services.__dict__["retrieval"] = BrokenRetrieval()
    with client:
        response = client.post("/retrieve", json={"question": "test query"})
        assert response.status_code == 503
