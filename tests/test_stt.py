from __future__ import annotations

import httpx
import pytest

from backend.app.stt import (
    ElevenLabsSTT,
    TranscriptionError,
    TranscriptionRateLimited,
    TranscriptionUnavailable,
)


@pytest.mark.asyncio
async def test_elevenlabs_transcription_success() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["xi-api-key"] == "test-key"
        return httpx.Response(
            200,
            json={
                "text": "What is retrieval?",
                "language_code": "en",
                "language_probability": 0.99,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stt = ElevenLabsSTT("test-key", client=client)
    result = await stt.transcribe(b"audio" * 100, "test.webm", "audio/webm")
    assert result.text == "What is retrieval?"
    assert result.provider == "elevenlabs"
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(429, TranscriptionRateLimited), (401, TranscriptionUnavailable), (500, TranscriptionError)],
)
async def test_elevenlabs_maps_provider_failures(
    status_code: int, error_type: type[Exception]
) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status_code, json={"detail": "x"}))
    )
    stt = ElevenLabsSTT("test-key", client=client)
    with pytest.raises(error_type):
        await stt.transcribe(b"audio" * 100, "test.webm", "audio/webm")
    await client.aclose()


@pytest.mark.asyncio
async def test_elevenlabs_requires_credential() -> None:
    stt = ElevenLabsSTT(None)
    with pytest.raises(TranscriptionUnavailable):
        await stt.transcribe(b"audio", "test.webm", "audio/webm")
    await stt.close()


@pytest.mark.asyncio
async def test_elevenlabs_handles_timeout() -> None:
    def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    stt = ElevenLabsSTT("test-key", client=client)
    with pytest.raises(TranscriptionUnavailable):
        await stt.transcribe(b"audio" * 100, "test.webm", "audio/webm")
    await client.aclose()
