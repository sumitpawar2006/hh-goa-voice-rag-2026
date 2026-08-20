from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class TranscriptionError(RuntimeError):
    pass


class TranscriptionUnavailable(TranscriptionError):
    pass


class TranscriptionRateLimited(TranscriptionError):
    pass


@dataclass(slots=True)
class Transcript:
    text: str
    language_code: str | None
    language_probability: float | None
    latency_ms: float
    provider: str = "elevenlabs"
    model: str = "scribe_v2"


class ElevenLabsSTT:
    endpoint = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(
        self,
        api_key: str | None,
        model: str = "scribe_v2",
        timeout_seconds: float = 30,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self._owns_client = client is None

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=0.5),
        reraise=True,
    )
    async def transcribe(self, data: bytes, filename: str, content_type: str) -> Transcript:
        if not self.api_key:
            raise TranscriptionUnavailable("ELEVENLABS_API_KEY is not configured")
        started = perf_counter()
        try:
            response = await self.client.post(
                self.endpoint,
                headers={"xi-api-key": self.api_key},
                files={"file": (filename, data, content_type)},
                data={
                    "model_id": self.model,
                    "tag_audio_events": "false",
                    "diarize": "false",
                    "timestamps_granularity": "none",
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TranscriptionUnavailable("Voice transcription timed out") from exc
        if response.status_code == 429:
            raise TranscriptionRateLimited("Speech-to-text rate limit reached")
        if response.status_code in {401, 403}:
            raise TranscriptionUnavailable("Speech-to-text credentials were rejected")
        if response.status_code >= 400:
            raise TranscriptionError("Voice transcription failed")
        payload = response.json()
        text = str(payload.get("text", "")).strip()
        if not text:
            raise TranscriptionError("Speech-to-text returned an empty transcript")
        return Transcript(
            text=text,
            language_code=payload.get("language_code"),
            language_probability=payload.get("language_probability"),
            latency_ms=round((perf_counter() - started) * 1000, 3),
            model=self.model,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
