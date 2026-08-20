# API reference

Interactive OpenAPI documentation is available at `/docs` while the backend is running.

## `GET /health`

Returns `ready` only when the vector index, configured generator, and ElevenLabs credential are ready. A missing credential produces `degraded`, not a false positive.

## `POST /query`

```json
{
  "question": "What was the immediate impact of the Manhattan Project?",
  "top_k": 5,
  "filters": { "language": "hi" }
}
```

Returns answer, sources, grounded flag, confidence, refusal reason, generator identity, retrieval count, chunk strategy, latency breakdown, request ID, and stage trace.

## `POST /retrieve`

Accepts the query schema and returns candidate count, result text/provenance/scores, filters, and embedding/retrieval/reranking times. It never generates an answer.

## `POST /transcribe`

Multipart field: `audio`. Accepted MIME types: WebM, WAV, MPEG/MP3, OGG, MP4, and M4A. The default maximum is 10 MiB. Successful responses include text, detected language/probability, provider, model, and real provider latency.

## `POST /voice-query`

Multipart field: `audio`. Executes transcription and the RAG harness with one request ID. Returns `transcript`, `result`, and `total_latency_ms`.

## `POST /benchmark`

```json
{
  "queries": ["question one", "question two"],
  "iterations": 3
}
```

Returns P50/P70/P100 by stage. Scope is text RAG and excludes STT.

## `POST /feedback`

```json
{
  "request_id": "uuid",
  "rating": 1,
  "comment": "optional"
}
```

Rating must be `1` or `-1`. Feedback is appended to an ignored JSONL runtime file.

## Common status codes

| Code | Meaning |
|---|---|
| 200/202 | Successful request/accepted feedback |
| 413 | Audio exceeds size limit |
| 415 | Unsupported audio type |
| 422 | Schema, empty, or too-short input |
| 429 | STT provider rate limit |
| 502 | Transcription/generation provider failed |
| 503 | STT credentials or retrieval service unavailable |
