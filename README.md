# NEXUS — Voice-Enabled RAG for Hacker House Goa 2026

NEXUS is a multilingual, evidence-first voice search system for the Hacker House Goa 2026 Task 2 submission. A browser records a question, ElevenLabs Scribe v2 transcribes it, a multilingual embedding retrieves AI4Bharat MSMARCO-XI passages from Qdrant, an optional local Qwen LLM generates a structured answer, and grounding checks suppress unsupported output.

The interface exposes the transcript, retrieved chunks, stable document/chunk IDs, similarity and reranking scores, chunking strategy, model-harness trace, grounding state, and measured stage latency. It does not seed answers or fabricate metrics.

## Status

- Backend, frontend, four chunkers, ingestion, vector search, reranking, guardrails, STT integration, evaluation harness, and benchmark tooling are implemented.
- Unit/integration coverage includes ingestion, all chunkers, embeddings, Qdrant, retrieval, generation, grounding, safety, API validation, audio errors, STT failures/timeouts, vector failure, generator failure, and malformed output.
- Live ElevenLabs transcription requires `ELEVENLABS_API_KEY`.
- The default `extractive` generator is a key-free grounded fail-safe. The competition LLM path is `llama_server`, backed by local Qwen2.5-0.5B-Instruct through llama.cpp.
- Read `reports/final_readiness.json` after running the final checker; it is the authoritative readiness result.

## Architecture

Offline:

```text
AI4Bharat MSMARCO-XI (streamed Parquet)
  → schema validation / Unicode normalization / deduplication
  → configured chunking strategy
  → cached multilingual FastEmbed vectors
  → persistent local or remote Qdrant collection
```

Online:

```text
Browser microphone
  → ElevenLabs Scribe v2
  → input validation + safety guard
  → multilingual query embedding
  → Qdrant candidate search
  → optional lexical or cross-encoder reranking
  → context relevance gate
  → structured generator (local Qwen or extractive fail-safe)
  → citation validation + grounding check
  → answer, sources, trace, and real latency
```

See [docs/architecture.md](docs/architecture.md) for component ownership and failure paths.

## Dataset

[AI4Bharat MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) contains translated MS MARCO questions, answers, and passages for 14 Indic-language configurations. The repository reports 11,451,314 total rows and 55.6 GB. The Hindi validation file is about 462 MB and contains 97,941 records.

Each record is validated against its real fields: `source_lang`, `target_lang`, `meta`, `query`, `Answer`, `query_id`, `query_type`, `passages`, `Eng_Query`, and `Eng_Answer`. Passage arrays retain `is_selected`, English text, and translated text. Stable IDs are SHA-256-derived from provenance and normalized content.

The loader streams a chosen language and split, accepts a reproducible sample limit, skips malformed/empty records, counts duplicates and missing values, and writes evaluation queries separately from documents.

## Chunking

Set `CHUNKING_STRATEGY` to one of:

- `fixed`: deterministic non-overlapping word windows.
- `overlap`: sliding windows with configurable continuity.
- `semantic`: multilingual sentence-boundary grouping with bounded continuity.
- `metadata`: paragraph/record-boundary preservation with language, query type, and selected-passage facets.

Every chunk includes `document_id`, `chunk_id`, `text`, `source`, `metadata`, `strategy`, `position`, and token count. `scripts/chunk_experiment.py` measures chunk count, size distribution, vector storage, recall@K, MRR, failure rate, and retrieval latency for all strategies. See [docs/chunking.md](docs/chunking.md).

## Embeddings and vector database

Production vectors use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` through FastEmbed/ONNX. Embeddings are generated in batches and cached in SQLite by model and normalized text hash, so unchanged chunks are not recomputed.

Qdrant provides persistent cosine search, upsert, top-K, metadata filters, deletion, reindexing, and health checks. It runs in embedded local mode by default and supports a remote Qdrant URL/key without changing retrieval code.

The deterministic hash embedder exists only for tests and diagnostics; it is never presented as the production model.

## Retrieval and reranking

The retriever normalizes the query, measures query-embedding time, requests `CANDIDATE_K` Qdrant candidates, applies the configured reranker, and returns `TOP_K` results with full provenance.

The default reranker fuses cosine score with multilingual Unicode word/character overlap. An optional FastEmbed cross-encoder is available. `scripts/evaluate.py` compares vector-only and vector-plus-reranking using held-out `is_selected` passage judgments and recommends reranking only when measured MRR improves.

## Generation and grounding

All generators return:

```json
{
  "answer": "...",
  "source_chunk_ids": ["chk_..."],
  "confidence": 0.0
}
```

`llama_server` uses Qwen2.5-0.5B-Instruct GGUF via llama.cpp's local OpenAI-compatible server. The prompt treats retrieved text as untrusted data, restricts citations to supplied IDs, and requests `INSUFFICIENT_CONTEXT` when support is absent. The validator rejects malformed JSON, unknown citations, unsupported claims, weak/no context, unsafe requests, and prompt-injection attempts.

`extractive` copies the most relevant source sentences and is the reliable key-free fallback when the local LLM is not running. It remains explicitly labeled in API/UI output.

## Model harness

The harness is not a single prompt call. It executes:

```text
validation → safety → embedding → vector search → reranking
→ context validation → generation → structured validation → grounding
```

Each request receives a UUID, structured trace, stage statuses, measured latency, safe refusal reason, and source list. External failures map to safe user messages while structured logs retain stage and request context without secrets.

## Speech-to-text

NEXUS uses exactly one STT provider: [ElevenLabs Speech to Text](https://elevenlabs.io/docs/api-reference/speech-to-text/convert). The backend submits recorded WebM/OGG/WAV/MP3/M4A data to `/v1/speech-to-text` with `model_id=scribe_v2`. It enforces MIME and size limits and handles empty/short audio, unsupported formats, missing/rejected credentials, timeouts, rate limits, provider errors, and empty transcripts.

The API key never reaches the browser.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Vector, embedding, generation, and STT readiness |
| POST | `/transcribe` | Audio to ElevenLabs transcript |
| POST | `/query` | Text question through the full RAG harness |
| POST | `/voice-query` | Audio → STT → RAG in one request |
| POST | `/retrieve` | Retrieval-only transparency/debug endpoint |
| POST | `/benchmark` | P50/P70/P100 text-RAG measurements |
| POST | `/feedback` | Store request-linked feedback |

OpenAPI is served at `/docs`. See [docs/api.md](docs/api.md).

## Local setup

Requirements: Python 3.11–3.13 and Node.js 24+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
cd frontend
npm install
cd ..
```

Add `ELEVENLABS_API_KEY` to the untracked `.env` for live voice.

Inspect and build a real sampled index:

```powershell
.\.venv\Scripts\python.exe -m scripts.inspect_dataset --language hi --limit 100
.\.venv\Scripts\python.exe -m scripts.ingest --language hi --limit 500 --strategy semantic --recreate
```

Run the backend and frontend:

```powershell
.\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --port 8000
cd frontend
npm run dev
```

## Local LLM

```powershell
py -3.11 -m venv .venv-llm
.\.venv-llm\Scripts\python.exe -m pip install "llama-cpp-python[server]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.\.venv\Scripts\python.exe -m scripts.download_model
$env:MODEL = "rag/data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
.\.venv-llm\Scripts\python.exe -m llama_cpp.server --model $env:MODEL --host 127.0.0.1 --port 8080 --chat_format chatml
```

Set `GENERATOR_PROVIDER=llama_server`. The extractive fallback remains available for constrained deployments.

## Evaluation and latency

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --limit 100
.\.venv\Scripts\python.exe -m scripts.chunk_experiment --query-limit 30
.\.venv\Scripts\python.exe -m scripts.benchmark --queries 10 --iterations 3
```

Reports are written under `reports/`. P50, P70, and P100 use nearest-rank percentiles over every request. The text RAG report excludes STT and labels that scope. External STT and local-LLM latency are reported separately; the project never claims sub-200 ms end-to-end performance unless measured P100 proves it. See [docs/evaluation.md](docs/evaluation.md).

### Verified results — 20 August 2026

- Real Hindi validation ingestion: 500 records, 4,978 deduplicated documents, 5,048 semantic chunks, and 5,048 persistent 384-dimensional Qdrant points.
- Retrieval evaluation, 100 cold isolated judged queries: lexical reranking improved Recall@5 from `0.525` to `0.590` and MRR from `0.3012` to `0.3492`; mean search-plus-rerank latency was `16.705 ms`.
- Cold-query extractive RAG benchmark, 30 distinct queries with the embedding model prewarmed: total P50 `165.411 ms`, P70 `191.633 ms`, and P100 `267.974 ms`.
- The measured target is met at P50/P70, but **not at P100**. External ElevenLabs STT is excluded because no credential was available for a real voice measurement.
- Optional local Qwen CPU benchmark, three requests: total P50 `12,961.944 ms`, P70/P100 `24,170.064 ms`. This path is functional but not the latency-default path.

The checked-in documentation records these values because generated reports and data artifacts are intentionally ignored by Git.

## Tests and final gate

```powershell
.\.venv\Scripts\ruff.exe check backend rag scripts tests
.\.venv\Scripts\python.exe -m pytest
cd frontend
npm run lint
npm test
npm run build
cd ..
.\.venv\Scripts\python.exe -m scripts.final_check
```

The final checker exits non-zero and says `NOT_READY` when credentials, measurements, GitHub, deployment, or live verification are absent.

## Security

- `.env`, model weights, raw/processed data, vector files, and caches are ignored.
- CORS is allowlisted; credentials are disabled by default.
- Query length, audio type/size, request timeout, and schema validation are enforced.
- Retrieved documents are untrusted content.
- API keys and private prompts are not returned or logged.
- Provider errors become non-sensitive client messages.

## Deployment

The multi-stage `Dockerfile` builds React and serves it with FastAPI as one container. `docker-compose.yml` persists embedded Qdrant. `render.yaml` defines a Singapore Docker web service with a persistent disk and streams a bounded real MSMARCO-XI sample into Qdrant on the first start, so dataset passages are not copied into this repository. Render prompts for the ElevenLabs secret during Blueprint creation. See [docs/deployment.md](docs/deployment.md).

## Submission assets

- [docs/demo-script.md](docs/demo-script.md): end-to-end judge demo.
- [docs/team-video.md](docs/team-video.md): 90-second process video shot list.
- [docs/social-checklist.md](docs/social-checklist.md): per-member Instagram/X verification checklist.

## Known limitations

- The source dataset is 55.6 GB; local development indexes a reproducible configured sample unless remote infrastructure is provided.
- ElevenLabs requires a valid key and network access.
- Qwen 0.5B is small and slower than the 200 ms full-pipeline goal on CPU; grounding/refusal checks compensate for quality risk, not speed.
- Embedded Qdrant is for a single instance. Multi-replica deployment requires remote Qdrant.
- Social posting and final video recording require human team accounts and appearances and are never marked complete automatically.

## License and attribution

MSMARCO-XI follows the original MS MARCO licensing terms; consult its dataset card before redistribution. Qwen and the FastEmbed model use their respective upstream licenses.
