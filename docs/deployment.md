# Deployment

## Single-container path

The root `Dockerfile` builds the React app with Node 24, installs the Python project, copies the static build, bootstraps Qdrant when a generated seed exists, and serves UI plus API on one port.

```powershell
docker build -t nexus-voice-rag .
docker run --env-file .env -p 8000:8000 nexus-voice-rag
```

Docker was not available in the inspected development environment, so a successful local frontend build and backend test suite do not by themselves constitute a verified Docker build.

## Render Blueprint

`render.yaml` is the prepared production path. It uses the repository Dockerfile, the Singapore region, a free Render web service, CI-gated automatic deploys, and `/health`. The free service uses ephemeral storage under `/app/runtime`.

On an empty runtime, `scripts.bootstrap_index` loads 500 committed semantic chunks exported from the official Hindi validation split of AI4Bharat MSMARCO-XI, generates real multilingual vectors, and writes Qdrant locally. The seed avoids loading the large Parquet/datasets stack on the memory-limited free instance. The verified local evaluation remains the 500-record index described in the README.

Create a Blueprint from the public repository, add `ELEVENLABS_API_KEY` as a secret in the Render service environment, and wait for the initial vector bootstrap before judging readiness. Do not put the key in `render.yaml`.

## Required production state

- `ELEVENLABS_API_KEY` as a secret.
- A real index, supplied either by remote Qdrant or `rag/data/bootstrap/chunks.jsonl` generated from `scripts/export_bootstrap.py`.
- A deployment seed that rebuilds embedded Qdrant after free-instance restarts, or remote `QDRANT_URL` and `QDRANT_API_KEY` storage.
- `CORS_ORIGINS` set to the deployed origin.
- A production generator: local llama server sidecar/model volume or a compatible managed inference service. `extractive` is a labeled fail-safe, not a claim that an LLM is active.
- Health check against `/health`.

## Deployment gates

After deployment, verify in order:

1. `GET /health` returns vector points greater than zero.
2. The frontend loads over HTTPS.
3. A text query returns real retrieved chunks and a grounded answer/refusal.
4. Browser microphone permission succeeds.
5. `/voice-query` reports ElevenLabs as provider and a nonzero STT latency.
6. An unsupported query refuses.
7. No API key appears in frontend source, responses, or logs.
8. Set `LIVE_URL` and run `python -m scripts.final_check`.

Do not mark deployment complete from a build log. A successful live request and microphone flow are required.

## Current environment constraint

The free Render instance can host the single container but loses its local Qdrant files when it spins down or restarts, so the real MSMARCO-XI deployment seed is rebuilt at startup. ElevenLabs voice transcription remains unavailable until `ELEVENLABS_API_KEY` is configured as a Render secret.
