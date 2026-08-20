# Deployment

## Single-container path

The root `Dockerfile` builds the React app with Node 24, installs the Python project, copies the static build, bootstraps Qdrant when a generated seed exists, and serves UI plus API on one port.

```powershell
docker build -t nexus-voice-rag .
docker run --env-file .env -p 8000:8000 nexus-voice-rag
```

Docker was not available in the inspected development environment, so a successful local frontend build and backend test suite do not by themselves constitute a verified Docker build.

## Required production state

- `ELEVENLABS_API_KEY` as a secret.
- A real index, supplied either by remote Qdrant or `rag/data/bootstrap/chunks.jsonl` generated from `scripts/export_bootstrap.py`.
- Persistent storage for embedded Qdrant, or `QDRANT_URL` and `QDRANT_API_KEY`.
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

Firebase CLI authentication is available, but Firebase Hosting alone cannot host this stateful Python/Qdrant/LLM container. A full deployment therefore needs a container platform with secrets and persistent/remote vector storage. Creating a static-only site would not satisfy the end-to-end requirement and is deliberately not reported as success.
