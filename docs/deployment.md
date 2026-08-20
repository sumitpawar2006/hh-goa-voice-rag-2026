# Deployment

## Single-container path

The root `Dockerfile` builds the React app with Node 24, installs the Python project, copies the static build, bootstraps Qdrant when a generated seed exists, and serves UI plus API on one port.

```powershell
docker build -t nexus-voice-rag .
docker run --env-file .env -p 8000:8000 nexus-voice-rag
```

Docker was not available in the inspected development environment, so a successful local frontend build and backend test suite do not by themselves constitute a verified Docker build.

## Render Blueprint

`render.yaml` is the prepared production path. It uses the repository Dockerfile, the Singapore region, CI-gated automatic deploys, `/health`, and a persistent disk mounted at `/app/runtime`. Persistent disks require a paid Render web-service plan.

On an empty disk, `scripts.bootstrap_index` streams 50 official Hindi validation records directly from AI4Bharat MSMARCO-XI, applies semantic chunking, generates real multilingual vectors, and writes Qdrant under the mounted disk. Later restarts detect existing points and skip the operation. The bounded deployment sample keeps first-start resource use practical; the verified local evaluation remains the 500-record index described in the README.

Create a Blueprint from the public repository, supply `ELEVENLABS_API_KEY` when Render prompts for the `sync: false` value, and wait for the initial dataset/vector bootstrap before judging readiness. Do not put the key in `render.yaml`.

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

Firebase Hosting alone cannot host this stateful Python/Qdrant/LLM container. Render is therefore the selected deployment target, but no Render authentication or API key was available in the inspected environment. Creating a static-only Firebase site would not satisfy the end-to-end requirement and is deliberately not reported as success.
