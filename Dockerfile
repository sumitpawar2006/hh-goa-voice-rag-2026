FROM node:24-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    HOME=/app/runtime \
    XDG_CACHE_HOME=/app/runtime/.cache \
    HF_HOME=/app/runtime/.cache/huggingface \
    QDRANT_PATH=/app/runtime/qdrant \
    GENERATOR_PROVIDER=extractive
WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend/ backend/
COPY rag/ rag/
COPY scripts/ scripts/
RUN pip install --no-cache-dir .
COPY --from=frontend-build /app/frontend/dist frontend/dist
RUN mkdir -p /app/runtime/qdrant /app/runtime/.cache/huggingface && chown -R 10001:10001 /app
USER 10001
EXPOSE 8000
CMD ["sh", "-c", "python -m scripts.bootstrap_index && uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
