from __future__ import annotations

from pathlib import Path

from backend.app.config import Settings
from rag.chunking.models import Chunk
from rag.embeddings.provider import build_embedder
from rag.vector_store.qdrant_store import QdrantVectorStore
from scripts.common import read_jsonl


def main() -> None:
    settings = Settings()
    seed_path = Path("rag/data/bootstrap/chunks.jsonl")
    if not seed_path.exists():
        print("No deployment bootstrap seed found; skipping index bootstrap.")
        return
    store = QdrantVectorStore(
        settings.qdrant_collection,
        settings.qdrant_path,
        settings.qdrant_url,
        settings.qdrant_api_key,
    )
    if store.health().get("points", 0):
        print("Vector index already contains data; skipping bootstrap.")
        store.close()
        return
    chunks = [Chunk.model_validate(item) for item in read_jsonl(seed_path)]
    embedder = build_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_batch_size,
    )
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    store.ensure_collection(embedder.dimension)
    store.upsert(chunks, vectors)
    print(f"Bootstrapped {len(chunks)} chunks into {settings.qdrant_collection}.")
    store.close()


if __name__ == "__main__":
    main()
