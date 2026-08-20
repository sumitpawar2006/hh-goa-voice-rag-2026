from __future__ import annotations

from pathlib import Path

from backend.app.config import Settings
from rag.chunking.models import Chunk
from rag.chunking.router import ChunkerRouter
from rag.embeddings.provider import build_embedder
from rag.ingestion.msmarco import MSMARCOIngestor
from rag.vector_store.qdrant_store import QdrantVectorStore
from scripts.common import read_jsonl


def _deployment_chunks(settings: Settings, seed_path: Path) -> list[Chunk]:
    if seed_path.exists():
        chunks = [Chunk.model_validate(item) for item in read_jsonl(seed_path)]
        print(f"Loaded {len(chunks)} chunks from the deployment seed.")
        return chunks
    if not settings.bootstrap_from_dataset:
        print("No deployment seed found and dataset bootstrap is disabled.")
        return []

    print(
        f"Streaming {settings.dataset_sample_size} real "
        f"{settings.dataset_language}/{settings.dataset_split} MSMARCO-XI records..."
    )
    ingestor = MSMARCOIngestor(
        settings.dataset_name,
        settings.dataset_language,
        settings.dataset_split,
    )
    documents, _, stats = ingestor.ingest(limit=settings.dataset_sample_size)
    if not documents:
        raise RuntimeError("Dataset bootstrap produced no valid documents")
    chunks, _ = ChunkerRouter(settings.chunk_size, settings.chunk_overlap).chunk_documents(
        documents, settings.chunking_strategy
    )
    print(
        f"Validated {stats.records_valid} records and generated {len(chunks)} "
        f"{settings.chunking_strategy} chunks."
    )
    return chunks


def bootstrap(settings: Settings, seed_path: Path = Path("rag/data/bootstrap/chunks.jsonl")) -> int:
    store = QdrantVectorStore(
        settings.qdrant_collection,
        settings.qdrant_path,
        settings.qdrant_url,
        settings.qdrant_api_key,
    )
    embedder = None
    try:
        existing = int(store.health().get("points", 0))
        if existing:
            print(f"Vector index already contains {existing} points; skipping bootstrap.")
            return existing
        chunks = _deployment_chunks(settings, seed_path)
        if not chunks:
            return 0
        embedder = build_embedder(
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_batch_size,
            threads=settings.embedding_threads,
        )
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        store.ensure_collection(embedder.dimension)
        indexed = store.upsert(chunks, vectors)
        print(f"Bootstrapped {indexed} chunks into {settings.qdrant_collection}.")
        return indexed
    finally:
        if embedder is not None:
            embedder.close()
        store.close()


def main() -> None:
    bootstrap(Settings())


if __name__ == "__main__":
    main()
