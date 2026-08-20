from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from backend.app.config import Settings
from rag.chunking.router import ChunkerRouter
from rag.embeddings.provider import build_embedder
from rag.ingestion.msmarco import LANGUAGE_FILE_PREFIX, MSMARCOIngestor
from rag.vector_store.qdrant_store import QdrantVectorStore
from scripts.common import write_json, write_jsonl


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Ingest and index AI4Bharat MSMARCO-XI.")
    parser.add_argument(
        "--language", choices=sorted(LANGUAGE_FILE_PREFIX), default=settings.dataset_language
    )
    parser.add_argument("--split", choices=["train", "validation"], default=settings.dataset_split)
    parser.add_argument("--limit", type=int, default=settings.dataset_sample_size)
    parser.add_argument(
        "--strategy",
        choices=["fixed", "overlap", "semantic", "metadata"],
        default=settings.chunking_strategy,
    )
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    print(f"Streaming {args.limit} {args.language}/{args.split} records from MSMARCO-XI...")
    ingestor = MSMARCOIngestor(settings.dataset_name, args.language, args.split)
    documents, queries, ingestion_stats = ingestor.ingest(limit=args.limit)
    print(f"Validated {ingestion_stats.records_valid} records -> {len(documents)} documents")

    router = ChunkerRouter(settings.chunk_size, settings.chunk_overlap)
    chunks, chunk_stats = router.chunk_documents(documents, args.strategy)
    print(f"Chunked with {args.strategy}: {len(chunks)} chunks")

    processed = Path("rag/data/processed")
    write_jsonl(processed / "documents.jsonl", documents)
    write_jsonl(processed / "evaluation_queries.jsonl", queries)
    write_jsonl(processed / "chunks.jsonl", chunks)

    embedder = build_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_batch_size,
        threads=settings.embedding_threads,
    )
    print(f"Embedding {len(chunks)} chunks with {embedder.model_name}...")
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    store = QdrantVectorStore(
        settings.qdrant_collection,
        settings.qdrant_path,
        settings.qdrant_url,
        settings.qdrant_api_key,
    )
    store.ensure_collection(embedder.dimension, recreate=args.recreate)
    indexed = store.upsert(chunks, vectors)
    health = store.health()
    store.close()
    report = {
        "dataset": settings.dataset_name,
        "language": args.language,
        "split": args.split,
        "sample_limit": args.limit,
        "ingestion": ingestion_stats.model_dump(),
        "chunking": chunk_stats.model_dump(),
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "indexed_chunks": indexed,
        "vector_store": health,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    write_json(Path("reports/ingestion.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
