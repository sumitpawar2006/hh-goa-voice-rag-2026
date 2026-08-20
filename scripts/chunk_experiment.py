from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from backend.app.config import Settings
from rag.chunking.models import Document
from rag.chunking.router import ChunkerRouter
from rag.embeddings.provider import build_embedder
from rag.evaluation.metrics import evaluate_rankings
from rag.reranking.service import NoOpReranker
from rag.retrieval.service import RetrievalService
from rag.vector_store.qdrant_store import QdrantVectorStore
from scripts.common import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare four chunking strategies with real retrieval."
    )
    parser.add_argument("--document-limit", type=int, default=1000)
    parser.add_argument("--query-limit", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("reports/chunking_experiment.json"))
    args = parser.parse_args()
    settings = Settings()
    documents = [
        Document.model_validate(item)
        for item in read_jsonl(Path("rag/data/processed/documents.jsonl"))
    ][: args.document_limit]
    document_ids = {document.document_id for document in documents}
    queries: list[dict[str, Any]] = []
    for item in read_jsonl(Path("rag/data/processed/evaluation_queries.jsonl")):
        relevant_ids = [
            document_id
            for document_id in item.get("relevant_document_ids", [])
            if document_id in document_ids
        ]
        if relevant_ids:
            queries.append({**item, "relevant_document_ids": relevant_ids})
        if len(queries) >= args.query_limit:
            break
    embedder = build_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_batch_size,
        threads=settings.embedding_threads,
    )
    router = ChunkerRouter(settings.chunk_size, settings.chunk_overlap)
    results: dict[str, dict[str, Any]] = {}

    for strategy in ("fixed", "overlap", "semantic", "metadata"):
        started = perf_counter()
        chunks, stats = router.chunk_documents(documents, strategy)
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        with tempfile.TemporaryDirectory(prefix=f"nexus-{strategy}-") as directory:
            store = QdrantVectorStore(f"experiment_{strategy}", Path(directory))
            store.ensure_collection(embedder.dimension, recreate=True)
            store.upsert(chunks, vectors)
            retrieval = RetrievalService(
                embedder, store, NoOpReranker(), settings.top_k, settings.candidate_k
            )
            rankings: list[list[str]] = []
            relevant: list[set[str]] = []
            latencies: list[float] = []
            for item in queries:
                query_started = perf_counter()
                response = retrieval.retrieve(str(item["query"]))
                latencies.append((perf_counter() - query_started) * 1000)
                rankings.append([result.document_id for result in response.results])
                relevant.append(set(item["relevant_document_ids"]))
            metrics = evaluate_rankings(rankings, relevant, latencies)
            store.close()
        results[strategy] = {
            "chunk_stats": stats.model_dump(),
            "retrieval": metrics.model_dump(),
            "storage_vector_values": len(chunks) * embedder.dimension,
            "experiment_elapsed_ms": round((perf_counter() - started) * 1000, 3),
        }
    recommended = max(
        results,
        key=lambda name: (
            float(results[name]["retrieval"]["mean_reciprocal_rank"]),
            float(results[name]["retrieval"]["recall_at_k"]),
            -int(results[name]["storage_vector_values"]),
            -float(results[name]["retrieval"]["mean_latency_ms"]),
        ),
    )
    report = {
        "document_count": len(documents),
        "query_count": len(queries),
        "embedding_model": embedder.model_name,
        "strategies": results,
        "recommended_strategy": recommended,
        "selection_order": "MRR, recall@K, storage, retrieval latency",
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
