from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from backend.app.config import Settings
from rag.embeddings.provider import build_embedder
from rag.evaluation.metrics import evaluate_rankings
from rag.reranking.service import LexicalReranker
from rag.vector_store.qdrant_store import QdrantVectorStore
from scripts.common import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and reranking on held-out queries."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--cold-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    settings = Settings()
    query_path = Path("rag/data/processed/evaluation_queries.jsonl")
    if not query_path.exists():
        raise SystemExit("Run scripts/ingest.py before evaluation")
    queries = [item for item in read_jsonl(query_path) if item.get("relevant_document_ids")][
        : args.limit
    ]
    cache_path = Path("rag/data/cache/evaluation_embeddings.sqlite3")
    if args.cold_cache and cache_path.exists():
        cache_path.unlink()
    embedder = build_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_batch_size,
        cache_path=cache_path,
    )
    store = QdrantVectorStore(
        settings.qdrant_collection,
        settings.qdrant_path,
        settings.qdrant_url,
        settings.qdrant_api_key,
    )
    reranker = LexicalReranker()
    vector_rankings: list[list[str]] = []
    reranked_rankings: list[list[str]] = []
    relevant: list[set[str]] = []
    embedding_latencies: list[float] = []
    vector_latencies: list[float] = []
    reranked_latencies: list[float] = []
    for item in queries:
        query = str(item["query"])
        started = perf_counter()
        vector = embedder.embed_query(query)
        embedding_latencies.append((perf_counter() - started) * 1000)

        started = perf_counter()
        candidates = store.search(vector, settings.candidate_k)
        search_ms = (perf_counter() - started) * 1000
        vector_results = candidates[: settings.top_k]

        started = perf_counter()
        reranked_results = reranker.rerank(query, candidates, settings.top_k)
        rerank_ms = (perf_counter() - started) * 1000
        vector_latencies.append(search_ms)
        reranked_latencies.append(search_ms + rerank_ms)
        vector_rankings.append([result.document_id for result in vector_results])
        reranked_rankings.append([result.document_id for result in reranked_results])
        relevant.append(set(item.get("relevant_document_ids", [])))

    vector_metrics = evaluate_rankings(vector_rankings, relevant, vector_latencies)
    reranked_metrics = evaluate_rankings(reranked_rankings, relevant, reranked_latencies)
    report = {
        "dataset": settings.dataset_name,
        "query_sample": len(queries),
        "top_k": settings.top_k,
        "candidate_k": settings.candidate_k,
        "embedding_cache": "cold_isolated" if args.cold_cache else "reused_isolated",
        "shared_query_embedding_mean_ms": round(mean(embedding_latencies), 3),
        "vector_only": vector_metrics.model_dump(),
        "vector_plus_lexical_reranking": reranked_metrics.model_dump(),
    }
    vector_mrr = vector_metrics.mean_reciprocal_rank
    reranked_mrr = reranked_metrics.mean_reciprocal_rank
    report["recommended_mode"] = "lexical" if reranked_mrr > vector_mrr else "none"
    store.close()
    close_embedder = getattr(embedder, "close", None)
    if close_embedder is not None:
        close_embedder()
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
