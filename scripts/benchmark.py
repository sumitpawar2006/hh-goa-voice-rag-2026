from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from backend.app.config import Settings
from backend.app.services import ServiceContainer
from rag.embeddings.cache import EmbeddingCache
from scripts.common import read_jsonl, write_json


def percentiles(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def rank(percentile: float) -> float:
        return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]

    return {
        "p50": round(rank(0.5), 3),
        "p70": round(rank(0.7), 3),
        "p100": round(ordered[-1], 3),
        "samples": len(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real text-query RAG latency.")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cold-queries", action="store_true")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    source = read_jsonl(Path("rag/data/processed/evaluation_queries.jsonl"))[: args.queries]
    if not source:
        raise SystemExit("Run scripts/ingest.py before benchmarking")
    settings = Settings()
    if args.cold_queries and settings.embedding_provider == "fastembed":
        cache = EmbeddingCache(Path("rag/data/cache/embeddings.sqlite3"))
        cache.delete_many(settings.embedding_model, [str(item["query"]) for item in source])
        cache.close()
    services = ServiceContainer(settings)
    if not args.no_prewarm:
        services.embedder.warmup()
    stages: dict[str, list[float]] = {
        name: []
        for name in ("embedding", "vector_search", "reranking", "generation", "grounding", "total")
    }
    outcomes = {"grounded": 0, "refused": 0, "failed": 0}
    for _ in range(args.iterations):
        for item in source:
            try:
                response = services.harness.run(str(item["query"]))
            except Exception:
                outcomes["failed"] += 1
                continue
            outcomes["grounded" if response.grounded else "refused"] += 1
            for name in stages:
                stages[name].append(float(getattr(response.latency, f"{name}_ms")))
    if not stages["total"]:
        raise SystemExit("All benchmark requests failed")
    report = {
        "scope": "RAG text query; excludes external speech-to-text",
        "generator": services.settings.generator_provider,
        "embedding_cache": "cold_queries" if args.cold_queries else "configured_state",
        "model_prewarmed": not args.no_prewarm,
        "query_count": len(source),
        "iterations": args.iterations,
        "outcomes": outcomes,
        "latency_ms": {name: percentiles(values) for name, values in stages.items()},
        "under_200ms_p100": max(stages["total"]) < 200,
    }
    asyncio.run(services.close())
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
