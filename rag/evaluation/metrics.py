from __future__ import annotations

from pydantic import BaseModel


class RetrievalMetrics(BaseModel):
    query_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    failure_rate: float
    mean_latency_ms: float


def evaluate_rankings(
    rankings: list[list[str]],
    relevant_ids: list[set[str]],
    latencies: list[float],
) -> RetrievalMetrics:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    failures = 0
    for ranked, relevant in zip(rankings, relevant_ids, strict=True):
        if not relevant:
            continue
        hits = relevant.intersection(ranked)
        recalls.append(len(hits) / len(relevant))
        first = next((index for index, item in enumerate(ranked, 1) if item in relevant), None)
        reciprocal_ranks.append(1 / first if first else 0)
        failures += int(not hits)
    count = len(recalls)
    return RetrievalMetrics(
        query_count=count,
        recall_at_k=round(sum(recalls) / count, 4) if count else 0,
        mean_reciprocal_rank=round(sum(reciprocal_ranks) / count, 4) if count else 0,
        failure_rate=round(failures / count, 4) if count else 0,
        mean_latency_ms=round(sum(latencies) / len(latencies), 3) if latencies else 0,
    )
