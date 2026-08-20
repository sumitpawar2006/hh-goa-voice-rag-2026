from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from rag.chunking.models import Chunk
from rag.vector_store.base import SearchResult

_POINT_NAMESPACE = uuid.UUID("53f24176-0604-44ff-90a3-f0fe930627a6")


class QdrantVectorStore:
    def __init__(
        self,
        collection: str,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.collection = collection
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=10)
            self.mode = "remote"
        else:
            resolved_path = Path(path or "rag/data/qdrant")
            resolved_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(resolved_path))
            self.mode = "local"

    def ensure_collection(self, dimension: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if recreate and exists:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            points = []
            for chunk, vector in zip(
                chunks[start : start + batch_size],
                vectors[start : start + batch_size],
                strict=True,
            ):
                points.append(
                    models.PointStruct(
                        id=self._point_id(chunk.chunk_id),
                        vector=vector,
                        payload={
                            "text": chunk.text,
                            "document_id": chunk.document_id,
                            "chunk_id": chunk.chunk_id,
                            "source": chunk.source,
                            "metadata": chunk.metadata,
                            "strategy": chunk.strategy,
                            "position": chunk.position,
                        },
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(chunks)

    @staticmethod
    def _filter(filters: dict[str, Any] | None) -> models.Filter | None:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            payload_key = key if key.startswith("metadata.") else f"metadata.{key}"
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(key=payload_key, match=models.MatchAny(any=value))
                )
            else:
                conditions.append(
                    models.FieldCondition(key=payload_key, match=models.MatchValue(value=value))
                )
        return models.Filter(must=conditions)

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=self._filter(filters),
            limit=top_k,
            with_payload=True,
        )
        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    text=str(payload.get("text", "")),
                    document_id=str(payload.get("document_id", "")),
                    chunk_id=str(payload.get("chunk_id", "")),
                    similarity_score=float(point.score),
                    metadata=dict(payload.get("metadata", {})),
                    source=str(payload.get("source", "")),
                    strategy=str(payload.get("strategy", "unknown")),
                    position=int(payload.get("position", 0)),
                )
            )
        return results

    def delete(self, filters: dict[str, Any]) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._filter(filters) or models.Filter()),
            wait=True,
        )

    def reindex(self, dimension: int) -> None:
        self.ensure_collection(dimension=dimension, recreate=True)

    def health(self) -> dict[str, Any]:
        if not self.client.collection_exists(self.collection):
            return {
                "status": "empty",
                "mode": self.mode,
                "collection": self.collection,
                "points": 0,
            }
        info = self.client.get_collection(self.collection)
        return {
            "status": "ready",
            "mode": self.mode,
            "collection": self.collection,
            "points": int(info.points_count or 0),
        }

    def close(self) -> None:
        self.client.close()
