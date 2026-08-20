from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from rag.chunking.models import Document
from rag.chunking.router import ChunkerRouter
from rag.embeddings.provider import HashEmbeddingProvider
from rag.generation.extractive import ExtractiveGenerator
from rag.guardrails.validator import GuardrailValidator
from rag.orchestration.harness import RAGHarness
from rag.reranking.service import LexicalReranker
from rag.retrieval.service import RetrievalService
from rag.vector_store.qdrant_store import QdrantVectorStore


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Provide a Windows-sandbox-safe, isolated temporary directory."""
    root = Path.cwd() / ".test-tmp"
    root.mkdir(exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            document_id="doc_manhattan",
            text=(
                "The immediate impact of the success of the Manhattan Project was the use "
                "of atomic bombs in Japan and the end of the Second World War."
            ),
            source="hf://ai4bharat/MSMARCO-XI/hi/validation",
            metadata={"language": "hi", "query_type": "DESCRIPTION", "is_selected": True},
        ),
        Document(
            document_id="doc_science",
            text="Scientific communication allowed researchers to coordinate complex work.",
            source="hf://ai4bharat/MSMARCO-XI/hi/validation",
            metadata={"language": "hi", "query_type": "DESCRIPTION"},
        ),
        Document(
            document_id="doc_goa",
            text="Goa is a state on the western coast of India.",
            source="hf://ai4bharat/MSMARCO-XI/hi/validation",
            metadata={"language": "hi", "query_type": "LOCATION"},
        ),
    ]


@pytest.fixture
def indexed_pipeline(
    tmp_path: Path, sample_documents: list[Document]
) -> Iterator[dict[str, object]]:
    embedder = HashEmbeddingProvider()
    chunks, _ = ChunkerRouter(chunk_size=32, overlap=8).chunk_documents(
        sample_documents, "semantic"
    )
    store = QdrantVectorStore("test_collection", tmp_path / "qdrant")
    store.ensure_collection(embedder.dimension)
    store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    retrieval = RetrievalService(embedder, store, LexicalReranker(), top_k=3, candidate_k=3)
    harness = RAGHarness(
        retrieval,
        ExtractiveGenerator(),
        GuardrailValidator(minimum_relevance=-1),
    )
    yield {
        "embedder": embedder,
        "chunks": chunks,
        "store": store,
        "retrieval": retrieval,
        "harness": harness,
    }
    store.close()
