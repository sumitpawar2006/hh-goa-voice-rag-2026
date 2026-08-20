from __future__ import annotations

import pytest

from rag.chunking.models import Document
from rag.chunking.router import ChunkerRouter


@pytest.mark.parametrize("strategy", ["fixed", "overlap", "semantic", "metadata"])
def test_all_chunking_strategies_preserve_provenance(strategy: str) -> None:
    document = Document(
        document_id="doc_1",
        text="First sentence has useful context. Second sentence extends it. " * 12,
        source="hf://dataset",
        metadata={"language": "en", "query_type": "DESCRIPTION"},
    )
    chunks, stats = ChunkerRouter(chunk_size=32, overlap=8).chunk_documents(
        [document],
        strategy,  # type: ignore[arg-type]
    )
    assert chunks
    assert stats.total_chunks == len(chunks)
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.strategy == strategy for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_overlap_configuration_is_reported() -> None:
    document = Document(document_id="d", text="word " * 100, source="test")
    _, stats = ChunkerRouter(20, 5).chunk_documents([document], "overlap")
    assert stats.overlap_tokens == 5
    assert stats.total_chunks > 5
