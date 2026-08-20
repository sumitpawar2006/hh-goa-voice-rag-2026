from __future__ import annotations

import re

from rag.chunking.models import Chunk, Document, stable_chunk_id

_PARAGRAPH = re.compile(r"\n\s*\n+")


class MetadataAwareChunker:
    """Preserves paragraph and record boundaries while attaching retrieval facets."""

    strategy = "metadata"

    def __init__(self, max_size: int = 220) -> None:
        if max_size < 16:
            raise ValueError("max_size must be at least 16")
        self.max_size = max_size

    def chunk(self, document: Document) -> list[Chunk]:
        paragraphs = [part.strip() for part in _PARAGRAPH.split(document.text) if part.strip()]
        if not paragraphs:
            paragraphs = [document.text.strip()]
        groups: list[str] = []
        current: list[str] = []
        size = 0
        for paragraph in paragraphs:
            words = paragraph.split()
            if current and size + len(words) > self.max_size:
                groups.append("\n\n".join(current))
                current, size = [], 0
            while len(words) > self.max_size:
                groups.append(" ".join(words[: self.max_size]))
                words = words[self.max_size :]
            if words:
                current.append(" ".join(words))
                size += len(words)
        if current:
            groups.append("\n\n".join(current))

        chunks: list[Chunk] = []
        for position, text in enumerate(groups):
            metadata = {
                **document.metadata,
                "language": document.metadata.get("language", "unknown"),
                "query_type": document.metadata.get("query_type", "unknown"),
                "passage_selected": bool(document.metadata.get("is_selected", False)),
            }
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=stable_chunk_id(document.document_id, self.strategy, position, text),
                    text=text,
                    source=document.source,
                    metadata=metadata,
                    strategy=self.strategy,
                    position=position,
                    token_count=len(text.split()),
                )
            )
        return chunks
