from __future__ import annotations

import re

from rag.chunking.models import Chunk, Document, stable_chunk_id

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥。！？])\s+|\n+")


class SemanticChunker:
    """Sentence-boundary chunking with one-sentence continuity between chunks."""

    strategy = "semantic"

    def __init__(self, target_size: int = 180, max_size: int | None = None) -> None:
        if target_size < 16:
            raise ValueError("target_size must be at least 16")
        self.target_size = target_size
        self.max_size = max_size or int(target_size * 1.35)

    def _sentences(self, text: str) -> list[str]:
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
        return sentences or [text.strip()]

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._sentences(document.text)
        groups: list[list[str]] = []
        current: list[str] = []
        current_size = 0

        for sentence in sentences:
            words = sentence.split()
            while len(words) > self.max_size:
                if current:
                    groups.append(current)
                    current = []
                    current_size = 0
                groups.append([" ".join(words[: self.max_size])])
                words = words[self.max_size :]
            if not words:
                continue
            normalized = " ".join(words)
            if current and current_size + len(words) > self.target_size:
                groups.append(current)
                continuity = (
                    current[-1] if len(current[-1].split()) <= self.target_size // 4 else ""
                )
                current = [continuity] if continuity else []
                current_size = len(continuity.split()) if continuity else 0
            current.append(normalized)
            current_size += len(words)
        if current:
            groups.append(current)

        chunks: list[Chunk] = []
        for position, group in enumerate(groups):
            text = " ".join(group).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=stable_chunk_id(document.document_id, self.strategy, position, text),
                    text=text,
                    source=document.source,
                    metadata={**document.metadata, "sentence_count": len(group)},
                    strategy=self.strategy,
                    position=position,
                    token_count=len(text.split()),
                )
            )
        return chunks
