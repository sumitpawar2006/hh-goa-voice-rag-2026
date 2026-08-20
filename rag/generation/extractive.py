from __future__ import annotations

import re

from rag.generation.models import GeneratedAnswer
from rag.reranking.service import LexicalReranker
from rag.vector_store.base import SearchResult

_SENTENCE = re.compile(r"(?<=[.!?।॥。！？])\s+|\n+")


class ExtractiveGenerator:
    """Key-free, deterministic grounded fallback; it never invents text outside the context."""

    provider_name = "extractive-grounded"

    def generate(self, question: str, contexts: list[SearchResult]) -> GeneratedAnswer:
        if not contexts:
            raise ValueError("At least one context is required")
        query_features = LexicalReranker._features(question)
        candidates: list[tuple[float, str, str]] = []
        for context in contexts[:4]:
            sentences = [part.strip() for part in _SENTENCE.split(context.text) if part.strip()]
            for sentence in sentences or [context.text]:
                score = LexicalReranker._cosine(query_features, LexicalReranker._features(sentence))
                candidates.append((score, sentence, context.chunk_id))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, str, str]] = []
        seen: set[str] = set()
        for item in candidates:
            if item[1] in seen:
                continue
            selected.append(item)
            seen.add(item[1])
            if len(selected) == 2:
                break
        answer = " ".join(item[1] for item in selected).strip()
        source_ids = list(dict.fromkeys(item[2] for item in selected))
        confidence = min(1.0, max(0.2, 0.45 + sum(item[0] for item in selected) / 4))
        return GeneratedAnswer(
            answer=answer,
            source_chunk_ids=source_ids,
            confidence=round(confidence, 3),
        )
