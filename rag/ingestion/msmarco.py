from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

from rag.chunking.models import Document

LANGUAGE_FILE_PREFIX = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}

_WHITESPACE = re.compile(r"\s+")


class PassageSet(BaseModel):
    is_selected: list[int]
    English_passages: list[str]
    Translated_passages: list[str]


class MSMARCORecord(BaseModel):
    source_lang: str = "eng_Latn"
    target_lang: str
    meta: dict[str, Any] = Field(default_factory=dict)
    query: str
    Answer: str = ""
    query_id: int | str
    query_type: str = "UNKNOWN"
    passages: PassageSet
    Eng_Query: str = ""
    Eng_Answer: str = ""


class IngestionStats(BaseModel):
    records_seen: int = 0
    records_valid: int = 0
    invalid_records: int = 0
    documents_emitted: int = 0
    duplicate_documents: int = 0
    empty_passages: int = 0
    missing_values: int = 0


class MSMARCOIngestor:
    def __init__(
        self,
        dataset_name: str = "ai4bharat/MSMARCO-XI",
        language: str = "hi",
        split: str = "validation",
    ) -> None:
        if language not in LANGUAGE_FILE_PREFIX:
            raise ValueError(
                f"Unsupported language {language!r}; use {sorted(LANGUAGE_FILE_PREFIX)}"
            )
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.dataset_name = dataset_name
        self.language = language
        self.split = split

    @property
    def parquet_url(self) -> str:
        prefix = LANGUAGE_FILE_PREFIX[self.language]
        suffix = "train" if self.split == "train" else "val"
        return (
            f"https://huggingface.co/datasets/{self.dataset_name}/resolve/main/"
            f"{self.split}/{prefix}{suffix}.parquet"
        )

    @property
    def local_parquet_path(self) -> Path:
        prefix = LANGUAGE_FILE_PREFIX[self.language]
        suffix = "train" if self.split == "train" else "val"
        return Path("rag/data/raw") / f"{prefix}{suffix}.parquet"

    def stream(self) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset  # type: ignore[import-untyped]

        source = (
            str(self.local_parquet_path) if self.local_parquet_path.exists() else self.parquet_url
        )
        dataset = load_dataset(
            "parquet",
            data_files={self.split: source},
            split=self.split,
            streaming=True,
        )
        return cast(Iterable[dict[str, Any]], dataset)

    @staticmethod
    def normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "")
        return _WHITESPACE.sub(" ", normalized).strip()

    @staticmethod
    def _document_id(language: str, query_id: int | str, index: int, text: str) -> str:
        digest = hashlib.sha256(f"{language}\0{query_id}\0{index}\0{text}".encode()).hexdigest()[
            :24
        ]
        return f"doc_{digest}"

    def ingest(
        self,
        records: Iterable[dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> tuple[list[Document], list[dict[str, Any]], IngestionStats]:
        source_records = records if records is not None else self.stream()
        stats = IngestionStats()
        documents: list[Document] = []
        evaluation_queries: list[dict[str, Any]] = []
        seen_text_hashes: set[str] = set()

        for raw in source_records:
            if limit is not None and stats.records_seen >= limit:
                break
            stats.records_seen += 1
            try:
                record = MSMARCORecord.model_validate(raw)
            except ValidationError:
                stats.invalid_records += 1
                continue
            stats.records_valid += 1
            translated = record.passages.Translated_passages
            english = record.passages.English_passages
            selected = record.passages.is_selected
            if not record.query or not translated:
                stats.missing_values += 1

            relevant_ids: list[str] = []
            for index, passage in enumerate(translated):
                text = self.normalize(passage)
                if not text:
                    stats.empty_passages += 1
                    continue
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if content_hash in seen_text_hashes:
                    stats.duplicate_documents += 1
                    continue
                seen_text_hashes.add(content_hash)
                document_id = self._document_id(self.language, record.query_id, index, text)
                is_selected = bool(selected[index]) if index < len(selected) else False
                if is_selected:
                    relevant_ids.append(document_id)
                documents.append(
                    Document(
                        document_id=document_id,
                        text=text,
                        source=f"hf://{self.dataset_name}/{self.language}/{self.split}",
                        metadata={
                            "dataset": self.dataset_name,
                            "language": self.language,
                            "target_lang": record.target_lang,
                            "query_id": str(record.query_id),
                            "query_type": record.query_type,
                            "passage_index": index,
                            "is_selected": is_selected,
                            "english_passage": self.normalize(english[index])
                            if index < len(english)
                            else "",
                        },
                    )
                )
                stats.documents_emitted += 1

            query = self.normalize(record.query)
            if query:
                evaluation_queries.append(
                    {
                        "query_id": str(record.query_id),
                        "query": query,
                        "english_query": self.normalize(record.Eng_Query),
                        "answer": self.normalize(record.Answer),
                        "relevant_document_ids": relevant_ids,
                        "language": self.language,
                    }
                )

        return documents, evaluation_queries, stats

    def inspect(
        self, records: Iterable[dict[str, Any]] | None = None, limit: int = 100
    ) -> dict[str, Any]:
        documents, queries, stats = self.ingest(records=records, limit=limit)
        lengths = [len(document.text.split()) for document in documents]
        return {
            "dataset": self.dataset_name,
            "language": self.language,
            "split": self.split,
            "source_url": self.parquet_url,
            "local_cache": str(self.local_parquet_path)
            if self.local_parquet_path.exists()
            else None,
            "schema": list(MSMARCORecord.model_fields),
            "passage_schema": list(PassageSet.model_fields),
            "stats": stats.model_dump(),
            "query_count": len(queries),
            "document_word_lengths": {
                "minimum": min(lengths, default=0),
                "maximum": max(lengths, default=0),
                "average": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            },
        }


def iter_json_records(path: str) -> Iterator[dict[str, Any]]:
    import json
    from pathlib import Path

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
