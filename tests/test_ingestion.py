from __future__ import annotations

from rag.ingestion.msmarco import MSMARCOIngestor


def record(passage: str, *, selected: int = 1) -> dict[str, object]:
    return {
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "meta": {"model_name": "test"},
        "query": "तत्काल प्रभाव क्या था?",
        "Answer": "एक उत्तर",
        "query_id": 42,
        "query_type": "DESCRIPTION",
        "passages": {
            "is_selected": [selected],
            "English_passages": ["What was the impact?"],
            "Translated_passages": [passage],
        },
        "Eng_Query": "What was the impact?",
        "Eng_Answer": "An answer",
    }


def test_ingestion_validates_normalizes_and_deduplicates() -> None:
    ingestor = MSMARCOIngestor(language="hi")
    documents, queries, stats = ingestor.ingest(
        [record("  यह   एक वास्तविक अनुच्छेद है। "), record("यह एक वास्तविक अनुच्छेद है।")]
    )
    assert len(documents) == 1
    assert documents[0].text == "यह एक वास्तविक अनुच्छेद है।"
    assert documents[0].metadata["is_selected"] is True
    assert queries[0]["relevant_document_ids"] == [documents[0].document_id]
    assert stats.duplicate_documents == 1


def test_ingestion_skips_invalid_record() -> None:
    _, _, stats = MSMARCOIngestor(language="hi").ingest([{"query": "missing schema"}])
    assert stats.invalid_records == 1
    assert stats.documents_emitted == 0
