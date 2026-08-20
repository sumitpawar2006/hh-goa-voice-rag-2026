from __future__ import annotations

from pathlib import Path

from backend.app.config import Settings
from rag.chunking.models import Chunk
from scripts.bootstrap_index import bootstrap
from scripts.common import write_jsonl


def test_bootstrap_indexes_seed_and_is_idempotent(tmp_path: Path) -> None:
    seed_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        seed_path,
        [
            Chunk(
                document_id="doc_seed",
                chunk_id="chk_seed",
                text="यह वास्तविक परिनियोजन परीक्षण सामग्री है।",
                source="test://bootstrap",
                metadata={"language": "hi"},
                strategy="semantic",
                position=0,
                token_count=6,
            )
        ],
    )
    settings = Settings(
        app_env="test",
        embedding_provider="hash",
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="bootstrap_test",
        prewarm_embeddings=False,
    )

    assert bootstrap(settings, seed_path) == 1
    assert bootstrap(settings, seed_path) == 1
