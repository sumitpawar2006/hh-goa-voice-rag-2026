from __future__ import annotations

import argparse
from pathlib import Path

from rag.chunking.models import Chunk
from scripts.common import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a small, reproducible real-data index seed for deployment."
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("rag/data/bootstrap/chunks.jsonl"))
    args = parser.parse_args()
    source = Path("rag/data/processed/chunks.jsonl")
    if not source.exists():
        raise SystemExit("Run scripts/ingest.py before exporting a deployment seed")
    chunks = [Chunk.model_validate(item) for item in read_jsonl(source)[: args.limit]]
    write_jsonl(args.output, chunks)
    print(f"Exported {len(chunks)} real MSMARCO-XI chunks to {args.output}")


if __name__ == "__main__":
    main()
