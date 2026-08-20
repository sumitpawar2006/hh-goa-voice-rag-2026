from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.ingestion.msmarco import LANGUAGE_FILE_PREFIX, MSMARCOIngestor
from scripts.common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the real AI4Bharat MSMARCO-XI schema.")
    parser.add_argument("--language", choices=sorted(LANGUAGE_FILE_PREFIX), default="hi")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_inspection.json"))
    args = parser.parse_args()
    report = MSMARCOIngestor(language=args.language, split=args.split).inspect(limit=args.limit)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Inspection report: {args.output}")


if __name__ == "__main__":
    main()
