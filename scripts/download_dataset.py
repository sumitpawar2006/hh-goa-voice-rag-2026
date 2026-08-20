from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from rag.ingestion.msmarco import LANGUAGE_FILE_PREFIX


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache one official MSMARCO-XI Parquet split.")
    parser.add_argument("--language", choices=sorted(LANGUAGE_FILE_PREFIX), default="hi")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    args = parser.parse_args()
    prefix = LANGUAGE_FILE_PREFIX[args.language]
    suffix = "train" if args.split == "train" else "val"
    filename = f"{args.split}/{prefix}{suffix}.parquet"
    target = Path("rag/data/raw") / Path(filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename=filename,
        repo_type="dataset",
    )
    if Path(cached).resolve() != target.resolve():
        shutil.copy2(cached, target)
    print(f"Cached {filename} at {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
