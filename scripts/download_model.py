from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the local Qwen GGUF generator.")
    parser.add_argument("--quant", default="q4_k_m")
    args = parser.parse_args()
    filename = f"qwen2.5-0.5b-instruct-{args.quant}.gguf"
    target_dir = Path("rag/data/models")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF", filename=filename, local_dir=target_dir
    )
    print(f"Local LLM ready: {path}")


if __name__ == "__main__":
    main()
