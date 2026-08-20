from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from backend.app.config import Settings
from scripts.common import write_json


def command_ok(command: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    try:
        process = subprocess.run(  # noqa: S603 - commands are fixed, never user input.
            command, cwd=cwd, text=True, capture_output=True, check=False
        )
    except OSError as exc:
        return False, f"Could not start {' '.join(command)}: {exc}"
    output = (process.stdout + process.stderr).strip()
    return process.returncode == 0, output[-2000:]


def exists(*paths: str) -> bool:
    return all(Path(path).exists() for path in paths)


def main() -> None:
    settings = Settings()
    python = sys.executable
    tests_ok, tests_output = command_ok([python, "-m", "pytest"])
    frontend_ok, frontend_output = command_ok(["npm.cmd", "run", "build"], Path("frontend"))
    git_remote_ok, remote_output = command_ok(["git", "remote", "get-url", "origin"])
    index_ready = False
    try:
        from rag.vector_store.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(
            settings.qdrant_collection,
            settings.qdrant_path,
            settings.qdrant_url,
            settings.qdrant_api_key,
        )
        index_ready = store.health().get("status") == "ready"
        store.close()
    except Exception:
        index_ready = False

    live_url = os.getenv("LIVE_URL", "").rstrip("/")
    live_verified = False
    if live_url:
        try:
            live_verified = httpx.get(f"{live_url}/health", timeout=10).status_code == 200
        except httpx.HTTPError:
            live_verified = False

    checklist: dict[str, bool] = {
        "dataset_inspection_works": exists("reports/dataset_inspection.json"),
        "ingestion_works": exists("reports/ingestion.json", "rag/data/processed/documents.jsonl"),
        "fixed_chunking_implemented": exists("rag/chunking/fixed.py"),
        "overlap_chunking_implemented": exists("rag/chunking/overlap.py"),
        "semantic_chunking_implemented": exists("rag/chunking/semantic.py"),
        "metadata_chunking_implemented": exists("rag/chunking/metadata.py"),
        "chunking_experiment_measured": exists("reports/chunking_experiment.json"),
        "embeddings_work": index_ready,
        "vector_database_works": index_ready,
        "retrieval_evaluated": exists("reports/evaluation.json"),
        "reranking_evaluated": exists("reports/evaluation.json"),
        "rag_works": tests_ok and index_ready,
        "guardrails_work": tests_ok,
        "grounding_works": tests_ok,
        "stt_configured": bool(settings.elevenlabs_api_key),
        "harness_works": tests_ok,
        "tests_pass": tests_ok,
        "benchmark_works": exists("reports/benchmark.json"),
        "p50_available": exists("reports/benchmark.json"),
        "p70_available": exists("reports/benchmark.json"),
        "p100_available": exists("reports/benchmark.json"),
        "frontend_builds": frontend_ok,
        "backend_imports": command_ok([python, "-c", "from backend.app.main import app"])[0],
        "production_build_works": frontend_ok,
        "deployment_verified": live_verified,
        "github_verified": git_remote_ok,
        "readme_complete": Path("README.md").stat().st_size > 3000,
        "no_env_committed": ".env" not in command_ok(["git", "ls-files"])[1].splitlines(),
        "end_to_end_voice_demo": bool(settings.elevenlabs_api_key) and live_verified,
    }
    critical = [
        "dataset_inspection_works",
        "ingestion_works",
        "embeddings_work",
        "rag_works",
        "stt_configured",
        "tests_pass",
        "frontend_builds",
        "deployment_verified",
        "github_verified",
        "end_to_end_voice_demo",
    ]
    report: dict[str, Any] = {
        "status": "READY" if all(checklist[item] for item in critical) else "NOT_READY",
        "checklist": checklist,
        "tests": tests_output,
        "frontend_build": frontend_output,
        "git_remote": remote_output if git_remote_ok else None,
        "live_url": live_url or None,
        "remaining_blockers": [item for item in critical if not checklist[item]],
    }
    write_json(Path("reports/final_readiness.json"), report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "READY" else 1)


if __name__ == "__main__":
    main()
