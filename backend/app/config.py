from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] | str = ["http://localhost:5173"]
    max_query_chars: int = Field(default=1000, ge=32, le=10_000)
    max_audio_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    elevenlabs_api_key: str | None = None
    elevenlabs_stt_model: str = "scribe_v2"

    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_language: str = "hi"
    dataset_split: Literal["train", "validation"] = "validation"
    dataset_sample_size: int = Field(default=500, ge=1)
    chunking_strategy: Literal["fixed", "overlap", "semantic", "metadata"] = "semantic"
    chunk_size: int = Field(default=180, ge=32, le=2048)
    chunk_overlap: int = Field(default=36, ge=0, le=1024)

    embedding_provider: Literal["fastembed", "hash"] = "fastembed"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    prewarm_embeddings: bool = True
    qdrant_path: Path = Path("rag/data/qdrant")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "msmarco_xi"
    top_k: int = Field(default=5, ge=1, le=25)
    candidate_k: int = Field(default=12, ge=1, le=100)
    min_relevance_score: float = Field(default=0.50, ge=-1, le=1)
    reranker_mode: Literal["none", "lexical", "cross_encoder"] = "lexical"

    generator_provider: Literal["extractive", "llama_cpp", "llama_server"] = "extractive"
    llm_server_url: str = "http://127.0.0.1:8080/v1"
    llm_model_path: Path = Path("rag/data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    llm_context_size: int = Field(default=4096, ge=512, le=32768)
    llm_max_tokens: int = Field(default=256, ge=32, le=2048)
    llm_temperature: float = Field(default=0.1, ge=0, le=2)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {})
        chunk_size = data.get("chunk_size")
        if isinstance(chunk_size, int) and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        if isinstance(self.cors_origins, str):
            return [self.cors_origins]
        return self.cors_origins


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
