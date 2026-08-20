from __future__ import annotations

from functools import cached_property

from backend.app.config import Settings
from backend.app.stt import ElevenLabsSTT
from rag.embeddings.provider import EmbeddingProvider, build_embedder
from rag.generation.extractive import ExtractiveGenerator
from rag.generation.llama_cpp import LlamaCppGenerator
from rag.generation.llama_server import LlamaServerGenerator
from rag.generation.models import AnswerGenerator
from rag.guardrails.validator import GuardrailValidator
from rag.orchestration.harness import RAGHarness
from rag.reranking.service import CrossEncoderReranker, LexicalReranker, NoOpReranker, Reranker
from rag.retrieval.service import RetrievalService
from rag.vector_store.qdrant_store import QdrantVectorStore


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def embedder(self) -> EmbeddingProvider:
        return build_embedder(
            self.settings.embedding_provider,
            self.settings.embedding_model,
            self.settings.embedding_batch_size,
            threads=self.settings.embedding_threads,
        )

    @cached_property
    def store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            collection=self.settings.qdrant_collection,
            path=self.settings.qdrant_path,
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )

    @cached_property
    def reranker(self) -> Reranker:
        if self.settings.reranker_mode == "none":
            return NoOpReranker()
        if self.settings.reranker_mode == "cross_encoder":
            return CrossEncoderReranker()
        return LexicalReranker()

    @cached_property
    def retrieval(self) -> RetrievalService:
        return RetrievalService(
            self.embedder,
            self.store,
            self.reranker,
            self.settings.top_k,
            self.settings.candidate_k,
        )

    @cached_property
    def generator(self) -> AnswerGenerator:
        if self.settings.generator_provider == "llama_server":
            return LlamaServerGenerator(
                self.settings.llm_server_url,
                self.settings.llm_max_tokens,
                self.settings.llm_temperature,
                self.settings.request_timeout_seconds,
            )
        if self.settings.generator_provider == "llama_cpp":
            return LlamaCppGenerator(
                self.settings.llm_model_path,
                self.settings.llm_context_size,
                self.settings.llm_max_tokens,
                self.settings.llm_temperature,
            )
        return ExtractiveGenerator()

    @cached_property
    def harness(self) -> RAGHarness:
        return RAGHarness(
            self.retrieval,
            self.generator,
            GuardrailValidator(self.settings.min_relevance_score),
            self.settings.max_query_chars,
        )

    @cached_property
    def stt(self) -> ElevenLabsSTT:
        return ElevenLabsSTT(
            self.settings.elevenlabs_api_key,
            self.settings.elevenlabs_stt_model,
            self.settings.request_timeout_seconds,
        )

    async def close(self) -> None:
        if "stt" in self.__dict__:
            await self.stt.close()
        if "generator" in self.__dict__:
            close_generator = getattr(self.generator, "close", None)
            if close_generator is not None:
                close_generator()
        if "embedder" in self.__dict__:
            close_embedder = getattr(self.embedder, "close", None)
            if close_embedder is not None:
                close_embedder()
        if "store" in self.__dict__:
            self.store.close()
