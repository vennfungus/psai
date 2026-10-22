"""Question processing with existing retrieval and local NLP."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol

from retrieval.config import RetrievalConfig
from retrieval.retriever import Retriever

from .answerer import HybridAnswerer
from .config import NLPConfig


LOGGER = logging.getLogger("nlp_system")


class RetrievalClient(Protocol):
    """Interface expected from the existing Retrieval implementation."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class NLPSystem:
    """Retrieve relevant chunks and return context-preserving evidence."""

    def __init__(
        self,
        config: NLPConfig | None = None,
        *,
        retriever: RetrievalClient | None = None,
        answerer: HybridAnswerer | None = None,
    ) -> None:
        self.config = config or NLPConfig()
        retrieval_config = RetrievalConfig(
            embeddings_dir=self.config.embeddings_dir,
            persistence_directory=self.config.persistence_directory,
            collection_name=self.config.collection_name,
            model_name=self.config.embedding_model,
            device=self.config.device,
            top_k=self.config.top_k,
        )
        self._retriever = retriever or Retriever(retrieval_config)
        self._answerer = answerer or HybridAnswerer()

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer a question and preserve the real retrieved source references."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty")
        result_limit = self.config.top_k if top_k is None else top_k
        if result_limit < 1:
            raise ValueError("top_k must be >= 1")

        retrieval_result = self._retriever.retrieve(
            question.strip(),
            top_k=result_limit,
            filters=filters,
        )
        retrieved = retrieval_result.get("results")
        if not isinstance(retrieved, list):
            raise RuntimeError("Retrieval returned an invalid results field")
        retrieval_metadata = {
            "collection_name": retrieval_result.get("collection_name"),
            "top_k": retrieval_result.get("top_k", result_limit),
            "retrieved_count": len(retrieved),
            "dense_model": self.config.embedding_model,
            "ranking": "dense semantic retrieval from ChromaDB",
        }
        if not retrieved:
            return {
                "answer": (
                    "Không tìm thấy đoạn tài liệu liên quan, nên chưa thể trả "
                    "lời câu hỏi từ nguồn PSAI hiện có."
                ),
                "sources": [],
                "retrieval_metadata": retrieval_metadata,
                "model_metadata": self._model_metadata(),
                "warnings": ["Không có chunk liên quan; không tạo câu trả lời."],
            }

        extracted = self._answerer.build_answer(
            question,
            retrieved,
            dense_weight=self.config.dense_weight,
            lexical_weight=self.config.lexical_weight,
            max_sentences=self.config.max_answer_sentences,
            max_characters=self.config.max_answer_characters,
        )
        if not extracted["answer"]:
            return {
                "answer": (
                    "Các đoạn được truy xuất không có câu văn bản dùng được, "
                    "nên chưa thể trả lời từ bằng chứng hiện có."
                ),
                "sources": [],
                "retrieval_metadata": retrieval_metadata,
                "model_metadata": self._model_metadata(),
                "warnings": extracted["warnings"],
            }

        retrieval_metadata.update(
            {
                "answer_method": "hybrid_extractive_nlp",
                "dense_weight": self.config.dense_weight,
                "lexical_weight": self.config.lexical_weight,
                "candidate_sentence_count": extracted[
                    "candidate_sentence_count"
                ],
                "selected_sentence_count": extracted[
                    "selected_sentence_count"
                ],
            }
        )
        return {
            "answer": extracted["answer"],
            "sources": extracted["sources"],
            "retrieval_metadata": retrieval_metadata,
            "model_metadata": self._model_metadata(),
            "warnings": extracted["warnings"],
        }

    def _model_metadata(self) -> dict[str, Any]:
        return {
            "answer_method": "classical_extractive_sentence_ranking",
            "query_embedding_model": self.config.embedding_model,
            "requires_llm_api": False,
        }

    def close(self) -> None:
        """Release retrieval resources when supported."""
        close_retriever = getattr(self._retriever, "close", None)
        if callable(close_retriever):
            close_retriever()
