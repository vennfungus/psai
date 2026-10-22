"""Single-turn retrieval-augmented answer generation."""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Protocol

from retrieval.config import RetrievalConfig
from retrieval.retriever import Retriever

from .config import RAGConfig
from .llm_client import LLMClient, OpenAICompatibleClient
from .prompt_builder import PromptBuilder


LOGGER = logging.getLogger("rag")
_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


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


class RAGGenerator:
    """Retrieve evidence, build a bounded prompt, and generate a grounded answer."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        *,
        retriever: RetrievalClient | None = None,
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        retrieval_config = RetrievalConfig(
            embeddings_dir=self.config.embeddings_dir,
            persistence_directory=self.config.persistence_directory,
            collection_name=self.config.collection_name,
            model_name=self.config.embedding_model,
            device=self.config.device,
            top_k=self.config.top_k,
        )
        self._retriever = retriever or Retriever(retrieval_config)
        self._llm_client = llm_client or OpenAICompatibleClient(self.config)
        self._prompt_builder = prompt_builder or PromptBuilder()

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
        }
        if not retrieved:
            return {
                "answer": (
                    "I could not find relevant information in the retrieved "
                    "PSAI documents, so I cannot answer this question from the "
                    "available evidence."
                ),
                "sources": [],
                "retrieval_metadata": retrieval_metadata,
                "model_metadata": self._model_metadata(),
                "warnings": ["No relevant chunks were retrieved; the LLM was not called."],
            }

        prompt = self._prompt_builder.build(
            question,
            retrieved,
            context_budget_tokens=self.config.context_budget_tokens,
            max_output_tokens=self.config.max_output_tokens,
        )
        if not prompt["sources"]:
            return {
                "answer": (
                    "The retrieved results did not contain usable text within "
                    "the configured context budget, so I cannot answer from "
                    "the available evidence."
                ),
                "sources": [],
                "retrieval_metadata": retrieval_metadata,
                "model_metadata": self._model_metadata(),
                "warnings": prompt["warnings"],
            }

        answer_text = self._llm_client.generate(prompt["messages"])
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise RuntimeError("LLM client returned an empty or invalid answer")
        source_refs = {source["source_ref"] for source in prompt["sources"]}
        cited_refs = set(_CITATION_PATTERN.findall(answer_text))
        invalid_refs = sorted(cited_refs - source_refs)
        warnings = list(prompt["warnings"])
        if invalid_refs:
            warnings.append(
                "The model returned citation reference(s) not present in the "
                f"supplied context: {', '.join(invalid_refs)}."
            )

        retrieval_metadata.update(
            {
                "included_source_count": len(prompt["sources"]),
                "estimated_input_tokens": prompt["estimated_input_tokens"],
                "available_input_tokens": prompt["available_input_tokens"],
                "shortened_chunk_ids": prompt["shortened_chunk_ids"],
                "omitted_chunk_ids": prompt["omitted_chunk_ids"],
            }
        )
        return {
            "answer": answer_text.strip(),
            "sources": prompt["sources"],
            "retrieval_metadata": retrieval_metadata,
            "model_metadata": self._model_metadata(),
            "warnings": warnings,
        }

    def _model_metadata(self) -> dict[str, Any]:
        return {
            "provider": "openai-compatible",
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
        }

    def close(self) -> None:
        """Release owned provider and retrieval resources when supported."""
        close_llm = getattr(self._llm_client, "close", None)
        if callable(close_llm):
            close_llm()
        close_retriever = getattr(self._retriever, "close", None)
        if callable(close_retriever):
            close_retriever()
