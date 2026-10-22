"""Configuration for the PSAI retrieval-augmented generation pipeline."""

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass
class RAGConfig:
    """Retrieval, prompt-budget, and OpenAI-compatible model settings."""

    embeddings_dir: Path = Path("data/processed/embeddings")
    persistence_directory: Path = Path("data/processed/chroma_db")
    collection_name: str = "psai_chunks"
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    device: str = "cpu"
    top_k: int = 5
    context_budget_tokens: int = 6000
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 1000
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.embeddings_dir = Path(self.embeddings_dir)
        self.persistence_directory = Path(self.persistence_directory)
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, mps, auto")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.context_budget_tokens < 1:
            raise ValueError("context_budget_tokens must be >= 1")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.api_key_env.strip():
            raise ValueError("api_key_env must not be empty")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if self.max_output_tokens >= self.context_budget_tokens:
            raise ValueError(
                "max_output_tokens must be less than context_budget_tokens"
            )
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
