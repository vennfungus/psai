"""Configuration for PSAI's local NLP system."""

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass
class NLPConfig:
    """Dense retrieval and classical sentence-extraction settings."""

    embeddings_dir: Path = Path("data/processed/embeddings")
    persistence_directory: Path = Path("data/processed/chroma_db")
    collection_name: str = "psai_chunks"
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    device: str = "cpu"
    top_k: int = 5
    max_answer_sentences: int = 5
    max_answer_characters: int = 2500
    dense_weight: float = 0.65
    lexical_weight: float = 0.35

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
        if self.max_answer_sentences < 1:
            raise ValueError("max_answer_sentences must be >= 1")
        if self.max_answer_characters < 1:
            raise ValueError("max_answer_characters must be >= 1")
        for name, weight in (
            ("dense_weight", self.dense_weight),
            ("lexical_weight", self.lexical_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if abs(self.dense_weight + self.lexical_weight - 1.0) > 1e-8:
            raise ValueError("dense_weight and lexical_weight must sum to 1")
