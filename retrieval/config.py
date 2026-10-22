"""Configuration for semantic retrieval."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalConfig:
    """Settings shared by query embedding and persistent vector search."""

    embeddings_dir: Path = Path("data/processed/embeddings")
    persistence_directory: Path = Path("data/processed/chroma_db")
    collection_name: str = "psai_chunks"
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"
    top_k: int = 5

    def __post_init__(self) -> None:
        self.embeddings_dir = Path(self.embeddings_dir)
        self.persistence_directory = Path(self.persistence_directory)
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, mps, auto")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
