"""Configuration for PSAI's persistent vector database."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VectorDBConfig:
    """Settings for a persistent ChromaDB collection."""

    embeddings_dir: Path = Path("data/processed/embeddings")
    persistence_directory: Path = Path("data/processed/chroma_db")
    collection_name: str = "psai_chunks"
    batch_size: int = 256
    distance_metric: str | None = None

    def __post_init__(self) -> None:
        self.embeddings_dir = Path(self.embeddings_dir)
        self.persistence_directory = Path(self.persistence_directory)

        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.distance_metric not in {None, "cosine", "l2", "ip"}:
            raise ValueError("distance_metric must be cosine, l2, or ip")
