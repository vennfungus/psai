"""Configuration for PSAI's local embedding pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Settings for model inference and embedding artifact output."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"
    output_dir: Path = Path("data/processed/embeddings")
    normalize_embeddings: bool = True
    embeddings_filename: str = "embeddings.npy"
    metadata_filename: str = "embedding_metadata.jsonl"
    manifest_filename: str = "embedding_manifest.json"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, mps, auto")

        filenames = (
            self.embeddings_filename,
            self.metadata_filename,
            self.manifest_filename,
        )
        if any(not name or Path(name).name != name for name in filenames):
            raise ValueError("Output filenames must be simple filenames")
        if len(set(filenames)) != len(filenames):
            raise ValueError("Output filenames must be distinct")
