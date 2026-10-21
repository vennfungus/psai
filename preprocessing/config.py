"""Shared configuration for the PSAI preprocessing pipeline."""

from dataclasses import dataclass, field
import math
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """Configuration for PDF discovery, native extraction, and OCR."""

    raw_pdf_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    ocr_language: str = "eng"
    min_native_text_chars: int = 30
    ocr_dpi: int = 300
    max_chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 20
    tokenizer: str = "regex"
    recursive: bool = True
    random_seed: int = 42
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    split_group_key: str = "document_id"
    documents_filename: str = "documents.jsonl"
    chunks_filename: str = "chunks.jsonl"
    errors_filename: str = "errors.jsonl"
    quality_filename: str = "quality_report.json"

    def __post_init__(self) -> None:
        self.raw_pdf_dir = Path(self.raw_pdf_dir)
        self.processed_dir = Path(self.processed_dir)

        if not self.ocr_language.strip():
            raise ValueError("ocr_language must not be empty")
        if self.min_native_text_chars < 0:
            raise ValueError("min_native_text_chars must be >= 0")
        if self.ocr_dpi < 1:
            raise ValueError("ocr_dpi must be >= 1")
        if self.max_chunk_size < 1:
            raise ValueError("max_chunk_size must be >= 1")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.max_chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and less than max_chunk_size"
            )
        if self.min_chunk_size < 0 or self.min_chunk_size > self.max_chunk_size:
            raise ValueError(
                "min_chunk_size must be >= 0 and no greater than max_chunk_size"
            )
        if not self.tokenizer.strip():
            raise ValueError("tokenizer must not be empty")
        if not self.split_group_key.strip():
            raise ValueError("split_group_key must not be empty")

        expected_splits = {"train", "validation", "test"}
        if set(self.split_ratios) != expected_splits:
            raise ValueError(
                "split_ratios must contain train, validation, and test"
            )
        if any(
            not isinstance(ratio, (int, float))
            or isinstance(ratio, bool)
            or not math.isfinite(ratio)
            or ratio < 0
            for ratio in self.split_ratios.values()
        ):
            raise ValueError("split ratios must be non-negative")
        if abs(sum(self.split_ratios.values()) - 1.0) > 1e-8:
            raise ValueError("split ratios must sum to 1")

        for filename in (
            self.documents_filename,
            self.chunks_filename,
            self.errors_filename,
            self.quality_filename,
        ):
            if not filename or Path(filename).name != filename:
                raise ValueError("Output filenames must be simple filenames")
        if len(
            {
                self.documents_filename,
                self.chunks_filename,
                self.errors_filename,
                self.quality_filename,
            }
        ) != 4:
            raise ValueError("Output filenames must be distinct")

    def ensure_output_dirs(self) -> None:
        """Create the processed output directory when required."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
