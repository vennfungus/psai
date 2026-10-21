from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PreprocessingConfig:
    """Shared configuration for PSAI preprocessing."""
    raw_pdf_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

     # PDF validation thresholds
    min_file_size_bytes: int = 100
    max_file_size_bytes: Optional[int] = None
    min_pages: int = 1
    max_pages: Optional[int] = None

    # A page below this threshold may need OCR.
    min_native_text_chars: int = 30

    # OCR configuration for later stages
    ocr_language: str = "eng"

    # Other settings
    recursive: bool = True
    random_seed: int = 42

    def __post_init__(self):
        self.raw_pdf_dir = Path(self.raw_pdf_dir)
        self.processed_dir = Path(self.processed_dir)

        if self.min_file_size_bytes < 0:
            raise ValueError("min_file_size_bytes must be >= 0")

        if (
            self.max_file_size_bytes is not None
            and self.max_file_size_bytes < self.min_file_size_bytes
        ):
            raise ValueError(
                "max_file_size_bytes must be >= min_file_size_bytes"
            )

        if self.min_pages < 1:
            raise ValueError("min_pages must be >= 1")

        if (
            self.max_pages is not None
            and self.max_pages < self.min_pages
        ):
            raise ValueError("max_pages must be >= min_pages")

        if self.min_native_text_chars < 0:
            raise ValueError("min_native_text_chars must be >= 0")

    def ensure_output_dirs(self):
        """Create output directory when required."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
