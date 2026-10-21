"""Extract native text from PDF files for PSAI."""

import hashlib
from pathlib import Path
from typing import Any

import fitz

from .config import PreprocessingConfig


def calculate_sha256(file_path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def extract_pdf_text(
    pdf_path: Path,
    config: PreprocessingConfig | None = None,
) -> dict[str, Any]:
    """Extract text page by page and identify potential OCR candidates.

    Unreadable and encrypted PDFs are returned with ``status="error"`` and
    an explanatory ``error`` field so a batch caller can report the failure
    without losing the source filename.
    """
    pdf_path = Path(pdf_path)
    config = config or PreprocessingConfig()

    result: dict[str, Any] = {
        "document_id": None,
        "file_name": pdf_path.name,
        "source_path": str(pdf_path.resolve()),
        "page_count": 0,
        "metadata": {},
        "extraction_metadata": {"engine": "PyMuPDF"},
        "pages": [],
        "status": "error",
        "error": None,
    }

    try:
        result["document_id"] = calculate_sha256(pdf_path)
    except OSError as exc:
        result["error"] = f"Unable to read PDF for hashing: {exc}"
        return result

    try:
        with fitz.open(pdf_path) as pdf:
            if pdf.is_encrypted:
                result["error"] = (
                    "PDF is encrypted and cannot be extracted without a password."
                )
                return result

            result["page_count"] = pdf.page_count
            result["metadata"] = pdf.metadata or {}

            for page_index in range(pdf.page_count):
                page_number = page_index + 1
                try:
                    native_text = pdf.load_page(page_index).get_text("text")
                except (RuntimeError, ValueError) as exc:
                    result["pages"].append(
                        {
                            "page_number": page_number,
                            "native_text": "",
                            "text": "",
                            "native_char_count": 0,
                            "ocr_candidate": True,
                            "extraction_method": "native",
                            "ocr_attempted": False,
                            "ocr_output_selected": False,
                            "ocr_used": False,
                            "extraction_error": str(exc),
                        }
                    )
                    continue

                native_char_count = sum(
                    not character.isspace() for character in native_text
                )
                result["pages"].append(
                    {
                        "page_number": page_number,
                        "native_text": native_text,
                        "text": native_text,
                        "native_char_count": native_char_count,
                        "extraction_method": "native",
                        "ocr_attempted": False,
                        "ocr_output_selected": False,
                        "ocr_used": False,
                        "ocr_candidate": (
                            native_char_count < config.min_native_text_chars
                        ),
                    }
                )

        result["status"] = "success"
        return result
    except (fitz.FileDataError, fitz.EmptyFileError, OSError, RuntimeError, ValueError) as exc:
        result["error"] = f"Unable to open or read PDF: {exc}"
        return result
