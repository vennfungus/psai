"""OCR fallback for PDF pages with insufficient native text."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import fitz
from PIL import Image
import pytesseract
from pytesseract import Output
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError

from .config import PreprocessingConfig


def _validate_tesseract(language: str) -> None:
    """Raise an actionable error if Tesseract or requested language data is absent."""
    try:
        pytesseract.get_tesseract_version()
    except TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or is not on PATH. Install the "
            "Tesseract engine separately from pytesseract, or configure "
            "pytesseract.pytesseract.tesseract_cmd."
        ) from exc

    try:
        installed_languages = set(pytesseract.get_languages(config=""))
    except TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or is not on PATH. Install the "
            "Tesseract engine separately from pytesseract, or configure "
            "pytesseract.pytesseract.tesseract_cmd."
        ) from exc
    except TesseractError as exc:
        raise RuntimeError(
            f"Could not query Tesseract language data: {exc}"
        ) from exc

    requested_languages = set(language.split("+"))
    missing_languages = requested_languages - installed_languages
    if missing_languages:
        missing = ", ".join(sorted(missing_languages))
        raise RuntimeError(
            f"Tesseract language data missing for: {missing}. Install the "
            "corresponding traineddata files (for example, eng or vie) and "
            "verify Tesseract's tessdata path."
        )


def _recognize_page(
    page: fitz.Page,
    language: str,
    dpi: int,
) -> tuple[str, float | None]:
    """Recognize page text and return its mean word confidence when available."""
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("L")
    result = pytesseract.image_to_data(
        image,
        lang=language,
        output_type=Output.DICT,
    )

    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for word, confidence, block, paragraph, line in zip(
        result["text"],
        result["conf"],
        result["block_num"],
        result["par_num"],
        result["line_num"],
    ):
        word = word.strip()
        if not word:
            continue
        line_key = (int(block), int(paragraph), int(line))
        lines.setdefault(line_key, []).append(word)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            continue
        if confidence_value >= 0:
            confidences.append(confidence_value)

    text_lines = [" ".join(words) for words in lines.values()]
    text = "\n".join(text_lines)
    quality = round(mean(confidences), 2) if confidences else None
    return text, quality


def apply_ocr(
    document: dict[str, Any],
    pdf_path: Path,
    config: PreprocessingConfig | None = None,
    *,
    selected_pages: Iterable[int] | None = None,
    all_pages: bool = False,
) -> dict[str, Any]:
    """Apply OCR to selected pages, preserving native and OCR text separately.

    By default, only pages marked ``ocr_candidate`` are processed. OCR text is
    selected only when native text is empty; non-empty native text remains the
    canonical ``text`` until a later review explicitly chooses otherwise.
    Page selections use one-based page numbers.
    """
    config = config or PreprocessingConfig()
    pdf_path = Path(pdf_path)

    if selected_pages is not None and all_pages:
        raise ValueError("Set either selected_pages or all_pages, not both")

    pages = [dict(page) for page in document.get("pages", [])]
    page_numbers = {page.get("page_number") for page in pages}

    if selected_pages is not None:
        requested_pages = set(selected_pages)
        invalid_pages = requested_pages - page_numbers
        if invalid_pages:
            invalid = ", ".join(str(number) for number in sorted(invalid_pages))
            raise ValueError(f"Selected page numbers are not in the document: {invalid}")
        pages_to_process = requested_pages
    elif all_pages:
        pages_to_process = page_numbers
    else:
        pages_to_process = {
            page["page_number"] for page in pages if page.get("ocr_candidate", False)
        }

    output = dict(document)
    output["pages"] = pages
    if not pages_to_process:
        output["ocr_summary"] = {
            "attempted_pages": 0,
            "selected_pages": 0,
            "language": config.ocr_language,
            "dpi": config.ocr_dpi,
        }
        output["ocr_used"] = False
        return output

    _validate_tesseract(config.ocr_language)

    try:
        with fitz.open(pdf_path) as pdf:
            for page_data in pages:
                page_number = page_data.get("page_number")
                if page_number not in pages_to_process:
                    continue

                page_data["ocr_attempted"] = True
                try:
                    ocr_text, confidence = _recognize_page(
                        pdf.load_page(page_number - 1),
                        config.ocr_language,
                        config.ocr_dpi,
                    )
                except (fitz.FileDataError, OSError, RuntimeError, TesseractError, ValueError) as exc:
                    page_data["ocr_text"] = ""
                    page_data["ocr_quality"] = None
                    page_data["ocr_output_selected"] = False
                    page_data["ocr_error"] = str(exc)
                    continue

                native_text = page_data.get("native_text", "")
                selected = not native_text.strip() and bool(ocr_text.strip())
                page_data["ocr_text"] = ocr_text
                page_data["ocr_quality"] = confidence
                page_data["ocr_output_selected"] = selected
                page_data["ocr_used"] = selected
                if selected:
                    page_data["extraction_method"] = "ocr"
                page_data["text"] = ocr_text if selected else native_text

    except (fitz.FileDataError, fitz.EmptyFileError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Unable to reopen PDF for OCR: {exc}") from exc

    attempted_count = sum(bool(page.get("ocr_attempted")) for page in pages)
    selected_count = sum(bool(page.get("ocr_output_selected")) for page in pages)
    output["ocr_summary"] = {
        "attempted_pages": attempted_count,
        "selected_pages": selected_count,
        "language": config.ocr_language,
        "dpi": config.ocr_dpi,
    }
    output["ocr_used"] = selected_count > 0
    return output