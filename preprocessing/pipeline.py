"""Command-line orchestration for the PSAI preprocessing pipeline."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .chunking import chunk_document
from .cleaning import clean_document
from .config import PreprocessingConfig
from .data_quality import check_data_quality
from .extract_text import extract_pdf_text
from .ocr import apply_ocr


LOGGER = logging.getLogger("preprocessing")


def _discover_pdfs(input_path: Path, recursive: bool) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a file or directory: {input_path}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(
        (path for path in input_path.glob(pattern) if path.is_file()),
        key=lambda path: str(path).casefold(),
    )


def _write_jsonl(file_path: Path, records: list[dict[str, Any]]) -> None:
    with file_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline(
    input_path: Path,
    config: PreprocessingConfig,
    *,
    enable_ocr: bool = True,
) -> dict[str, Any]:
    """Process a PDF or directory and write traceable JSONL outputs."""
    input_path = Path(input_path)
    pdf_paths = _discover_pdfs(input_path, config.recursive)
    config.ensure_output_dirs()

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        LOGGER.info("Processing %s", pdf_path)
        document = extract_pdf_text(pdf_path, config)
        if document["status"] == "error":
            errors.append(
                {
                    "file_name": document["file_name"],
                    "source_path": document["source_path"],
                    "stage": "extraction",
                    "error": document["error"],
                }
            )
            documents.append(document)
            continue

        if enable_ocr:
            try:
                document = apply_ocr(document, pdf_path, config)
            except RuntimeError as exc:
                errors.append(
                    {
                        "document_id": document["document_id"],
                        "file_name": document["file_name"],
                        "source_path": document["source_path"],
                        "stage": "ocr",
                        "error": str(exc),
                    }
                )

        cleaned_document = clean_document(document)
        chunk_result = chunk_document(cleaned_document, config)
        documents.append(cleaned_document)
        chunks.extend(chunk_result["chunks"])
        LOGGER.info(
            "Processed %s: %d pages, %d chunks",
            pdf_path.name,
            cleaned_document["page_count"],
            chunk_result["statistics"]["chunk_count"],
        )

    document_quality = check_data_quality(documents)
    chunk_quality = check_data_quality(
        chunks,
        max_chunk_size=config.max_chunk_size,
    )
    quality_report = {
        "documents": document_quality,
        "chunks": chunk_quality,
    }

    output_paths = [
        config.processed_dir / config.documents_filename,
        config.processed_dir / config.chunks_filename,
        config.processed_dir / config.errors_filename,
        config.processed_dir / config.quality_filename,
    ]
    source_paths = {path.resolve() for path in pdf_paths}
    if any(path.resolve() in source_paths for path in output_paths):
        raise ValueError("An output filename would overwrite an input PDF")

    _write_jsonl(output_paths[0], documents)
    _write_jsonl(output_paths[1], chunks)
    _write_jsonl(output_paths[2], errors)
    with output_paths[3].open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(quality_report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    summary = {
        "pdf_count": len(pdf_paths),
        "successful_documents": sum(
            document["status"] == "success" for document in documents
        ),
        "failed_documents": sum(
            document["status"] == "error" for document in documents
        ),
        "chunk_count": len(chunks),
        "error_count": len(errors),
        "output_directory": str(config.processed_dir),
        "quality_issue_count": (
            document_quality["issue_count"] + chunk_quality["issue_count"]
        ),
    }
    LOGGER.info("Pipeline summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract, OCR, clean, and chunk manually approved PDF files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="PDF file or directory containing PDFs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for JSONL and quality outputs",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback",
    )
    parser.add_argument(
        "--ocr-language",
        default="eng",
        help="Tesseract language code, for example eng, vie, or vie+eng",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=300,
        help="PDF rendering resolution for OCR",
    )
    parser.add_argument(
        "--min-native-text-chars",
        type=int,
        default=30,
        help="Pages below this non-whitespace character count are OCR candidates",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Do not search input subdirectories",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    config = PreprocessingConfig(
        raw_pdf_dir=args.input_dir,
        processed_dir=args.output_dir,
        ocr_language=args.ocr_language,
        ocr_dpi=args.ocr_dpi,
        min_native_text_chars=args.min_native_text_chars,
        recursive=not args.non_recursive,
    )
    run_pipeline(args.input_dir, config, enable_ocr=not args.no_ocr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())