"""Focused tests for the preprocessing modules."""

from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from PIL import Image, ImageDraw
from pytesseract.pytesseract import TesseractNotFoundError

from preprocessing.annotation import load_annotations
from preprocessing.chunking import chunk_document
from preprocessing.cleaning import clean_text
from preprocessing.config import PreprocessingConfig
from preprocessing.deduplication import find_text_duplicates
from preprocessing.extract_text import extract_pdf_text
from preprocessing.ocr import apply_ocr
from preprocessing.pipeline import run_pipeline
from preprocessing.split_dataset import split_records


class ConfigTests(unittest.TestCase):
    def test_invalid_chunk_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_overlap"):
            PreprocessingConfig(max_chunk_size=10, chunk_overlap=10)

    def test_invalid_split_ratios_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            PreprocessingConfig(
                split_ratios={"train": 0.7, "validation": 0.2, "test": 0.2}
            )


class ExtractTextTests(unittest.TestCase):
    def test_extracts_pages_and_marks_sparse_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "sample.pdf"
            pdf = fitz.open()
            pdf.new_page().insert_text((72, 72), "Project Alpha task T-42")
            pdf.new_page()
            pdf.save(pdf_path)
            pdf.close()

            result = extract_pdf_text(
                pdf_path,
                PreprocessingConfig(min_native_text_chars=5),
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["page_count"], 2)
            self.assertEqual(result["pages"][0]["page_number"], 1)
            self.assertFalse(result["pages"][0]["ocr_candidate"])
            self.assertTrue(result["pages"][1]["ocr_candidate"])
            self.assertEqual(len(result["document_id"]), 64)

    def test_malformed_pdf_returns_a_reported_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "broken.pdf"
            pdf_path.write_text("not a PDF", encoding="utf-8")

            result = extract_pdf_text(pdf_path)

            self.assertEqual(result["status"], "error")
            self.assertIn("Unable to open or read PDF", result["error"])


class OcrTests(unittest.TestCase):
    def test_ocr_only_selects_output_when_native_text_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "pages.pdf"
            pdf = fitz.open()
            pdf.new_page().insert_text((72, 72), "Readable native content")
            scanned_page = pdf.new_page()
            image = Image.new("RGB", (600, 200), "white")
            ImageDraw.Draw(image).text((20, 60), "Scanned page content", fill="black")
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            scanned_page.insert_image(
                scanned_page.rect,
                stream=image_bytes.getvalue(),
            )
            pdf.save(pdf_path)
            pdf.close()
            document = extract_pdf_text(
                pdf_path,
                PreprocessingConfig(min_native_text_chars=5),
            )

            with patch("preprocessing.ocr._validate_tesseract"):
                with patch(
                    "preprocessing.ocr._recognize_page",
                    return_value=("recognized words", 95.0),
                ) as recognize:
                    result = apply_ocr(
                        document,
                        pdf_path,
                        PreprocessingConfig(min_native_text_chars=5),
                    )

            recognize.assert_called_once()
            self.assertEqual(
                result["pages"][0]["text"].strip(), "Readable native content"
            )
            self.assertFalse(result["pages"][0].get("ocr_attempted", False))
            self.assertEqual(result["pages"][1]["text"], "recognized words")
            self.assertTrue(result["pages"][1]["ocr_output_selected"])

    def test_missing_tesseract_has_actionable_error(self) -> None:
        document = {
            "pages": [{"page_number": 1, "ocr_candidate": True, "native_text": ""}]
        }
        with patch(
            "preprocessing.ocr.pytesseract.get_tesseract_version",
            side_effect=TesseractNotFoundError(),
        ):
            with self.assertRaisesRegex(RuntimeError, "Install the Tesseract engine"):
                apply_ocr(document, Path("unused.pdf"))


class TextProcessingTests(unittest.TestCase):
    def test_cleaning_preserves_project_details_and_paragraphs(self) -> None:
        cleaned = clean_text(
            " Project   ID: PRJ-42\r\nTask T-7 depends on T-6; duration: 3 days.\x00"
            "\r\n\r\nBudget: $1,250 "
        )

        self.assertIn("PRJ-42", cleaned)
        self.assertIn("T-7 depends on T-6", cleaned)
        self.assertIn("3 days", cleaned)
        self.assertIn("Budget: $1,250", cleaned)
        self.assertIn("\n\n", cleaned)
        self.assertNotIn("\x00", cleaned)

    def test_chunks_respect_token_limit_and_keep_page_metadata(self) -> None:
        document = {
            "document_id": "doc-1",
            "file_name": "plan.pdf",
            "source_path": "data/raw/plan.pdf",
            "pages": [
                {
                    "page_number": 3,
                    "cleaned_text": (
                        "# Milestones\n\nAlpha task details here.\n\n"
                        "Beta task details here.\n\nGamma task details here."
                    ),
                }
            ],
        }
        result = chunk_document(
            document,
            PreprocessingConfig(
                max_chunk_size=8,
                chunk_overlap=2,
                min_chunk_size=0,
            ),
        )

        self.assertTrue(result["chunks"])
        self.assertTrue(
            all(chunk["token_count"] <= 8 for chunk in result["chunks"])
        )
        self.assertTrue(all(chunk["page_number"] == 3 for chunk in result["chunks"]))
        self.assertEqual(
            len({chunk["chunk_id"] for chunk in result["chunks"]}),
            len(result["chunks"]),
        )


class AnnotationAndDatasetTests(unittest.TestCase):
    def test_jsonl_reports_malformed_lines_and_preserves_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotation_path = Path(directory) / "labels.jsonl"
            annotation_path.write_text(
                '{"label": "  task  ", "future_field": 1}\n'
                '{"missing quote: true}\n',
                encoding="utf-8",
            )

            result = load_annotations(
                annotation_path,
                required_fields=("label",),
                schema={"label": str},
            )

        self.assertEqual(result["records"][0]["label"], "task")
        self.assertEqual(result["records"][0]["future_field"], 1)
        self.assertEqual(result["errors"][0]["line_number"], 2)

    def test_exact_duplicates_are_detected(self) -> None:
        result = find_text_duplicates(
            [
                {"document_id": "one", "text": "Project plan   for T-1"},
                {"document_id": "two", "text": "project plan for t-1"},
            ]
        )

        self.assertEqual(len(result["exact_duplicate_groups"]), 1)
        self.assertEqual(result["near_duplicate_candidates"], [])

    def test_split_keeps_document_groups_together_deterministically(self) -> None:
        records = [
            {"document_id": document_id, "chunk_id": f"{document_id}-{index}"}
            for document_id in ("a", "b", "c", "d", "e")
            for index in range(2)
        ]
        first = split_records(records, random_seed=17)
        second = split_records(records, random_seed=17)
        group_splits = {}
        for split_name, split_records_list in first["splits"].items():
            for record in split_records_list:
                previous_split = group_splits.setdefault(
                    record["document_id"], split_name
                )
                self.assertEqual(previous_split, split_name)

        self.assertEqual(first["summary"]["groups"], second["summary"]["groups"])
        self.assertEqual(sum(first["summary"]["record_counts"].values()), len(records))
        self.assertEqual(len(group_splits), 5)


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_document_chunk_and_report_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "raw"
            output_dir = root / "processed"
            input_dir.mkdir()
            pdf_path = input_dir / "plan.pdf"
            pdf = fitz.open()
            pdf.new_page().insert_text(
                (72, 72),
                "Project plan. Task T-12 depends on T-11 and takes 4 days.",
            )
            pdf.save(pdf_path)
            pdf.close()

            summary = run_pipeline(
                input_dir,
                PreprocessingConfig(
                    raw_pdf_dir=input_dir,
                    processed_dir=output_dir,
                    min_native_text_chars=5,
                ),
                enable_ocr=False,
            )

            self.assertEqual(summary["successful_documents"], 1)
            self.assertTrue((output_dir / "documents.jsonl").is_file())
            self.assertTrue((output_dir / "chunks.jsonl").is_file())
            self.assertTrue((output_dir / "quality_report.json").is_file())
            document = (output_dir / "documents.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("T-12 depends on T-11", document)


if __name__ == "__main__":
    unittest.main()
