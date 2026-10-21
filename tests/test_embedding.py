"""Unit tests for JSONL embedding orchestration without downloaded models."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

from embedding.config import EmbeddingConfig
from embedding.pipeline import run_embedding_pipeline


class MockEmbedder:
    device = "cpu"

    def __init__(self, dimension: int = 3, *, wrong_dimension: bool = False) -> None:
        self.dimension = dimension
        self.wrong_dimension = wrong_dimension
        self.encoded_batches: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.encoded_batches.append(list(texts))
        output_dimension = self.dimension + 1 if self.wrong_dimension else self.dimension
        return np.asarray(
            [
                [float(len(text) + index + 1) for index in range(output_dimension)]
                for text in texts
            ],
            dtype=np.float32,
        )


class EmbeddingPipelineTests(unittest.TestCase):
    def test_jsonl_errors_empty_text_and_metadata_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "chunks.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "chunk_id": "chunk-a",
                                "document_id": "doc-a",
                                "file_name": "a.pdf",
                                "page_number": 2,
                                "section_heading": "Schedule",
                                "text": "Plan tasks",
                            }
                        ),
                        json.dumps({"chunk_id": "empty", "text": "  "}),
                        "{bad json",
                        json.dumps(["not", "an", "object"]),
                        json.dumps({"chunk_id": "invalid-text", "text": None}),
                        json.dumps(
                            {
                                "chunk_id": "chunk-b",
                                "document_id": "doc-b",
                                "source_path": "b.pdf",
                                "text": "Risk register",
                                "custom_metadata": {"owner": "team"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "embeddings"
            embedder = MockEmbedder()

            manifest = run_embedding_pipeline(
                input_path,
                EmbeddingConfig(
                    model_name="mock-model",
                    batch_size=1,
                    output_dir=output_dir,
                ),
                embedder=embedder,
            )

            vectors = np.load(output_dir / "embeddings.npy", allow_pickle=False)
            metadata = [
                json.loads(line)
                for line in (output_dir / "embedding_metadata.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(manifest["input_line_count"], 6)
            self.assertEqual(manifest["processed_count"], 2)
            self.assertEqual(manifest["skipped_count"], 1)
            self.assertEqual(manifest["failed_count"], 3)
            self.assertEqual(manifest["vector_shape"], [2, 3])
            self.assertEqual(vectors.shape, (2, 3))
            self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1.0))
            self.assertEqual(
                [record["chunk_id"] for record in metadata],
                ["chunk-a", "chunk-b"],
            )
            self.assertEqual([record["embedding_index"] for record in metadata], [0, 1])
            self.assertEqual(metadata[0]["page_number"], 2)
            self.assertEqual(metadata[0]["section_heading"], "Schedule")
            self.assertEqual(metadata[1]["custom_metadata"]["owner"], "team")
            self.assertTrue((output_dir / "embedding_manifest.json").is_file())

    def test_inconsistent_vector_dimension_marks_batch_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "chunks.jsonl"
            input_path.write_text(
                json.dumps({"chunk_id": "chunk-1", "text": "some text"}) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "embeddings"
            manifest = run_embedding_pipeline(
                input_path,
                EmbeddingConfig(model_name="mock-model", output_dir=output_dir),
                embedder=MockEmbedder(wrong_dimension=True),
            )

            vectors = np.load(output_dir / "embeddings.npy", allow_pickle=False)
            self.assertEqual(manifest["processed_count"], 0)
            self.assertEqual(manifest["failed_count"], 1)
            self.assertEqual(vectors.shape, (0, 3))
            self.assertIn("Expected embedding shape", manifest["errors"][0]["error"])

    def test_invalid_batch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            EmbeddingConfig(batch_size=0)


if __name__ == "__main__":
    unittest.main()
