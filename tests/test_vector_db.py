"""Tests for persistent ChromaDB ingestion and vector search."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vector_db.config import VectorDBConfig
from vector_db.store import ChromaVectorStore


class ChromaVectorStoreTests(unittest.TestCase):
    def _write_artifacts(self, directory: Path) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        np.save(directory / "embeddings.npy", vectors, allow_pickle=False)
        records = [
            {
                "chunk_id": "chunk-a",
                "document_id": "doc-a",
                "file_name": "plan-a.pdf",
                "source_path": "data/raw/plan-a.pdf",
                "page_number": 1,
                "section_heading": "Planning",
                "text": "Project planning tasks",
                "embedding_model": "mock-model",
                "embedding_dimension": 3,
                "optional_field": None,
                "custom_metadata": {"owner": "team-a"},
            },
            {
                "chunk_id": "chunk-b",
                "document_id": "doc-b",
                "file_name": "plan-b.pdf",
                "source_path": "data/raw/plan-b.pdf",
                "page_number": 2,
                "section_heading": "Risk",
                "text": "Risk register",
                "embedding_model": "mock-model",
                "embedding_dimension": 3,
            },
            {
                "chunk_id": "chunk-c",
                "document_id": "doc-c",
                "file_name": "plan-c.pdf",
                "source_path": "data/raw/plan-c.pdf",
                "page_number": 3,
                "section_heading": None,
                "text": "Schedule details",
                "embedding_model": "mock-model",
                "embedding_dimension": 3,
            },
        ]
        (directory / "embedding_metadata.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        (directory / "embedding_manifest.json").write_text(
            json.dumps(
                {
                    "processed_count": 3,
                    "embedding_dimension": 3,
                    "vector_shape": [3, 3],
                    "model_name": "mock-model",
                    "normalized": True,
                    "similarity_metric": "cosine",
                    "outputs": {
                        "vectors": "embeddings.npy",
                        "metadata": "embedding_metadata.jsonl",
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_ingest_is_persistent_and_idempotent_and_query_returns_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            persistence = root / "chroma"
            artifacts.mkdir()
            self._write_artifacts(artifacts)
            config = VectorDBConfig(
                embeddings_dir=artifacts,
                persistence_directory=persistence,
                collection_name="test_chunks",
                batch_size=2,
            )

            first_store = ChromaVectorStore(config)
            first = first_store.ingest()
            first_store.close()
            second_store = ChromaVectorStore(config)
            second = second_store.ingest()
            second_store.close()
            reopened = ChromaVectorStore(config)
            inspected = reopened.inspect()
            result = reopened.query([[1.0, 0.0, 0.0]], n_results=2)
            reopened.close()

            self.assertEqual(first["upserted_count"], 3)
            self.assertEqual(second["upserted_count"], 3)
            self.assertEqual(second["collection_count"], 3)
            self.assertEqual(inspected["count"], 3)
            self.assertEqual(result["ids"][0][0], "chunk-a")
            self.assertEqual(
                result["metadatas"][0][0]["document_id"],
                "doc-a",
            )
            self.assertEqual(
                result["metadatas"][0][0]["file_name"],
                "plan-a.pdf",
            )
            self.assertEqual(
                result["metadatas"][0][0]["page_number"],
                1,
            )
            self.assertEqual(
                result["metadatas"][0][0]["section_heading"],
                "Planning",
            )
            self.assertEqual(
                result["metadatas"][0][0]["custom_metadata"],
                '{"owner":"team-a"}',
            )
            self.assertNotIn("optional_field", result["metadatas"][0][0])
            self.assertEqual(result["documents"][0][0], "Project planning tasks")

    def test_vector_dimension_mismatch_is_rejected_before_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            self._write_artifacts(artifacts)
            store = ChromaVectorStore(
                VectorDBConfig(
                    embeddings_dir=artifacts,
                    persistence_directory=root / "chroma",
                    collection_name="dimension_test",
                )
            )
            store.ingest()
            with self.assertRaisesRegex(
                ValueError, "does not match collection dimension"
            ):
                store.query([[1.0, 0.0]], n_results=1)
            store.close()

    def test_metadata_vector_row_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            self._write_artifacts(artifacts)
            metadata_path = artifacts / "embedding_metadata.jsonl"
            metadata_path.write_text(
                metadata_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            )
            store = ChromaVectorStore(
                VectorDBConfig(
                    embeddings_dir=artifacts,
                    persistence_directory=root / "chroma",
                )
            )
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                store.ingest()
            store.close()


if __name__ == "__main__":
    unittest.main()
