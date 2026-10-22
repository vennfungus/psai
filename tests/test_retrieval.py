"""Retrieval tests using temporary Chroma collections and a local mock model."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

from retrieval.config import RetrievalConfig
from retrieval.retriever import Retriever
from vector_db.config import VectorDBConfig
from vector_db.store import ChromaVectorStore


class MockQueryEmbedder:
    model_name = "mock-model"

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            if "risk" in text.casefold():
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


class RetrievalTests(unittest.TestCase):
    def _setup_retriever(
        self,
        root: Path,
        *,
        create_collection: bool = True,
    ) -> tuple[Retriever, ChromaVectorStore]:
        embeddings_dir = root / "embeddings"
        embeddings_dir.mkdir()
        records = [
            {
                "chunk_id": "chunk-task",
                "document_id": "doc-project",
                "file_name": "plan.pdf",
                "source_path": "data/raw/plan.pdf",
                "page_number": 4,
                "section_heading": "Dependencies",
                "text": "Task A depends on Task B.",
                "extraction_method": "native",
            },
            {
                "chunk_id": "chunk-risk",
                "document_id": "doc-project",
                "file_name": "plan.pdf",
                "source_path": "data/raw/plan.pdf",
                "page_number": 8,
                "section_heading": "Risk",
                "text": "Risk register for the project.",
                "extraction_method": "native",
            },
            {
                "chunk_id": "chunk-other",
                "document_id": "doc-other",
                "file_name": "other.pdf",
                "source_path": "data/raw/other.pdf",
                "page_number": 1,
                "section_heading": None,
                "text": "Task dependencies and milestones.",
                "extraction_method": "ocr",
            },
        ]
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        np.save(embeddings_dir / "embeddings.npy", vectors, allow_pickle=False)
        (embeddings_dir / "embedding_metadata.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        (embeddings_dir / "embedding_manifest.json").write_text(
            json.dumps(
                {
                    "processed_count": 3,
                    "model_name": "mock-model",
                    "embedding_dimension": 3,
                    "vector_shape": [3, 3],
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

        vector_store = ChromaVectorStore(
            VectorDBConfig(
                embeddings_dir=embeddings_dir,
                persistence_directory=root / "chroma",
                collection_name="retrieval_test",
            )
        )
        if create_collection:
            vector_store.ingest()
        retriever = Retriever(
            RetrievalConfig(
                embeddings_dir=embeddings_dir,
                persistence_directory=root / "chroma",
                collection_name="retrieval_test",
                model_name="mock-model",
            ),
            embedder=MockQueryEmbedder(),
            vector_store=vector_store,
        )
        return retriever, vector_store

    def test_query_ranking_top_k_and_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever, store = self._setup_retriever(Path(directory))
            try:
                result = retriever.retrieve(
                    "What are the task dependencies?", top_k=2
                )
                self.assertEqual(result["count"], 2)
                self.assertEqual(
                    [item["chunk_id"] for item in result["results"]],
                    ["chunk-task", "chunk-other"],
                )
                self.assertEqual(
                    [item["rank"] for item in result["results"]],
                    [1, 2],
                )
                first = result["results"][0]
                self.assertEqual(first["document_id"], "doc-project")
                self.assertEqual(first["source_filename"], "plan.pdf")
                self.assertEqual(first["page_number"], 4)
                self.assertEqual(first["section_heading"], "Dependencies")
                self.assertEqual(first["text"], "Task A depends on Task B.")
                self.assertAlmostEqual(first["similarity"], 1.0)
                self.assertLessEqual(
                    result["results"][0]["distance"],
                    result["results"][1]["distance"],
                )
            finally:
                retriever.close()

    def test_metadata_filter_uses_actual_fields_and_combines_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever, _ = self._setup_retriever(Path(directory))
            try:
                result = retriever.retrieve(
                    "task dependencies",
                    filters={
                        "file_name": "plan.pdf",
                        "page_number": 8,
                    },
                )
                self.assertEqual(
                    [item["chunk_id"] for item in result["results"]],
                    ["chunk-risk"],
                )
                with self.assertRaisesRegex(ValueError, "Unknown metadata filter"):
                    retriever.retrieve("task", filters={"topic": "planning"})
            finally:
                retriever.close()

    def test_empty_query_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever, _ = self._setup_retriever(Path(directory))
            try:
                with self.assertRaisesRegex(ValueError, "query must not be empty"):
                    retriever.retrieve("   ")
            finally:
                retriever.close()

    def test_query_embedding_model_and_dimension_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever, _ = self._setup_retriever(Path(directory))
            retriever._embedder = type(
                "WrongDimensionEmbedder",
                (),
                {
                    "model_name": "mock-model",
                    "get_sentence_embedding_dimension": lambda self: 4,
                    "encode": lambda self, texts, **kwargs: np.ones((1, 4)),
                },
            )()
            try:
                with self.assertRaisesRegex(ValueError, "does not match stored"):
                    retriever.retrieve("task")
            finally:
                retriever.close()

    def test_empty_collection_returns_no_results_without_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever, store = self._setup_retriever(
                Path(directory),
                create_collection=False,
            )
            try:
                # Establish an empty collection with the metric/config expected by Chroma.
                store.client.get_or_create_collection(
                    "retrieval_test",
                    metadata={
                        "hnsw:space": "cosine",
                        "embedding_model": "mock-model",
                        "embedding_dimension": 3,
                        "normalized": True,
                    },
                )
                result = retriever.retrieve("task")
                self.assertEqual(result["count"], 0)
                self.assertEqual(result["results"], [])
            finally:
                retriever.close()

    def test_missing_collection_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever, _ = self._setup_retriever(
                Path(directory),
                create_collection=False,
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "Unable to find collection"):
                    retriever.retrieve("task")
            finally:
                retriever.close()


if __name__ == "__main__":
    unittest.main()
