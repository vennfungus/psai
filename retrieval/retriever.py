"""Query embedding and persistent ChromaDB retrieval."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from embedding.embedder import SentenceTransformerEmbedder
from vector_db.config import VectorDBConfig
from vector_db.store import ChromaVectorStore

from .config import RetrievalConfig


LOGGER = logging.getLogger("retrieval")


class QueryEmbedder(Protocol):
    """The query encoding interface required by the retriever."""

    model_name: str

    def get_sentence_embedding_dimension(self) -> int: ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Any: ...


def _load_embedding_manifest(embeddings_dir: Path) -> dict[str, Any]:
    manifest_path = embeddings_dir / "embedding_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read embedding manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Embedding manifest must contain a JSON object")
    if not isinstance(manifest.get("model_name"), str):
        raise ValueError("Embedding manifest must specify model_name")
    if (
        not isinstance(manifest.get("embedding_dimension"), int)
        or manifest["embedding_dimension"] < 1
    ):
        raise ValueError("Embedding manifest must specify a positive dimension")
    if not isinstance(manifest.get("normalized"), bool):
        raise ValueError("Embedding manifest must specify normalized")
    if manifest.get("similarity_metric") not in {"cosine", "dot_product"}:
        raise ValueError(
            "Embedding manifest has an unsupported similarity_metric"
        )
    return manifest


def _metadata_filter_schema(embeddings_dir: Path) -> dict[str, set[type]]:
    """Read available scalar filter fields from the actual embedding metadata."""
    manifest = _load_embedding_manifest(embeddings_dir)
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Embedding manifest 'outputs' must be an object")
    filename = outputs.get("metadata", "embedding_metadata.jsonl")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Embedding metadata filename must be a simple filename")
    metadata_path = embeddings_dir / filename
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Embedding metadata not found: {metadata_path}")

    schema: dict[str, set[type]] = {}
    with metadata_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed embedding metadata at line {line_number}: "
                    f"{exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Embedding metadata at line {line_number} must be an object"
                )
            for key, value in record.items():
                if key == "text" or value is None:
                    continue
                if isinstance(value, bool):
                    value_type = bool
                elif isinstance(value, int):
                    value_type = int
                elif isinstance(value, float):
                    value_type = float
                elif isinstance(value, str):
                    value_type = str
                else:
                    value_type = str
                schema.setdefault(key, set()).add(value_type)
    return schema


class Retriever:
    """Perform dense retrieval with the existing PSAI model and Chroma store."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        embedder: QueryEmbedder | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self._embedder = embedder
        vector_config = VectorDBConfig(
            embeddings_dir=self.config.embeddings_dir,
            persistence_directory=self.config.persistence_directory,
            collection_name=self.config.collection_name,
        )
        self._vector_store = vector_store or ChromaVectorStore(vector_config)
        self._manifest: dict[str, Any] | None = None
        self._filter_schema: dict[str, set[type]] | None = None

    def _get_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = _load_embedding_manifest(self.config.embeddings_dir)
        return self._manifest

    def _get_embedder(self) -> QueryEmbedder:
        manifest = self._get_manifest()
        if self.config.model_name != manifest["model_name"]:
            raise ValueError(
                f"Configured query model {self.config.model_name!r} does not "
                f"match stored embedding model {manifest['model_name']!r}"
            )
        if self._embedder is None:
            try:
                self._embedder = SentenceTransformerEmbedder(
                    self.config.model_name,
                    self.config.device,
                )
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to load query embedding model "
                    f"{self.config.model_name!r}: {exc}"
                ) from exc

        embedder_model = getattr(self._embedder, "model_name", None)
        if embedder_model is not None and embedder_model != manifest["model_name"]:
            raise ValueError(
                f"Query embedder model {embedder_model!r} does not match "
                f"stored embedding model {manifest['model_name']!r}"
            )
        dimension = self._embedder.get_sentence_embedding_dimension()
        if dimension != manifest["embedding_dimension"]:
            raise ValueError(
                f"Query model dimension {dimension} does not match stored "
                f"embedding dimension {manifest['embedding_dimension']}"
            )
        return self._embedder

    def _validate_filters(
        self,
        filters: Mapping[str, Any] | None,
    ) -> dict[str, str | int | float | bool] | None:
        if not filters:
            return None
        if self._filter_schema is None:
            self._filter_schema = _metadata_filter_schema(
                self.config.embeddings_dir
            )

        validated: dict[str, str | int | float | bool] = {}
        for key, value in filters.items():
            if key not in self._filter_schema:
                available = ", ".join(sorted(self._filter_schema))
                raise ValueError(
                    f"Unknown metadata filter field {key!r}. Available fields: "
                    f"{available}"
                )
            if value is None or not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"Filter {key!r} must use a non-null scalar value"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Filter {key!r} must be finite")
            expected_types = self._filter_schema[key]
            value_type = bool if isinstance(value, bool) else type(value)
            if expected_types != {value_type}:
                expected = ", ".join(
                    sorted(value_type.__name__ for value_type in expected_types)
                )
                raise ValueError(
                    f"Filter {key!r} must have metadata type {expected}; "
                    f"received {value_type.__name__}"
                )
            validated[key] = value
        return validated

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return ranked chunk records for a natural-language query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        result_limit = self.config.top_k if top_k is None else top_k
        if result_limit < 1:
            raise ValueError("top_k must be >= 1")
        where = self._validate_filters(filters)

        collection_info = self._vector_store.inspect()
        collection_count = collection_info["count"]
        if collection_count == 0:
            return {
                "query": query.strip(),
                "top_k": result_limit,
                "collection_name": self.config.collection_name,
                "count": 0,
                "results": [],
            }

        manifest = self._get_manifest()
        embedder = self._get_embedder()
        try:
            encoded = np.asarray(
                embedder.encode(
                    [query.strip()],
                    batch_size=1,
                    normalize_embeddings=manifest["normalized"],
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to encode retrieval query: {exc}") from exc

        expected_dimension = manifest["embedding_dimension"]
        if encoded.shape != (1, expected_dimension):
            raise ValueError(
                f"Query embedding must have shape (1, {expected_dimension}); "
                f"received {encoded.shape}"
            )
        if not np.isfinite(encoded).all():
            raise ValueError("Query embedding contains NaN or infinite values")
        if manifest["normalized"]:
            norm = float(np.linalg.norm(encoded[0]))
            if norm <= 0:
                raise ValueError("Query embedding has zero length")
            encoded[0] /= norm

        chroma_where: dict[str, Any] | None = where
        if where and len(where) > 1:
            chroma_where = {
                "$and": [{key: value} for key, value in where.items()]
            }
        response = self._vector_store.query(
            encoded[0].tolist(),
            n_results=result_limit,
            where=chroma_where,
        )
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        if not (
            len(ids) == len(documents) == len(metadatas) == len(distances)
        ):
            raise RuntimeError("ChromaDB returned misaligned query result fields")

        metric = collection_info.get("metadata", {}).get("hnsw:space")
        results: list[dict[str, Any]] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            metadata = metadata or {}
            distance_value = float(distance)
            result: dict[str, Any] = {
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": metadata.get("document_id"),
                "text": text,
                "distance": distance_value,
                "score_type": "distance",
                "source_filename": metadata.get("file_name"),
                "page_number": metadata.get("page_number"),
                "section_heading": metadata.get("section_heading"),
                "metadata": metadata,
            }
            if metric == "cosine":
                result["similarity"] = 1.0 - distance_value
                result["score_type"] = "cosine_distance_and_similarity"
            results.append(result)

        return {
            "query": query.strip(),
            "top_k": result_limit,
            "collection_name": self.config.collection_name,
            "count": len(results),
            "results": results,
        }

    def close(self) -> None:
        """Release the persistent ChromaDB client."""
        self._vector_store.close()
