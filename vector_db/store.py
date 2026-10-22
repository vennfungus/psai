"""Persistent ChromaDB ingestion, inspection, and vector queries."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import VectorDBConfig


LOGGER = logging.getLogger("vector_db")


def _load_embedding_artifacts(
    embeddings_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Load and validate the vector matrix, aligned metadata, and manifest."""
    manifest_path = embeddings_dir / "embedding_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read embedding manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Embedding manifest must contain a JSON object")

    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Embedding manifest 'outputs' must be an object")
    vector_name = outputs.get("vectors", "embeddings.npy")
    metadata_name = outputs.get("metadata", "embedding_metadata.jsonl")
    if (
        not isinstance(vector_name, str)
        or Path(vector_name).name != vector_name
        or not isinstance(metadata_name, str)
        or Path(metadata_name).name != metadata_name
    ):
        raise ValueError("Embedding output filenames must be simple filenames")

    vector_path = embeddings_dir / vector_name
    metadata_path = embeddings_dir / metadata_name
    if not vector_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Embedding artifacts referenced by the manifest are missing"
        )

    try:
        vectors = np.load(vector_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load embedding vectors: {exc}") from exc
    if vectors.ndim != 2:
        raise ValueError(
            f"Embeddings must be a 2D matrix; got shape {vectors.shape}"
        )
    if not np.issubdtype(vectors.dtype, np.number):
        raise ValueError(f"Embeddings must be numeric; got dtype {vectors.dtype}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinite values")

    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank metadata line at {metadata_path}:{line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed metadata JSON at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Metadata record at line {line_number} must be an object"
                )
            records.append(record)

    if len(records) != vectors.shape[0]:
        raise ValueError(
            "Embedding/metadata row count mismatch: "
            f"{vectors.shape[0]} vectors versus {len(records)} metadata rows"
        )

    dimension = vectors.shape[1]
    model_name = manifest.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Embedding manifest must specify model_name")
    manifest_dimension = manifest.get("embedding_dimension")
    if manifest_dimension != dimension:
        raise ValueError(
            "Embedding dimension does not match manifest: "
            f"{dimension} versus {manifest_dimension}"
        )
    manifest_shape = manifest.get("vector_shape")
    if manifest_shape is not None and manifest_shape != list(vectors.shape):
        raise ValueError(
            f"Embedding matrix shape {list(vectors.shape)} does not match "
            f"manifest shape {manifest_shape}"
        )
    if manifest.get("processed_count") != len(records):
        raise ValueError(
            "Metadata row count does not match manifest processed_count"
        )

    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Metadata record at index {index} is missing a valid chunk_id"
            )
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id in embedding metadata: {chunk_id}")
        seen_ids.add(chunk_id)
        embedding_index = record.get("embedding_index")
        if embedding_index is not None and embedding_index != index:
            raise ValueError(
                f"Metadata row {index} has embedding_index {embedding_index}"
            )
        record_dimension = record.get("embedding_dimension")
        if record_dimension is not None and record_dimension != dimension:
            raise ValueError(
                f"Chunk {chunk_id!r} dimension {record_dimension} does not "
                f"match matrix dimension {dimension}"
            )
        record_model = record.get("embedding_model")
        if record_model is not None and record_model != model_name:
            raise ValueError(
                f"Chunk {chunk_id!r} model {record_model!r} does not match "
                f"manifest model {model_name!r}"
            )

    return vectors, records, manifest


def _chroma_metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert JSON metadata to Chroma's supported scalar metadata values."""
    metadata: dict[str, str | int | float | bool] = {}
    for key, value in record.items():
        if key == "text" or value is None:
            continue
        if isinstance(value, bool):
            metadata[key] = value
        elif isinstance(value, int):
            metadata[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Metadata field {key!r} must be finite")
            metadata[key] = value
        elif isinstance(value, str):
            metadata[key] = value
        else:
            try:
                metadata[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Metadata field {key!r} is not JSON serialisable"
                ) from exc
    return metadata


class ChromaVectorStore:
    """Create or access a persistent collection of PSAI chunk vectors."""

    def __init__(
        self,
        config: VectorDBConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or VectorDBConfig()
        if client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError(
                    "ChromaDB is required. Install dependencies with "
                    "`python -m pip install -r requirements.txt`."
                ) from exc
            self.config.persistence_directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.config.persistence_directory)
            )
        self.client = client
        self.collection: Any | None = None

    def _collection(
        self,
        metric: str,
        model_name: str,
        dimension: int,
        normalized: bool,
    ) -> Any:
        try:
            collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "hnsw:space": metric,
                    "embedding_model": model_name,
                    "embedding_dimension": dimension,
                    "normalized": normalized,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to open ChromaDB collection "
                f"{self.config.collection_name!r}: {exc}"
            ) from exc

        collection_metadata = collection.metadata or {}
        existing_metric = collection_metadata.get("hnsw:space")
        if existing_metric is not None and existing_metric != metric:
            raise ValueError(
                f"Collection uses distance metric {existing_metric!r}, but "
                f"the embedding configuration requires {metric!r}. Use a "
                "different collection name or compatible embeddings."
            )
        existing_model = collection_metadata.get("embedding_model")
        if existing_model is not None and existing_model != model_name:
            raise ValueError(
                f"Collection uses embedding model {existing_model!r}, but "
                f"input uses {model_name!r}. Use a separate collection."
            )
        existing_dimension = collection_metadata.get("embedding_dimension")
        if existing_dimension is not None and existing_dimension != dimension:
            raise ValueError(
                f"Collection dimension {existing_dimension} does not match "
                f"input dimension {dimension}. Use a separate collection."
            )
        existing_normalized = collection_metadata.get("normalized")
        if (
            existing_normalized is not None
            and existing_normalized != normalized
        ):
            raise ValueError(
                "Collection normalisation setting does not match the input "
                "embedding manifest. Use a separate collection."
            )
        self.collection = collection
        return collection

    def ingest(self) -> dict[str, Any]:
        """Upsert existing embeddings in batches without regenerating vectors."""
        vectors, records, manifest = _load_embedding_artifacts(
            self.config.embeddings_dir
        )
        manifest_metric = manifest.get("similarity_metric")
        metric_map = {"cosine": "cosine", "dot_product": "ip"}
        if manifest_metric not in metric_map:
            raise ValueError(
                f"Unsupported embedding similarity metric: {manifest_metric!r}"
            )
        metric = metric_map[manifest_metric]
        if (
            self.config.distance_metric is not None
            and self.config.distance_metric != metric
        ):
            raise ValueError(
                f"Configured distance metric {self.config.distance_metric!r} "
                f"is incompatible with the embedding manifest metric "
                f"{manifest_metric!r} (Chroma metric {metric!r})"
            )

        model_name = manifest.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Embedding manifest must specify model_name")
        normalized = manifest.get("normalized")
        if not isinstance(normalized, bool):
            raise ValueError("Embedding manifest must specify normalized")
        collection = self._collection(
            metric,
            model_name,
            int(vectors.shape[1]),
            normalized,
        )
        try:
            existing_count = collection.count()
            if existing_count:
                sample = collection.get(
                    limit=1,
                    include=["embeddings"],
                )
                existing_vectors = sample.get("embeddings")
                if existing_vectors is not None and len(existing_vectors):
                    existing_dimension = len(existing_vectors[0])
                    if existing_dimension != vectors.shape[1]:
                        raise ValueError(
                            "Existing collection vector dimension "
                            f"{existing_dimension} does not match input "
                            f"dimension {vectors.shape[1]}"
                        )
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Unable to inspect existing collection: {exc}") from exc

        ids = [record["chunk_id"] for record in records]
        documents: list[str | None] = []
        metadatas: list[dict[str, str | int | float | bool]] = []
        for index, record in enumerate(records):
            text = record.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError(
                    f"Chunk {ids[index]!r} has a non-string text field"
                )
            documents.append(text)
            metadatas.append(_chroma_metadata(record))

        inserted_or_updated = 0
        for start in range(0, len(ids), self.config.batch_size):
            end = min(start + self.config.batch_size, len(ids))
            try:
                collection.upsert(
                    ids=ids[start:end],
                    embeddings=vectors[start:end].astype(np.float32).tolist(),
                    metadatas=metadatas[start:end],
                    documents=documents[start:end],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"ChromaDB upsert failed for records {start}–{end - 1}; "
                    f"check collection dimension and metadata: {exc}"
                ) from exc
            inserted_or_updated += end - start

        try:
            collection_count = collection.count()
        except Exception as exc:
            raise RuntimeError(f"Unable to count collection records: {exc}") from exc
        return {
            "collection_name": self.config.collection_name,
            "persistence_directory": str(self.config.persistence_directory),
            "embedding_model": manifest.get("model_name"),
            "embedding_dimension": int(vectors.shape[1]),
            "distance_metric": metric,
            "normalized": manifest.get("normalized"),
            "input_count": len(records),
            "upserted_count": inserted_or_updated,
            "collection_count": collection_count,
            "batch_size": self.config.batch_size,
        }

    def inspect(self) -> dict[str, Any]:
        """Return collection name, metadata, count, and a small sample."""
        if self.collection is None:
            try:
                collection = self.client.get_collection(
                    name=self.config.collection_name
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to find collection {self.config.collection_name!r}: {exc}"
                ) from exc
            self.collection = collection
        try:
            count = self.collection.count()
            sample = self.collection.get(
                limit=min(count, 5),
                include=["metadatas", "documents"],
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to inspect collection: {exc}") from exc
        return {
            "collection_name": self.config.collection_name,
            "count": count,
            "metadata": self.collection.metadata or {},
            "sample_ids": sample.get("ids", []),
            "sample_metadatas": sample.get("metadatas", []),
            "sample_documents": sample.get("documents", []),
        }

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        *,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query by one or more caller-provided vectors."""
        if n_results < 1:
            raise ValueError("n_results must be >= 1")
        vectors = np.asarray(query_embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError("query_embeddings must be a vector or 2D matrix")
        if not np.isfinite(vectors).all():
            raise ValueError("Query vectors contain NaN or infinite values")

        if self.collection is None:
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to find collection {self.config.collection_name!r}: {exc}"
                ) from exc

        collection_count = self.collection.count()
        if collection_count == 0:
            return {
                "ids": [[] for _ in range(vectors.shape[0])],
                "distances": [[] for _ in range(vectors.shape[0])],
                "metadatas": [[] for _ in range(vectors.shape[0])],
                "documents": [[] for _ in range(vectors.shape[0])],
            }

        sample = self.collection.get(limit=1, include=["embeddings"])
        stored = sample.get("embeddings")
        if stored is None or len(stored) == 0:
            raise RuntimeError("Unable to determine collection vector dimension")
        expected_dimension = len(stored[0])
        if vectors.shape[1] != expected_dimension:
            raise ValueError(
                f"Query vector dimension {vectors.shape[1]} does not match "
                f"collection dimension {expected_dimension}"
            )

        include = ["metadatas", "documents", "distances"]
        try:
            return self.collection.query(
                query_embeddings=vectors.tolist(),
                n_results=min(n_results, collection_count),
                where=where,
                include=include,
            )
        except Exception as exc:
            raise RuntimeError(f"ChromaDB similarity query failed: {exc}") from exc

    def close(self) -> None:
        """Release the persistent ChromaDB client and its file handles."""
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
