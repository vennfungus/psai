"""CLI and orchestration for generating chunk embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .config import EmbeddingConfig
from .embedder import SentenceTransformerEmbedder


LOGGER = logging.getLogger("embedding")


class Embedder(Protocol):
    """Minimal interface implemented by the model adapter and test doubles."""

    def get_sentence_embedding_dimension(self) -> int: ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Any: ...


def _read_chunks(
    input_path: Path,
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]], int]:
    """Read JSONL records and collect malformed/non-object input errors."""
    records: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    line_count = 0
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line_count += 1
            if not line.strip():
                errors.append(
                    {
                        "line_number": line_number,
                        "error": "Blank JSONL line",
                        "status": "failed",
                    }
                )
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "line_number": line_number,
                        "error": f"Malformed JSON: {exc.msg}",
                        "status": "failed",
                    }
                )
                continue
            if not isinstance(record, dict):
                errors.append(
                    {
                        "line_number": line_number,
                        "error": "Chunk record must be a JSON object",
                        "status": "failed",
                    }
                )
                continue
            records.append((line_number, record))
    return records, errors, line_count


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(file_path: Path, records: list[dict[str, Any]]) -> None:
    with file_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _write_json(file_path: Path, value: dict[str, Any]) -> None:
    with file_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _atomic_save(
    output_dir: Path,
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
    manifest: dict[str, Any],
    config: EmbeddingConfig,
) -> None:
    """Stage output files and publish the manifest last as a commit marker."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / config.embeddings_filename
    metadata_path = output_dir / config.metadata_filename
    manifest_path = output_dir / config.manifest_filename

    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_dir, prefix=".embeddings-", suffix=".tmp", delete=False
        ) as temporary:
            temporary_paths.append(Path(temporary.name))
            np.save(temporary, vectors, allow_pickle=False)
            temporary.flush()
            os.fsync(temporary.fileno())

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=".metadata-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_paths.append(Path(temporary.name))
            for record in metadata:
                temporary.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=".manifest-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_paths.append(Path(temporary.name))
            json.dump(manifest, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())

        # The manifest marks a complete, mutually consistent output set.
        manifest_path.unlink(missing_ok=True)
        os.replace(temporary_paths[0], vector_path)
        os.replace(temporary_paths[1], metadata_path)
        os.replace(temporary_paths[2], manifest_path)
        temporary_paths.clear()
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def run_embedding_pipeline(
    input_path: Path,
    config: EmbeddingConfig | None = None,
    *,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Generate one normalized vector per valid, non-empty chunk record."""
    input_path = Path(input_path)
    config = config or EmbeddingConfig()
    if not input_path.is_file():
        raise FileNotFoundError(f"Chunk JSONL file does not exist: {input_path}")

    records, errors, input_line_count = _read_chunks(input_path)
    eligible: list[tuple[int, dict[str, Any], str]] = []
    skipped_count = 0
    skipped_records: list[dict[str, Any]] = []
    for line_number, record in records:
        text = record.get("text")
        if not isinstance(text, str):
            errors.append(
                {
                    "line_number": line_number,
                    "error": "Chunk text must be a string",
                    "status": "failed",
                    "chunk_id": record.get("chunk_id"),
                }
            )
            continue
        if not text.strip():
            skipped_count += 1
            skipped_records.append(
                {
                    "line_number": line_number,
                    "chunk_id": record.get("chunk_id"),
                    "reason": "Empty or whitespace-only text",
                }
            )
            continue
        eligible.append((line_number, record, text))

    model = embedder or SentenceTransformerEmbedder(
        config.model_name,
        config.device,
    )
    dimension = model.get_sentence_embedding_dimension()
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Embedder must report a positive integer dimension")

    successful_vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for start in range(0, len(eligible), config.batch_size):
        batch = eligible[start : start + config.batch_size]
        try:
            encoded = np.asarray(
                model.encode(
                    [entry[2] for entry in batch],
                    batch_size=config.batch_size,
                    normalize_embeddings=config.normalize_embeddings,
                ),
                dtype=np.float32,
            )
            if encoded.ndim != 2 or encoded.shape != (len(batch), dimension):
                raise ValueError(
                    f"Expected embedding shape {(len(batch), dimension)}, "
                    f"received {encoded.shape}"
                )
            if not np.isfinite(encoded).all():
                raise ValueError("Embedding contains NaN or infinite values")
            if config.normalize_embeddings:
                norms = np.linalg.norm(encoded, axis=1, keepdims=True)
                if np.any(norms <= 0):
                    raise ValueError("Cannot normalize a zero-length embedding")
                encoded = encoded / norms
        except Exception as exc:
            LOGGER.exception("Embedding batch starting at record %d failed", start)
            for line_number, record, _ in batch:
                errors.append(
                    {
                        "line_number": line_number,
                        "error": f"Embedding failed: {exc}",
                        "status": "failed",
                        "chunk_id": record.get("chunk_id"),
                    }
                )
            continue

        for (_, record, text), vector in zip(batch, encoded):
            metadata_record = {
                key: value
                for key, value in record.items()
                if key != "_input_line_number"
            }
            metadata_record.update(
                {
                    "embedding_index": len(metadata),
                    "embedding_model": config.model_name,
                    "embedding_dimension": dimension,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
            successful_vectors.append(vector)
            metadata.append(metadata_record)

    vectors = (
        np.stack(successful_vectors).astype(np.float32, copy=False)
        if successful_vectors
        else np.empty((0, dimension), dtype=np.float32)
    )
    manifest = {
        "input_file": str(input_path.resolve()),
        "input_sha256": _sha256_file(input_path),
        "input_line_count": input_line_count,
        "valid_record_count": len(records),
        "processed_count": len(metadata),
        "skipped_count": skipped_count,
        "skipped_records": skipped_records,
        "failed_count": len(errors),
        "model_name": config.model_name,
        "device": getattr(model, "device", config.device),
        "batch_size": config.batch_size,
        "embedding_dimension": dimension,
        "normalized": config.normalize_embeddings,
        "similarity_metric": (
            "cosine"
            if config.normalize_embeddings
            else "dot_product"
        ),
        "vector_dtype": str(vectors.dtype),
        "vector_shape": list(vectors.shape),
        "outputs": {
            "vectors": config.embeddings_filename,
            "metadata": config.metadata_filename,
            "manifest": config.manifest_filename,
        },
        "metadata_order_matches_vector_rows": True,
        "errors": errors,
    }
    _atomic_save(config.output_dir, vectors, metadata, manifest, config)
    LOGGER.info(
        "Embedding complete: processed=%d skipped=%d failed=%d",
        manifest["processed_count"],
        manifest["skipped_count"],
        manifest["failed_count"],
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local embeddings for PSAI chunks JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="chunks.jsonl path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/embeddings"),
        help="Directory for vector and metadata output",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name or local model path",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
        help="Inference device; CPU is the default",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize vectors",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the embedding pipeline CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    config = EmbeddingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        normalize_embeddings=not args.no_normalize,
    )
    run_embedding_pipeline(args.input, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
