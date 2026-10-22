"""Command-line interface for the persistent PSAI ChromaDB collection."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .config import VectorDBConfig
from .store import ChromaVectorStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest or query existing PSAI embeddings in ChromaDB."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/processed/embeddings"),
        help="Directory containing embeddings.npy, metadata JSONL, and manifest",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/processed/chroma_db"),
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--collection",
        default="psai_chunks",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of vectors per ChromaDB upsert batch",
    )
    parser.add_argument(
        "--distance-metric",
        choices=("cosine", "l2", "ip"),
        default=None,
        help="Require this metric; incompatible manifest metrics are rejected",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="Upsert existing embeddings")
    commands.add_parser("inspect", help="Show collection count and a sample")
    query_parser = commands.add_parser(
        "query", help="Find nearest chunks for a JSON embedding vector"
    )
    query_parser.add_argument(
        "--vector",
        required=True,
        help="One query vector as a JSON array, e.g. '[0.1, 0.2, ...]'",
    )
    query_parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run an ingestion, inspection, or vector-query operation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "query" and args.top_k < 1:
        parser.error("--top-k must be >= 1")

    config = VectorDBConfig(
        embeddings_dir=args.embeddings_dir,
        persistence_directory=args.persist_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
        distance_metric=args.distance_metric,
    )
    store = ChromaVectorStore(config)

    if args.command == "ingest":
        result: dict[str, Any] = store.ingest()
    elif args.command == "inspect":
        result = store.inspect()
    else:
        try:
            vector = json.loads(args.vector)
        except json.JSONDecodeError as exc:
            parser.error(f"--vector must be a JSON numeric array: {exc.msg}")
        if not isinstance(vector, list):
            parser.error("--vector must be a JSON numeric array")
        result = store.query(vector, n_results=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
