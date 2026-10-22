"""Command-line interface for semantic retrieval from PSAI's ChromaDB."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .config import RetrievalConfig
from .retriever import Retriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve relevant PSAI chunks for a natural-language query."
    )
    parser.add_argument("--query", required=True, help="Natural-language query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="JSON",
        help=(
            "Metadata equality filter as a JSON object; repeat to add AND "
            'conditions, e.g. \'{"file_name":"PSAI001.pdf"}\''
        ),
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/processed/embeddings"),
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/processed/chroma_db"),
    )
    parser.add_argument("--collection", default="psai_chunks")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Must match the model that produced the stored embeddings",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
    )
    return parser


def _parse_filters(
    parser: argparse.ArgumentParser,
    filter_values: list[str],
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for raw_filter in filter_values:
        try:
            value = json.loads(raw_filter)
        except json.JSONDecodeError as exc:
            parser.error(f"--filter must be a JSON object: {exc.msg}")
        if not isinstance(value, dict) or not value:
            parser.error("--filter must be a non-empty JSON object")
        overlapping_keys = filters.keys() & value.keys()
        if overlapping_keys:
            key_list = ", ".join(sorted(overlapping_keys))
            parser.error(f"Duplicate filter field(s): {key_list}")
        filters.update(value)
    return filters


def main(argv: list[str] | None = None) -> int:
    """Run one semantic retrieval query and print JSON results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    config = RetrievalConfig(
        embeddings_dir=args.embeddings_dir,
        persistence_directory=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        device=args.device,
        top_k=args.top_k,
    )
    retriever = Retriever(config)
    try:
        result = retriever.retrieve(
            args.query,
            filters=_parse_filters(parser, args.filter),
        )
    finally:
        retriever.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
