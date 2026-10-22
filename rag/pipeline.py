"""Command-line interface for PSAI grounded question answering."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from .config import RAGConfig
from .generator import RAGGenerator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer a question using retrieved PSAI project documents."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="JSON",
        help="Metadata equality filter JSON; repeat to add AND conditions",
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
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PSAI_LLM_MODEL", "gpt-4o-mini"),
        help="Chat model; defaults to PSAI_LLM_MODEL or gpt-4o-mini",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PSAI_LLM_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the provider API key",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--context-budget-tokens", type=int, default=6000)
    return parser


def _parse_filters(
    parser: argparse.ArgumentParser,
    raw_filters: list[str],
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for raw_filter in raw_filters:
        try:
            parsed = json.loads(raw_filter)
        except json.JSONDecodeError as exc:
            parser.error(f"--filter must be a JSON object: {exc.msg}")
        if not isinstance(parsed, dict) or not parsed:
            parser.error("--filter must be a non-empty JSON object")
        duplicates = filters.keys() & parsed.keys()
        if duplicates:
            parser.error(
                "Duplicate filter field(s): " + ", ".join(sorted(duplicates))
            )
        filters.update(parsed)
    return filters


def main(argv: list[str] | None = None) -> int:
    """Run one RAG query and print a source-attributed JSON response."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    config = RAGConfig(
        embeddings_dir=args.embeddings_dir,
        persistence_directory=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        device=args.device,
        top_k=args.top_k,
        context_budget_tokens=args.context_budget_tokens,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout,
    )
    generator = RAGGenerator(config)
    try:
        response = generator.answer(
            args.question,
            filters=_parse_filters(parser, args.filter),
        )
    finally:
        generator.close()
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
