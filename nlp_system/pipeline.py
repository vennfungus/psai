"""Command-line interface for PSAI's local NLP system."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from generation import LocalGenerationConfig, LocalTransformerGenerator

from .config import NLPConfig
from .system import NLPSystem


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
        metavar="FILTER",
        help=(
            "Metadata filter as field=value (PowerShell-friendly) or a JSON "
            "object; repeat to add AND conditions"
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
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
    )
    parser.add_argument(
        "--max-answer-sentences",
        type=int,
        default=5,
        help="Maximum number of evidence sentences in the answer",
    )
    parser.add_argument(
        "--max-answer-characters",
        type=int,
        default=2500,
        help="Maximum answer length in characters",
    )
    parser.add_argument("--dense-weight", type=float, default=0.65)
    parser.add_argument("--lexical-weight", type=float, default=0.35)
    parser.add_argument(
        "--extract-facts",
        action="store_true",
        help="Include rule-extracted project facts with source evidence",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Use a pre-provisioned local Transformer to compose an answer",
    )
    parser.add_argument(
        "--planning",
        action="store_true",
        help="Generate a JSON WBS and validate/schedule it with Python",
    )
    parser.add_argument(
        "--local-model-path",
        type=Path,
        help="Path to an already downloaded local causal language model",
    )
    parser.add_argument(
        "--generation-device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--quantization",
        choices=("none", "dynamic-int8"),
        default="none",
    )
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
            if "=" not in raw_filter:
                parser.error(
                    "--filter must be a JSON object or field=value; "
                    f"could not parse JSON: {exc.msg}"
                )
            field, raw_value = raw_filter.split("=", maxsplit=1)
            field = field.strip()
            raw_value = raw_value.strip()
            if not field or not raw_value:
                parser.error("--filter field=value requires both values")
            try:
                parsed_value = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_value = raw_value
            parsed = {field: parsed_value}
        if not isinstance(parsed, dict) or not parsed:
            parser.error(
                "--filter must be a non-empty JSON object or field=value"
            )
        duplicates = filters.keys() & parsed.keys()
        if duplicates:
            parser.error(
                "Duplicate filter field(s): " + ", ".join(sorted(duplicates))
            )
        filters.update(parsed)
    return filters


def main(argv: list[str] | None = None) -> int:
    """Run one question and print a source-attributed NLP response."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.planning and not args.local_model_path:
        parser.error("--planning requires --local-model-path")
    if args.generate and not args.local_model_path:
        parser.error("--generate requires --local-model-path")
    if args.local_model_path and not (args.generate or args.planning):
        parser.error(
            "--local-model-path requires --generate or --planning"
        )
    config = NLPConfig(
        embeddings_dir=args.embeddings_dir,
        persistence_directory=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        device=args.device,
        top_k=args.top_k,
        max_answer_sentences=args.max_answer_sentences,
        max_answer_characters=args.max_answer_characters,
        dense_weight=args.dense_weight,
        lexical_weight=args.lexical_weight,
    )
    generator = None
    if args.local_model_path:
        generator = LocalTransformerGenerator(
            LocalGenerationConfig(
                model_path=args.local_model_path,
                device=args.generation_device,
                context_length=args.context_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                quantization=args.quantization,
            )
        )
    nlp_system = NLPSystem(config, generator=generator)
    try:
        response = nlp_system.answer(
            args.question,
            filters=_parse_filters(parser, args.filter),
            generate=args.generate,
            planning=args.planning,
            extract_facts=args.extract_facts,
        )
    finally:
        nlp_system.close()
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
