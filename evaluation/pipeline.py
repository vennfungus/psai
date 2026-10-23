"""Run a small JSONL retrieval/evidence evaluation without model generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generation import LocalGenerationConfig, LocalTransformerGenerator
from nlp_system.config import NLPConfig
from nlp_system.system import NLPSystem

from .metrics import evaluate_cases


def _read_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed evaluation JSON at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(case, dict) or not isinstance(
                case.get("question"), str
            ):
                raise ValueError(
                    f"Evaluation record at line {line_number} must have a question"
                )
            cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation questions found in {path}")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and evidence against labelled JSONL cases."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/processed/embeddings"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/processed/chroma_db"))
    parser.add_argument("--collection", default="psai_chunks")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--planning", action="store_true")
    parser.add_argument("--local-model-path", type=Path)
    parser.add_argument(
        "--generation-device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if (args.generate or args.planning) and not args.local_model_path:
        parser.error("--generate/--planning requires --local-model-path")
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
    )
    generator = None
    if args.local_model_path:
        generator = LocalTransformerGenerator(
            LocalGenerationConfig(
                model_path=args.local_model_path,
                device=args.generation_device,
                context_length=args.context_length,
                max_new_tokens=args.max_new_tokens,
            )
        )
    system = NLPSystem(config, generator=generator)
    evaluated: list[dict[str, Any]] = []
    try:
        for case in _read_cases(args.dataset):
            response = system.answer(
                case["question"],
                top_k=args.top_k,
                filters=case.get("filters"),
                extract_facts=True,
                generate=args.generate,
                planning=args.planning,
            )
            evaluated.append(
                {
                    **case,
                    "retrieved_chunk_ids": response["retrieval_metadata"][
                        "retrieved_chunk_ids"
                    ],
                    "response": response,
                }
            )
    finally:
        system.close()
    print(
        json.dumps(
            {
                "metrics": evaluate_cases(evaluated, top_k=args.top_k),
                "cases": evaluated,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
