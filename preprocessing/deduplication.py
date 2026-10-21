"""Identify exact duplicates and optional near-duplicate text candidates."""

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def find_file_duplicates(file_paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Group source paths with identical SHA-256 file contents."""
    groups: dict[str, list[str]] = {}
    for file_path in file_paths:
        path = Path(file_path)
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        groups.setdefault(digest.hexdigest(), []).append(str(path))
    return [
        {"sha256": digest, "source_paths": sorted(paths)}
        for digest, paths in sorted(groups.items())
        if len(paths) > 1
    ]


def find_text_duplicates(
    records: Iterable[dict[str, Any]],
    *,
    near_duplicate_threshold: float | None = None,
) -> dict[str, Any]:
    """Find normalized exact duplicates and optional similarity candidates.

    Near duplicates use ``difflib.SequenceMatcher`` and are only review
    candidates; they are never treated as definitive duplicate records.
    """
    if near_duplicate_threshold is not None and not (
        0.0 <= near_duplicate_threshold <= 1.0
    ):
        raise ValueError("near_duplicate_threshold must be between 0 and 1")

    exact_groups: dict[str, list[dict[str, Any]]] = {}
    normalized_records: list[tuple[str, str, str]] = []
    for index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str):
            continue
        normalized = _normalized_text(text)
        if not normalized:
            continue
        identifier = str(
            record.get("document_id")
            or record.get("chunk_id")
            or record.get("record_id")
            or index
        )
        exact_groups.setdefault(normalized, []).append(
            {
                "record_id": identifier,
                "source_path": record.get("source_path"),
                "file_name": record.get("file_name"),
            }
        )
        normalized_records.append((identifier, normalized, text))

    exact_duplicates = [
        {"records": references}
        for references in exact_groups.values()
        if len(references) > 1
    ]

    near_candidates: list[dict[str, Any]] = []
    if near_duplicate_threshold is not None:
        for index, (first_id, first_text, _) in enumerate(normalized_records):
            for second_id, second_text, _ in normalized_records[index + 1 :]:
                if first_text == second_text:
                    continue
                similarity = SequenceMatcher(
                    None, first_text, second_text, autojunk=False
                ).ratio()
                if similarity >= near_duplicate_threshold:
                    near_candidates.append(
                        {
                            "first_record_id": first_id,
                            "second_record_id": second_id,
                            "similarity": round(similarity, 4),
                            "review_required": True,
                        }
                    )

    return {
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_candidates": near_candidates,
    }