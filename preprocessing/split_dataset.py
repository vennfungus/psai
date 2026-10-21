"""Reproducibly split records by a shared document or project group."""

import random
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping


def split_records(
    records: Iterable[dict[str, Any]],
    *,
    group_key: str = "document_id",
    ratios: Mapping[str, float] | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Split whole groups, retaining every input record and preventing leakage."""
    split_ratios = dict(
        ratios or {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    if set(split_ratios) != {"train", "validation", "test"}:
        raise ValueError("ratios must contain train, validation, and test")
    if any(
        not isinstance(ratio, (int, float))
        or isinstance(ratio, bool)
        or not math.isfinite(ratio)
        or ratio < 0
        for ratio in split_ratios.values()
    ):
        raise ValueError("split ratios must be non-negative")
    if abs(sum(split_ratios.values()) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1")
    if not group_key:
        raise ValueError("group_key must not be empty")

    grouped_records: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {index} must be an object")
        group_id = record.get(group_key)
        if group_id is None or group_id == "":
            raise ValueError(
                f"Record at index {index} is missing grouping key {group_key!r}"
            )
        try:
            grouped_records[group_id].append(record)
        except TypeError as exc:
            raise ValueError(
                f"Grouping key {group_key!r} must contain a hashable value"
            ) from exc

    total_records = sum(len(group) for group in grouped_records.values())
    targets = {
        name: total_records * ratio for name, ratio in split_ratios.items()
    }
    rng = random.Random(random_seed)
    groups = list(grouped_records.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    splits: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("train", "validation", "test")
    }
    split_groups: dict[str, list[Any]] = {
        name: [] for name in ("train", "validation", "test")
    }

    for group_id, group in groups:
        candidates = [
            name for name, ratio in split_ratios.items() if ratio > 0
        ]
        if not candidates:
            raise ValueError("At least one split ratio must be greater than zero")
        rng.shuffle(candidates)

        def projected_error(candidate: str) -> float:
            return sum(
                abs(
                    len(splits[name])
                    + (len(group) if name == candidate else 0)
                    - targets[name]
                )
                for name in splits
            )

        chosen_split = min(candidates, key=projected_error)
        splits[chosen_split].extend(group)
        split_groups[chosen_split].append(group_id)

    summary = {
        "group_key": group_key,
        "random_seed": random_seed,
        "ratios": split_ratios,
        "record_counts": {name: len(items) for name, items in splits.items()},
        "group_counts": {name: len(items) for name, items in split_groups.items()},
        "groups": split_groups,
        "total_records": total_records,
        "total_groups": len(grouped_records),
    }
    return {"splits": splits, "summary": summary}