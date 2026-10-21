"""Load and normalize JSON or JSONL annotation records."""

import json
import re
from pathlib import Path
from typing import Any, Mapping


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        lines = [" ".join(line.split()) for line in value.split("\n")]
        return "\n".join(lines).strip()
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def load_annotations(
    file_path: Path,
    *,
    required_fields: tuple[str, ...] = (),
    schema: Mapping[str, type | tuple[type, ...]] | None = None,
) -> dict[str, Any]:
    """Load annotations, returning valid records and line-aware errors.

    ``schema`` maps field names to expected Python types. Unknown fields are
    retained, and malformed records are reported rather than silently fixed.
    """
    file_path = Path(file_path)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def validate_record(record: Any, line_number: int | None) -> None:
        location = {"line_number": line_number} if line_number is not None else {}
        if not isinstance(record, dict):
            errors.append(
                {
                    **location,
                    "error": "Annotation record must be a JSON object",
                }
            )
            return
        missing_fields = [field for field in required_fields if field not in record]
        type_errors = [
            field
            for field, expected_type in (schema or {}).items()
            if field in record and not isinstance(record[field], expected_type)
        ]
        if missing_fields or type_errors:
            messages = []
            if missing_fields:
                messages.append(f"missing required fields: {', '.join(missing_fields)}")
            if type_errors:
                messages.append(f"invalid field types: {', '.join(type_errors)}")
            errors.append({**location, "error": "; ".join(messages)})
            return
        records.append(_normalize(record))

    try:
        if file_path.suffix.lower() == ".jsonl":
            with file_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        validate_record(json.loads(line), line_number)
                    except json.JSONDecodeError as exc:
                        errors.append(
                            {
                                "line_number": line_number,
                                "error": f"Malformed JSON: {exc.msg}",
                            }
                        )
        elif file_path.suffix.lower() == ".json":
            with file_path.open("r", encoding="utf-8") as file:
                content = json.load(file)
            if isinstance(content, list):
                for index, record in enumerate(content, start=1):
                    validate_record(record, index)
            else:
                validate_record(content, 1)
    except OSError as exc:
        errors.append({"line_number": None, "error": f"Unable to read file: {exc}"})
    except json.JSONDecodeError as exc:
        errors.append(
            {
                "line_number": exc.lineno,
                "error": f"Malformed JSON: {exc.msg}",
            }
        )

    if file_path.suffix.lower() not in {".json", ".jsonl"}:
        errors.append(
            {
                "line_number": None,
                "error": "Annotation file must have a .json or .jsonl extension",
            }
        )
    return {"records": records, "errors": errors}