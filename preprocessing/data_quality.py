"""Non-destructive checks for preprocessing and annotation records."""

from collections import Counter
from typing import Any, Iterable


def check_data_quality(
    records: Iterable[dict[str, Any]],
    *,
    required_metadata: tuple[str, ...] = (),
    known_task_ids: set[str] | None = None,
    enforce_acyclic_dependencies: bool = False,
    max_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Return structured findings without changing or removing any records."""
    record_list = list(records)
    issues: list[dict[str, Any]] = []
    raw_identifiers = [
        (
            record.get("record_id")
            or record.get("chunk_id")
            or record.get("document_id")
            or record.get("task_id")
        )
        if isinstance(record, dict)
        else None
        for record in record_list
    ]
    identifiers = [
        identifier
        if isinstance(identifier, (str, int, float))
        else None
        for identifier in raw_identifiers
    ]
    counts = Counter(identifier for identifier in identifiers if identifier)

    def report(
        severity: str,
        record_id: Any,
        explanation: str,
        page_number: Any = None,
    ) -> None:
        issue = {
            "severity": severity,
            "record_id": record_id,
            "explanation": explanation,
        }
        if page_number is not None:
            issue["page_number"] = page_number
        issues.append(issue)

    for record, record_id in zip(record_list, identifiers):
        if not isinstance(record, dict):
            report("error", record_id, "Record must be an object")
            continue
        if record_id and counts[record_id] > 1:
            report("warning", record_id, "Duplicate record identifier")
        for field in required_metadata:
            if field not in record or record[field] is None:
                report("warning", record_id, f"Missing required metadata: {field}")
        for field in ("document_id", "chunk_id", "task_id"):
            if field in record and not isinstance(record[field], str):
                report("error", record_id, f"{field} must be a string")

        if "text" not in record and "pages" not in record:
            report("warning", record_id, "Missing text")
        elif "text" in record and not isinstance(record["text"], str):
            report("error", record_id, "text must be a string")
        elif "text" in record and not record["text"].strip():
            report("warning", record_id, "Text is empty")

        page_number = record.get("page_number")
        page_count = record.get("page_count")
        if page_number is not None:
            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
            ):
                report("error", record_id, "page_number must be a positive integer")
            elif isinstance(page_count, int) and page_number > page_count:
                report(
                    "error",
                    record_id,
                    f"page_number exceeds page_count ({page_count})",
                    page_number,
                )

        if "start" in record or "end" in record:
            start, end = record.get("start"), record.get("end")
            text_length = len(record.get("text", "")) if isinstance(
                record.get("text", ""), str
            ) else 0
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or end > text_length
            ):
                report("error", record_id, "Invalid annotation span")

        if max_chunk_size is not None and "token_count" in record:
            token_count = record["token_count"]
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count < 0
            ):
                report("error", record_id, "token_count must be a non-negative integer")
            elif token_count > max_chunk_size:
                report(
                    "warning",
                    record_id,
                    f"Chunk exceeds configured maximum ({token_count} > {max_chunk_size})",
                )

        if record.get("status") == "error" or record.get("extraction_error"):
            report("error", record_id, record.get("error", "Extraction failed"))

        pages = record.get("pages", [])
        if not isinstance(pages, list):
            report("error", record_id, "pages must be a list")
            pages = []
        for page in pages:
            if not isinstance(page, dict):
                report("error", record_id, "Page record must be an object")
                continue
            if (
                not isinstance(page.get("page_number"), int)
                or isinstance(page.get("page_number"), bool)
                or page["page_number"] < 1
            ):
                report("error", record_id, "Invalid page reference")
            if page.get("extraction_error"):
                report(
                    "warning",
                    record_id,
                    f"Page extraction failed: {page['extraction_error']}",
                    page.get("page_number"),
                )
            if page.get("ocr_error"):
                report(
                    "warning",
                    record_id,
                    f"OCR failed: {page['ocr_error']}",
                    page.get("page_number"),
                )

    if known_task_ids is not None:
        dependency_graph: dict[str, set[str]] = {}
        task_records = {
            record["task_id"]: record
            for record in record_list
            if isinstance(record, dict) and isinstance(record.get("task_id"), str)
        }
        for task_id, record in task_records.items():
            dependencies = record.get("depends_on", [])
            if isinstance(dependencies, str):
                dependencies = [dependencies]
            if not isinstance(dependencies, list):
                report("error", task_id, "depends_on must be a string or list")
                continue
            dependency_graph[task_id] = set()
            for dependency in dependencies:
                if not isinstance(dependency, str):
                    report("error", task_id, "Dependency reference must be a string")
                elif dependency not in known_task_ids:
                    report("warning", task_id, f"Unknown dependency reference: {dependency}")
                else:
                    dependency_graph[task_id].add(dependency)

        if enforce_acyclic_dependencies:
            visited: set[str] = set()
            active: set[str] = set()

            def visit(task_id: str) -> bool:
                if task_id in active:
                    return True
                if task_id in visited:
                    return False
                active.add(task_id)
                has_cycle = any(
                    visit(dependency)
                    for dependency in dependency_graph.get(task_id, set())
                    if dependency in dependency_graph
                )
                active.remove(task_id)
                visited.add(task_id)
                return has_cycle

            for task_id in dependency_graph:
                if visit(task_id):
                    report("error", task_id, "Dependency cycle detected")
                    break

    summary = Counter(issue["severity"] for issue in issues)
    return {
        "record_count": len(record_list),
        "issue_count": len(issues),
        "severity_counts": dict(summary),
        "issues": issues,
    }