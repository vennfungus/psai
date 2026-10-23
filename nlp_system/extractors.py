"""Conservative rule-based extraction of project facts from source passages."""

from __future__ import annotations

import re
from typing import Any, Iterable


_TASK_ID = r"(?:[A-Z]{1,5}[- ]?)?\d+(?:[.-]\d+)*"
_TASK_HEADING = re.compile(
    r"\b(?:task|task\s+id|nhiệm\s+vụ|công\s+việc)\s+"
    rf"(?P<id>{_TASK_ID})\s*[-:]\s*(?P<title>[^.;\n]+)",
    re.IGNORECASE,
)
_MILESTONE = re.compile(
    r"\b(?:milestone|mốc)\s*(?P<id>\d+)?\s*[-:]\s*(?P<name>[^.;\n]+)",
    re.IGNORECASE,
)
_DURATION = re.compile(
    r"\b(?P<amount>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>calendar\s+days?|working\s+days?|days?|weeks?|months?|"
    r"hours?|ngày|tuần|tháng|giờ)\b",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"\b(?:deadline|due(?:\s+date)?|by|before|on|ngày\s+hoàn\s+thành|"
    r"hạn\s+chót|ngày)\s*[:,-]?\s*"
    r"(?P<date>\d{4}-\d{1,2}-\d{1,2}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?))\b",
    re.IGNORECASE,
)
_COST = re.compile(
    r"(?P<amount>(?:[$€£₫]\s*)?\d[\d,]*(?:\.\d{1,2})?)\s*"
    r"(?P<currency>USD|EUR|GBP|VND|VNĐ|dollars?|euros?|pounds?)?"
    r"|(?P<currency_prefix>USD|EUR|GBP|VND|VNĐ)\s*"
    r"(?P<amount_prefix>\d[\d,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_BUDGET_AMOUNT = re.compile(
    r"\b(?:budget|cost|costs|ngân\s+sách|chi\s+phí)\b"
    r"[^\d]{0,20}(?P<amount>\d[\d,]*(?:\.\d{1,2})?)\s*"
    r"(?P<unit>thousand|million|billion|k|m|b|nghìn|triệu|tỷ)?",
    re.IGNORECASE,
)
_DEPENDENCY = re.compile(
    rf"\b(?P<task>(?:task|task\s+id)\s*{_TASK_ID})\s+"
    r"(?:depends\s+on|is\s+dependent\s+on|requires|follows)\s+"
    rf"(?P<prerequisite>(?:task|task\s+id)\s*{_TASK_ID})"
    rf"|(?P<task_vi>(?:nhiệm\s+vụ|công\s+việc)\s*{_TASK_ID})\s+"
    r"phụ\s+thuộc\s+(?:vào|sau)\s+"
    rf"(?P<prerequisite_vi>(?:nhiệm\s+vụ|công\s+việc)\s*{_TASK_ID})",
    re.IGNORECASE,
)
_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "requirement",
        re.compile(
            r"\b(?:must|shall|required to|needs to|should|phải|cần|bắt buộc)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "constraint",
        re.compile(
            r"\b(?:constraint|limited to|cannot|must not|only if|"
            r"ràng buộc|giới hạn|không được|chỉ khi)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "risk",
        re.compile(
            r"\b(?:risk|may fail|might delay|uncertain|nguy cơ|rủi ro|"
            r"có thể chậm|không chắc chắn)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "assumption",
        re.compile(
            r"\b(?:assume|assumption|assuming|giả định|với giả thiết)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "acceptance_criterion",
        re.compile(
            r"\b(?:acceptance criteria|definition of done|must pass|"
            r"tiêu chí chấp nhận|điều kiện nghiệm thu)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "goal",
        re.compile(
            r"\b(?:goal|objective|aim|purpose|mục tiêu|nhằm|để xây dựng)\b",
            re.IGNORECASE,
        ),
    ),
)


def _source_fields(source: dict[str, Any]) -> dict[str, Any]:
    metadata = source.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    values = {
        "source_ref": source.get("source_ref"),
        "document_id": source.get("document_id") or metadata.get("document_id"),
        "chunk_id": source.get("chunk_id") or metadata.get("chunk_id"),
        "file_name": source.get("source_filename") or metadata.get("file_name"),
        "page_number": source.get("page_number") or metadata.get("page_number"),
        "section_heading": source.get("section_heading")
        or metadata.get("section_heading"),
    }
    return {key: value for key, value in values.items() if value is not None}


def extract_project_facts(
    passages: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Extract explicit patterns and retain their exact evidence and source."""
    facts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    available_fields = {
        "goals": False,
        "requirements": False,
        "tasks": False,
        "milestones": False,
        "deadlines": False,
        "durations": False,
        "costs": False,
        "dependencies": False,
        "constraints": False,
        "acceptance_criteria": False,
        "risks": False,
        "assumptions": False,
    }

    for passage in passages:
        text = passage.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        source = _source_fields(passage)
        for line in text.splitlines() or [text]:
            for match in _TASK_HEADING.finditer(line):
                _add_fact(
                    facts,
                    seen,
                    available_fields,
                    "task",
                    {
                        "id": match.group("id"),
                        "title": match.group("title").strip(),
                    },
                    match.group(0),
                    source,
                    "tasks",
                )
            for match in _MILESTONE.finditer(line):
                _add_fact(
                    facts,
                    seen,
                    available_fields,
                    "milestone",
                    {
                        "id": match.group("id"),
                        "name": match.group("name").strip(),
                    },
                    match.group(0),
                    source,
                    "milestones",
                )

        for match in _DURATION.finditer(text):
            _add_fact(
                facts,
                seen,
                available_fields,
                "duration",
                {
                    "amount": match.group("amount").replace(",", "."),
                    "unit": match.group("unit").casefold(),
                },
                match.group(0),
                source,
                "durations",
            )
        for match in _DATE.finditer(text):
            _add_fact(
                facts,
                seen,
                available_fields,
                "date",
                match.group("date"),
                match.group(0),
                source,
                "deadlines",
            )
        cost_spans: list[tuple[int, int]] = []
        for match in _COST.finditer(text):
            amount = match.group("amount") or match.group("amount_prefix")
            currency = (
                match.group("currency_prefix")
                or match.group("currency")
                or next(
                    (
                        symbol
                        for symbol in ("$", "€", "£", "₫")
                        if symbol in match.group(0)
                    ),
                    None,
                )
            )
            if amount and currency:
                cost_spans.append(match.span())
                _add_fact(
                    facts,
                    seen,
                    available_fields,
                    "cost",
                    {
                        "amount": amount.lstrip("$€£₫").strip().replace(",", ""),
                        "currency": currency,
                    },
                    match.group(0),
                    source,
                    "costs",
                )
        for match in _BUDGET_AMOUNT.finditer(text):
            if any(
                start < match.end() and match.start() < end
                for start, end in cost_spans
            ):
                continue
            _add_fact(
                facts,
                seen,
                available_fields,
                "cost",
                {
                    "amount": match.group("amount").replace(",", ""),
                    "currency": None,
                    "unit": match.group("unit"),
                },
                match.group(0),
                source,
                "costs",
            )
        for match in _DEPENDENCY.finditer(text):
            task = match.group("task") or match.group("task_vi")
            prerequisite = match.group("prerequisite") or match.group(
                "prerequisite_vi"
            )
            _add_fact(
                facts,
                seen,
                available_fields,
                "dependency",
                {
                    "task": re.sub(
                        r"^(?:task(?:\s+id)?|nhiệm\s+vụ|công\s+việc)\s*",
                        "",
                        task,
                        flags=re.IGNORECASE,
                    ),
                    "prerequisite": re.sub(
                        r"^(?:task(?:\s+id)?|nhiệm\s+vụ|công\s+việc)\s*",
                        "",
                        prerequisite,
                        flags=re.IGNORECASE,
                    ),
                },
                match.group(0),
                source,
                "dependencies",
            )

        for category, pattern in _FACT_PATTERNS:
            for match in pattern.finditer(text):
                sentence = _containing_sentence(text, match.start(), match.end())
                field = (
                    "acceptance_criteria"
                    if category == "acceptance_criterion"
                    else f"{category}s" if category != "goal" else "goals"
                )
                _add_fact(
                    facts,
                    seen,
                    available_fields,
                    category,
                    sentence,
                    sentence,
                    source,
                    field,
                )

    unresolved = [
        field
        for field, found in available_fields.items()
        if not found
    ]
    return {
        "facts": facts,
        "field_status": {
            field: "found" if found else "unknown_requires_review"
            for field, found in available_fields.items()
        },
        "unresolved_fields": unresolved,
    }


def _add_fact(
    facts: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    available_fields: dict[str, bool],
    kind: str,
    value: Any,
    evidence: str,
    source: dict[str, Any],
    field: str,
) -> None:
    identity = (kind, evidence.casefold(), str(source.get("chunk_id", "")))
    if identity in seen:
        return
    seen.add(identity)
    available_fields[field] = True
    fact = {
        "type": kind,
        "value": value,
        "evidence": evidence,
        "source": source,
    }
    if re.search(
        r"\b(?:not|never|no|without|cannot|must\s+not|"
        r"không|chưa|không\s+được|ngoại\s+trừ)\b",
        evidence,
        re.IGNORECASE,
    ):
        fact["polarity"] = "negative_or_exception"
    facts.append(fact)


def _containing_sentence(text: str, start: int, end: int) -> str:
    left = max(
        (text.rfind(mark, 0, start) for mark in (".", "!", "?", "\n")),
        default=-1,
    )
    right_candidates = [
        position
        for mark in (".", "!", "?", "\n")
        if (position := text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return " ".join(text[left + 1 : right].split())
