"""Conservative text normalization for extracted PDF content."""

import re
import unicodedata
from typing import Any


_HYPHENATED_LINE_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_SPACES = re.compile(r"[^\S\n]+")
_PARAGRAPH_BREAKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize text while retaining paragraph boundaries and content."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character
        for character in normalized
        if character == "\n"
        or character == "\t"
        or unicodedata.category(character) != "Cc"
    )
    normalized = _HYPHENATED_LINE_BREAK.sub("", normalized)

    lines = [_SPACES.sub(" ", line).strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = _PARAGRAPH_BREAKS.sub("\n\n", normalized)
    return normalized.strip()


def clean_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with cleaned page text and original text retained."""
    cleaned_document = dict(document)
    cleaned_pages: list[dict[str, Any]] = []
    for page in document.get("pages", []):
        cleaned_page = dict(page)
        raw_text = page.get("text", page.get("native_text", ""))
        cleaned_text = clean_text(raw_text)
        cleaned_page.setdefault("raw_text", raw_text)
        cleaned_page["cleaned_text"] = cleaned_text
        cleaned_page["text"] = cleaned_text
        cleaned_pages.append(cleaned_page)
    cleaned_document["pages"] = cleaned_pages
    return cleaned_document