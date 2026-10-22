"""Structure-aware, deterministic document chunking."""

import hashlib
import importlib
import re
from typing import Any

from .config import PreprocessingConfig


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+")


def _tokenizer(configured_name: str) -> tuple[str, Any]:
    """Select tiktoken when requested, otherwise use a regex word tokenizer."""
    if configured_name == "regex":
        return "regex", lambda text: _TOKEN_PATTERN.findall(text)

    try:
        tiktoken = importlib.import_module("tiktoken")
        encoding = tiktoken.get_encoding(configured_name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return "regex", lambda text: _TOKEN_PATTERN.findall(text)
    return configured_name, encoding.encode


def _split_oversized(text: str, max_size: int, count_tokens: Any) -> list[str]:
    """Split long blocks at sentence boundaries, then at whitespace if needed."""
    sentences = [part.strip() for part in _SENTENCE_BREAK.split(text) if part.strip()]
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        if count_tokens(sentence) <= max_size:
            candidate = f"{current} {sentence}".strip()
            if current and count_tokens(candidate) > max_size:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
            continue

        if current:
            pieces.append(current)
            current = ""

        words = sentence.split()
        word_chunk: list[str] = []
        for word in words:
            candidate = " ".join((*word_chunk, word))
            if word_chunk and count_tokens(candidate) > max_size:
                pieces.append(" ".join(word_chunk))
                word_chunk = [word]
            else:
                word_chunk.append(word)
        if word_chunk:
            pieces.append(" ".join(word_chunk))

    if current:
        pieces.append(current)
    return pieces


def _overlap_text(text: str, overlap_size: int, count_tokens: Any) -> str:
    """Return a trailing whitespace-token overlap within the configured budget."""
    if overlap_size == 0:
        return ""
    words = text.split()
    overlap: list[str] = []
    for word in reversed(words):
        candidate = " ".join([word, *overlap])
        if count_tokens(candidate) > overlap_size:
            break
        overlap.insert(0, word)
    return " ".join(overlap)


def _section_heading(text: str) -> str | None:
    """Return an explicit Markdown heading when one begins the chunk."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line.startswith("#"):
        heading = first_line.lstrip("#").strip()
        return heading or None
    return None


def chunk_document(
    document: dict[str, Any],
    config: PreprocessingConfig | None = None,
) -> dict[str, Any]:
    """Chunk each page independently and preserve its exact page attribution."""
    config = config or PreprocessingConfig()
    tokenizer_name, tokenize = _tokenizer(config.tokenizer)

    def count_tokens(text: str) -> int:
        return len(tokenize(text))

    chunks: list[dict[str, Any]] = []
    for page in document.get("pages", []):
        text = page.get("cleaned_text", page.get("text", ""))
        if not isinstance(text, str) or not text.strip():
            continue

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text.strip())
            if paragraph.strip()
        ]
        units: list[str] = []
        for paragraph in paragraphs:
            if count_tokens(paragraph) <= config.max_chunk_size:
                units.append(paragraph)
            else:
                units.extend(
                    _split_oversized(
                        paragraph, config.max_chunk_size, count_tokens
                    )
                )

        page_chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip()
            if current and count_tokens(candidate) > config.max_chunk_size:
                page_chunks.append(current)
                overlap = _overlap_text(
                    current, config.chunk_overlap, count_tokens
                )
                current = f"{overlap}\n\n{unit}".strip() if overlap else unit
                if count_tokens(current) > config.max_chunk_size:
                    current = unit
            else:
                current = candidate
        if current:
            page_chunks.append(current)

        merged_chunks: list[str] = []
        for page_chunk in page_chunks:
            if (
                merged_chunks
                and count_tokens(page_chunk) < config.min_chunk_size
                and count_tokens(f"{merged_chunks[-1]}\n\n{page_chunk}")
                <= config.max_chunk_size
            ):
                merged_chunks[-1] = f"{merged_chunks[-1]}\n\n{page_chunk}"
            else:
                merged_chunks.append(page_chunk)

        for page_chunk in merged_chunks:
            if not page_chunk.strip():
                continue
            chunk_index = len(chunks) + 1
            document_id = str(document.get("document_id") or "")
            source_identity = str(
                document.get("source_path") or document.get("file_name") or ""
            )
            page_number = page.get("page_number")
            chunk_id_source = (
                f"{document_id}:{source_identity}:{page_number}:"
                f"{chunk_index}:{page_chunk}"
            )
            extraction_method = (
                "ocr"
                if page.get("ocr_output_selected")
                else "native"
            )
            chunks.append(
                {
                    "chunk_id": hashlib.sha256(
                        chunk_id_source.encode("utf-8")
                    ).hexdigest(),
                    "document_id": document.get("document_id"),
                    "file_name": document.get("file_name"),
                    "source_path": document.get("source_path"),
                    "page_number": page_number,
                    "text": page_chunk,
                    "token_count": count_tokens(page_chunk),
                    "tokenizer": tokenizer_name,
                    "extraction_method": extraction_method,
                    "ocr_used": bool(page.get("ocr_output_selected")),
                    "section_heading": _section_heading(page_chunk),
                }
            )

    token_counts = [chunk["token_count"] for chunk in chunks]
    statistics = {
        "chunk_count": len(chunks),
        "tokenizer_requested": config.tokenizer,
        "tokenizer_used": tokenizer_name,
        "minimum_token_count": min(token_counts) if token_counts else 0,
        "maximum_token_count": max(token_counts) if token_counts else 0,
        "average_token_count": (
            round(sum(token_counts) / len(token_counts), 2) if token_counts else 0
        ),
        "chunks_over_maximum": sum(
            count > config.max_chunk_size for count in token_counts
        ),
    }
    return {"chunks": chunks, "statistics": statistics}