"""Local sentence extraction combining dense and classical lexical relevance."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence


_WORD_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")
_BULLET_PATTERN = re.compile(r"^\s*(?:[•●▪◦○◉*-]|\d+[.)])(?:\s+|$)")
_TASK_HEADING_PATTERN = re.compile(r"^\s*Task\s+\d+\s*[-:.]", re.IGNORECASE)
_PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "how",
    "các", "cho", "của", "đã", "đang", "đâu", "được", "gì", "khi",
    "là", "nào", "những", "ở", "sẽ", "trong", "và", "về", "với",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in _WORD_PATTERN.findall(text.casefold())
        if token not in _STOP_WORDS
    ]


def _sentences(text: str) -> list[str]:
    """Join hard-wrapped PDF lines and retain paragraphs and list boundaries."""
    sentences: list[str] = []
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    for paragraph in re.split(r"\n[ \t]*\n", normalized_text.strip()):
        lines = [
            line.strip()
            for line in paragraph.splitlines()
            if line.strip() and not _PAGE_NUMBER_PATTERN.fullmatch(line)
        ]
        if not lines:
            continue

        blocks: list[str] = []
        current: list[str] = []
        pending_bullet = False
        for line in lines:
            if _TASK_HEADING_PATTERN.match(line):
                if current:
                    blocks.append(" ".join(current))
                current = [line]
                pending_bullet = False
                continue
            bullet_match = _BULLET_PATTERN.match(line)
            if bullet_match:
                if current:
                    blocks.append(_join_wrapped_lines(current))
                bullet_text = line[bullet_match.end():].strip()
                current = [bullet_text] if bullet_text else []
                pending_bullet = not bullet_text
            elif pending_bullet:
                current = [line]
                pending_bullet = False
            elif current:
                current.append(line)
            else:
                current.append(line)
        if current:
            blocks.append(_join_wrapped_lines(current))

        for block in blocks:
            sentences.extend(
                part.strip()
                for part in _SENTENCE_PATTERN.split(block)
                if part.strip()
            )
    return sentences


def _join_wrapped_lines(lines: list[str]) -> str:
    """Undo line-wrap hyphenation before joining lines into one paragraph."""
    joined = " ".join(lines)
    return re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", joined)


class HybridAnswerer:
    """Extract answer sentences using dense retrieval plus TF-IDF-like overlap.

    The Transformer embedding is used by the existing retrieval layer to rank
    chunks. This class adds a lightweight classical lexical signal for
    sentence selection; it does not generate or paraphrase text.
    """

    def build_answer(
        self,
        question: str,
        results: Sequence[dict[str, Any]],
        *,
        dense_weight: float,
        lexical_weight: float,
        max_sentences: int,
        max_characters: int,
    ) -> dict[str, Any]:
        """Select evidence sentences and return their existing source metadata."""
        query_terms = set(_tokens(question))
        candidates: list[dict[str, Any]] = []
        document_frequency: Counter[str] = Counter()

        for result_index, result in enumerate(results, start=1):
            text = result.get("text")
            if not isinstance(text, str):
                continue
            metadata = result.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            chunk_id = result.get("chunk_id") or metadata.get("chunk_id")
            source_ref = f"S{result_index}"
            rank = result.get("rank", result_index)
            dense_score = result.get("similarity")
            if not isinstance(dense_score, (int, float)) or not math.isfinite(
                dense_score
            ):
                dense_score = 1.0 / math.log2(max(int(rank), 1) + 1)
            else:
                dense_score = min(1.0, max(0.0, float(dense_score)))

            for sentence in _sentences(text):
                terms = _tokens(sentence)
                term_set = set(terms)
                document_frequency.update(term_set)
                candidates.append(
                    {
                        "sentence": sentence,
                        "terms": term_set,
                        "dense_score": dense_score,
                        "rank": rank,
                        "result_index": result_index,
                        "result": result,
                        "source_ref": source_ref,
                        "chunk_id": chunk_id,
                    }
                )

        total_candidates = len(candidates)
        query_idf = {
            term: math.log(
                (total_candidates + 1) / (document_frequency[term] + 1)
            )
            + 1.0
            for term in query_terms
        }
        total_query_weight = sum(query_idf.values())

        for candidate in candidates:
            if total_query_weight:
                candidate["lexical_score"] = (
                    sum(
                        query_idf[term]
                        for term in candidate["terms"]
                        if term in query_idf
                    )
                    / total_query_weight
                )
            else:
                candidate["lexical_score"] = 0.0
            candidate["combined_score"] = (
                dense_weight * candidate["dense_score"]
                + lexical_weight * candidate["lexical_score"]
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate["combined_score"],
                candidate["rank"],
                candidate["result_index"],
            )
        )
        selected: list[dict[str, Any]] = []
        selected_sentences: set[str] = set()
        answer_characters = 0
        for candidate in candidates:
            sentence_key = candidate["sentence"].casefold()
            if sentence_key in selected_sentences:
                continue
            rendered_sentence = (
                f"- {candidate['sentence']} [{candidate['source_ref']}]"
            )
            addition_length = len(rendered_sentence) + (
                1 if selected else 0
            )
            if answer_characters + addition_length > max_characters:
                continue
            selected.append(candidate)
            selected_sentences.add(sentence_key)
            answer_characters += addition_length
            if len(selected) >= max_sentences:
                break

        source_refs: list[dict[str, Any]] = []
        seen_source_refs: set[str] = set()
        for candidate in selected:
            source_ref = candidate["source_ref"]
            if source_ref in seen_source_refs:
                continue
            seen_source_refs.add(source_ref)
            result = candidate["result"]
            metadata = result.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            source = {
                "source_ref": source_ref,
                "rank": result.get("rank", candidate["result_index"]),
                "chunk_id": candidate["chunk_id"],
                "document_id": result.get("document_id")
                or metadata.get("document_id"),
                "file_name": result.get("source_filename")
                or metadata.get("file_name"),
                "page_number": result.get("page_number")
                or metadata.get("page_number"),
                "page_range": result.get("page_range")
                or metadata.get("page_range"),
                "section_heading": result.get("section_heading")
                or metadata.get("section_heading"),
                "distance": result.get("distance"),
                "similarity": result.get("similarity"),
            }
            source_refs.append(
                {key: value for key, value in source.items() if value is not None}
            )

        answer_lines = [
            f"- {candidate['sentence']} [{candidate['source_ref']}]"
            for candidate in selected
        ]
        warnings: list[str] = []
        if not candidates:
            warnings.append("Retrieved chunks contained no usable sentences.")
        elif not selected:
            warnings.append(
                "No sentence fit within max_answer_characters."
            )
        elif not query_terms:
            warnings.append(
                "The query had no indexable lexical terms; sentence selection "
                "used dense retrieval ranking only."
            )
        if len(selected) < min(max_sentences, len(candidates)):
            if candidates and answer_characters >= max_characters:
                warnings.append(
                    "Answer was limited by max_answer_characters."
                )

        return {
            "answer": "\n".join(answer_lines),
            "sources": source_refs,
            "selected_sentence_count": len(selected),
            "candidate_sentence_count": total_candidates,
            "warnings": warnings,
        }
