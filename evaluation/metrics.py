"""Metrics for labelled retrieval, evidence, extraction and plans."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import re
from typing import Any


def evaluate_cases(
    cases: Iterable[Mapping[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Compute aggregate metrics from saved case input and output records."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    case_list = list(cases)
    if not case_list:
        raise ValueError("At least one evaluation case is required")

    precision_values: list[float] = []
    recall_values: list[float] = []
    coverage_values: list[float] = []
    citation_validity: list[float] = []
    fact_true_positives = 0
    fact_false_positives = 0
    fact_false_negatives = 0
    invalid_plan_count = 0
    latency_values: list[float] = []
    throughput_values: list[float] = []

    for index, case in enumerate(case_list):
        response = case.get("response")
        if not isinstance(response, Mapping):
            raise ValueError(f"Evaluation case {index} is missing response")
        sources = response.get("sources", [])
        if not isinstance(sources, list):
            raise ValueError(f"Evaluation case {index} sources must be a list")
        actual_ids = {
            source.get("chunk_id")
            for source in sources
            if isinstance(source, Mapping) and source.get("chunk_id") is not None
        }
        raw_relevant_ids = case.get("relevant_chunk_ids", [])
        if not isinstance(raw_relevant_ids, list):
            raise ValueError(
                f"Evaluation case {index} relevant_chunk_ids must be a list"
            )
        relevant_ids = set(raw_relevant_ids)
        if relevant_ids:
            raw_ranked_ids = case.get("retrieved_chunk_ids", [])
            if not isinstance(raw_ranked_ids, list):
                raise ValueError(
                    f"Evaluation case {index} retrieved_chunk_ids must be a list"
                )
            ranked_ids = raw_ranked_ids[:top_k]
            hits = len(set(ranked_ids) & relevant_ids)
            precision_values.append(hits / top_k)
            recall_values.append(hits / len(relevant_ids))
            coverage_values.append(
                len(actual_ids & relevant_ids) / len(relevant_ids)
            )

        source_refs = {
            source.get("source_ref")
            for source in sources
            if isinstance(source, Mapping)
        }
        answer = response.get("answer", "")
        citations = {
            match
            for match in re.findall(r"\[(S\d+)\]", str(answer))
        }
        structured_plan = response.get("project_plan")
        if isinstance(structured_plan, Mapping):
            for key in ("tasks", "requirements", "deliverables", "milestones"):
                items = structured_plan.get(key, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, Mapping):
                            refs = item.get("source_refs", [])
                            if isinstance(refs, list):
                                citations.update(
                                    ref for ref in refs if isinstance(ref, str)
                                )
        if citations:
            citation_validity.append(len(citations & source_refs) / len(citations))
        elif source_refs:
            citation_validity.append(0.0)

        raw_expected_types = case.get("expected_fact_types", [])
        if not isinstance(raw_expected_types, list):
            raise ValueError(
                f"Evaluation case {index} expected_fact_types must be a list"
            )
        expected_types = set(raw_expected_types)
        extraction = response.get("extracted_facts", {})
        facts = extraction.get("facts", []) if isinstance(extraction, Mapping) else []
        predicted_types: set[str] = set()
        if isinstance(facts, list):
            predicted_types.update(
                fact["type"]
                for fact in facts
                if isinstance(fact, Mapping) and isinstance(fact.get("type"), str)
            )
        fact_true_positives += len(expected_types & predicted_types)
        fact_false_positives += len(predicted_types - expected_types)
        fact_false_negatives += len(expected_types - predicted_types)
        plan_validation = response.get("plan_validation")
        if isinstance(plan_validation, list):
            invalid_plan_count += bool(plan_validation)
        schedule = response.get("schedule")
        if isinstance(schedule, Mapping):
            invalid_plan_count += bool(schedule.get("issues"))
            invalid_plan_count += sum(
                bool(item.get("deadline_violation"))
                for item in schedule.get("schedule", [])
                if isinstance(item, Mapping)
            )

        model_metadata = response.get("model_metadata", {})
        if isinstance(model_metadata, Mapping):
            latency = model_metadata.get("generation_seconds")
            throughput = model_metadata.get("tokens_per_second")
            if isinstance(latency, (int, float)):
                latency_values.append(float(latency))
            if isinstance(throughput, (int, float)):
                throughput_values.append(float(throughput))

    fact_precision = (
        fact_true_positives / (fact_true_positives + fact_false_positives)
        if fact_true_positives + fact_false_positives
        else 0.0
    )
    fact_recall = (
        fact_true_positives / (fact_true_positives + fact_false_negatives)
        if fact_true_positives + fact_false_negatives
        else 0.0
    )
    return {
        "case_count": len(case_list),
        "retrieval_precision_at_k": _mean(precision_values),
        "retrieval_recall_at_k": _mean(recall_values),
        "evidence_coverage": _mean(coverage_values),
        "citation_validity": _mean(citation_validity),
        "fact_type_precision": fact_precision,
        "fact_type_recall": fact_recall,
        "invalid_plan_count": invalid_plan_count,
        "mean_generation_seconds": _mean(latency_values),
        "mean_generated_tokens_per_second": _mean(throughput_values),
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None
