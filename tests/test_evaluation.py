"""Tests for labelled evaluation metric calculations."""

import unittest

from evaluation.metrics import evaluate_cases


class EvaluationMetricsTests(unittest.TestCase):
    def test_retrieval_citation_and_extraction_metrics(self) -> None:
        metrics = evaluate_cases(
            [
                {
                    "relevant_chunk_ids": ["a", "b"],
                    "retrieved_chunk_ids": ["a", "c"],
                    "expected_fact_types": ["task", "dependency"],
                    "response": {
                        "answer": "Task A depends on B [S1].",
                        "sources": [
                            {"source_ref": "S1", "chunk_id": "a"},
                        ],
                        "retrieval_metadata": {},
                        "extracted_facts": {
                            "facts": [{"type": "dependency"}],
                        },
                    },
                }
            ],
            top_k=2,
        )
        self.assertEqual(metrics["retrieval_precision_at_k"], 0.5)
        self.assertEqual(metrics["retrieval_recall_at_k"], 0.5)
        self.assertEqual(metrics["evidence_coverage"], 0.5)
        self.assertEqual(metrics["citation_validity"], 1.0)
        self.assertEqual(metrics["fact_type_precision"], 1.0)
        self.assertEqual(metrics["fact_type_recall"], 0.5)

    def test_unknown_citation_and_plan_violation_are_counted(self) -> None:
        metrics = evaluate_cases(
            [
                {
                    "response": {
                        "answer": "Claim [S9].",
                        "sources": [{"source_ref": "S1", "chunk_id": "a"}],
                        "retrieval_metadata": {},
                        "plan_validation": [{"code": "dependency_cycle"}],
                    }
                }
            ],
            top_k=5,
        )
        self.assertEqual(metrics["citation_validity"], 0.0)
        self.assertEqual(metrics["invalid_plan_count"], 1)

    def test_empty_cases_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            evaluate_cases([], top_k=5)


if __name__ == "__main__":
    unittest.main()
