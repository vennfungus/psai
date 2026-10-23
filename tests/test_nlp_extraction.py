"""Tests for conservative project-fact extraction."""

import unittest

from nlp_system.extractors import extract_project_facts


class ProjectFactExtractionTests(unittest.TestCase):
    def test_extracts_task_duration_cost_dependency_and_provenance(self) -> None:
        result = extract_project_facts(
            [
                {
                    "source_ref": "S1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "source_filename": "proposal.pdf",
                    "page_number": 4,
                    "text": (
                        "Task T-12 - Build the prototype. "
                        "Task T-12 depends on Task T-11. "
                        "The work takes 4 days. The budget is $1,250."
                    ),
                }
            ]
        )
        fact_types = {fact["type"] for fact in result["facts"]}
        self.assertTrue(
            {"task", "dependency", "duration", "cost"} <= fact_types
        )
        dependency = next(
            fact for fact in result["facts"] if fact["type"] == "dependency"
        )
        self.assertEqual(dependency["value"], {"task": "T-12", "prerequisite": "T-11"})
        self.assertEqual(dependency["source"]["source_ref"], "S1")
        self.assertEqual(dependency["source"]["page_number"], 4)
        self.assertEqual(
            next(f for f in result["facts"] if f["type"] == "cost")["value"],
            {"amount": "1250", "currency": "$"},
        )

    def test_requirement_negation_and_unknown_fields_preserve_evidence(self) -> None:
        result = extract_project_facts(
            [
                {
                    "source_ref": "S2",
                    "text": (
                        "The system must not rely on external APIs. "
                        "The budget is not stated."
                    ),
                }
            ]
        )
        requirement = next(
            fact for fact in result["facts"] if fact["type"] == "requirement"
        )
        self.assertEqual(
            requirement["evidence"],
            "The system must not rely on external APIs.",
        )
        self.assertEqual(
            requirement["polarity"],
            "negative_or_exception",
        )
        self.assertEqual(requirement["source"]["source_ref"], "S2")
        self.assertEqual(result["field_status"]["costs"], "unknown_requires_review")

    def test_vietnamese_dependency_pattern(self) -> None:
        result = extract_project_facts(
            [{"text": "Công việc T-2 phụ thuộc vào Công việc T-1."}]
        )
        dependency = next(
            fact for fact in result["facts"] if fact["type"] == "dependency"
        )
        self.assertEqual(dependency["value"], {"task": "T-2", "prerequisite": "T-1"})

    def test_budget_without_currency_and_milestones_are_preserved(self) -> None:
        result = extract_project_facts(
            [
                {
                    "source_ref": "S3",
                    "text": (
                        "Budget: 2.5 million. Milestone 2 - Prototype review."
                    ),
                }
            ]
        )
        cost = next(fact for fact in result["facts"] if fact["type"] == "cost")
        milestone = next(
            fact for fact in result["facts"] if fact["type"] == "milestone"
        )
        self.assertEqual(cost["value"]["amount"], "2.5")
        self.assertEqual(cost["value"]["unit"].casefold(), "million")
        self.assertEqual(milestone["value"]["id"], "2")
        self.assertEqual(milestone["source"]["source_ref"], "S3")


if __name__ == "__main__":
    unittest.main()
