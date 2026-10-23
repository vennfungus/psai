"""Tests for deterministic project plan validation and scheduling."""

from datetime import date
import unittest

from project_management import (
    Deliverable,
    Milestone,
    ProjectConstraint,
    ProjectPlan,
    ProjectRequirement,
    Task,
    schedule_plan,
    validate_plan,
)


class ProjectPlanTests(unittest.TestCase):
    def test_detects_duplicate_ids_unknown_dependencies_and_cycles(self) -> None:
        plan = ProjectPlan(
            tasks=[
                Task("A", "Task A", 1, ["B"]),
                Task("A", "Duplicate", 1, ["missing"]),
                Task("B", "Task B", 1, ["A"]),
            ]
        )
        codes = {issue["code"] for issue in validate_plan(plan)}
        self.assertIn("duplicate_task_id", codes)
        self.assertIn("unknown_dependency", codes)
        self.assertIn("dependency_cycle", codes)

    def test_missing_duration_and_start_date_are_unresolved_not_invented(self) -> None:
        plan = ProjectPlan(tasks=[Task("A", "Task A")])
        result = schedule_plan(plan)
        self.assertEqual(result["schedule"], [])
        self.assertTrue(any(issue["code"] == "missing_duration" for issue in result["issues"]))
        self.assertTrue(any("start_date" in item for item in result["unresolved"]))

    def test_schedule_is_deterministic_and_respects_dependencies(self) -> None:
        plan = ProjectPlan(
            start_date=date(2026, 1, 1),
            tasks=[
                Task("B", "Build", 2, ["A"]),
                Task("A", "Design", 3, []),
            ],
        )
        result = schedule_plan(plan)
        self.assertEqual(result["issues"], [])
        self.assertEqual(
            result["schedule"],
            [
                {
                    "task_id": "A",
                    "title": "Design",
                    "start_date": "2026-01-01",
                    "finish_date": "2026-01-03",
                    "duration_days": 3,
                    "dependencies": [],
                    "source_refs": [],
                },
                {
                    "task_id": "B",
                    "title": "Build",
                    "start_date": "2026-01-04",
                    "finish_date": "2026-01-05",
                    "duration_days": 2,
                    "dependencies": ["A"],
                    "source_refs": [],
                },
            ],
        )

    def test_deadline_violation_is_reported_not_adjusted(self) -> None:
        plan = ProjectPlan(
            start_date=date(2026, 1, 1),
            tasks=[
                Task(
                    "A",
                    "Long task",
                    5,
                    dependencies=[],
                    deadline=date(2026, 1, 3),
                )
            ],
        )
        result = schedule_plan(plan)
        self.assertIn("deadline_violation", result["schedule"][0])
        self.assertEqual(result["schedule"][0]["finish_date"], "2026-01-05")

    def test_plan_parser_rejects_invalid_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid ISO date"):
            ProjectPlan.from_dict(
                {
                    "start_date": "not-a-date",
                    "tasks": [],
                }
            )

    def test_plan_schema_preserves_requirements_deliverables_and_milestones(self) -> None:
        plan = ProjectPlan.from_dict(
            {
                "start_date": "2026-01-01",
                "tasks": [
                    {
                        "task_id": "T1",
                        "title": "Design",
                        "duration_days": 1,
                        "dependencies": [],
                    }
                ],
                "requirements": [
                    {
                        "requirement_id": "R1",
                        "description": "Use local processing",
                        "source_refs": ["S1"],
                    }
                ],
                "constraints": [
                    {
                        "description": "No hosted inference",
                        "constraint_type": "technical",
                        "source_refs": ["S1"],
                    }
                ],
                "deliverables": [
                    {
                        "deliverable_id": "D1",
                        "title": "Design document",
                        "task_refs": ["T1"],
                        "source_refs": ["S1"],
                    }
                ],
                "milestones": [
                    {
                        "name": "Design complete",
                        "target_date": None,
                        "task_refs": ["T1"],
                        "source_refs": ["S1"],
                    }
                ],
            }
        )
        self.assertIsInstance(plan.requirements[0], ProjectRequirement)
        self.assertIsInstance(plan.constraints[0], ProjectConstraint)
        self.assertIsInstance(plan.deliverables[0], Deliverable)
        self.assertIsInstance(plan.milestones[0], Milestone)
        schedule = schedule_plan(plan)
        self.assertEqual(schedule["milestones"][0]["date"], "2026-01-01")
        self.assertIn("Project constraint requires review", schedule["unresolved"][0])

    def test_unknown_dependency_information_blocks_scheduling(self) -> None:
        plan = ProjectPlan(
            start_date=date(2026, 1, 1),
            tasks=[Task("A", "Task A", 1)],
        )
        result = schedule_plan(plan)
        self.assertEqual(result["schedule"], [])
        self.assertIn(
            "unknown_dependencies",
            {issue["code"] for issue in result["issues"]},
        )

    def test_fractional_calendar_duration_is_rejected(self) -> None:
        plan = ProjectPlan(
            start_date=date(2026, 1, 1),
            tasks=[Task("A", "Task A", 1.5, [])],
        )
        self.assertIn(
            "invalid_duration",
            {issue["code"] for issue in validate_plan(plan)},
        )


if __name__ == "__main__":
    unittest.main()
