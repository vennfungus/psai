"""Small typed models for evidence-backed project plans."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class ProjectRequirement:
    requirement_id: str
    description: str
    source_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectRequirement":
        if not isinstance(value, dict):
            raise ValueError("Each requirement must be an object")
        requirement_id = value.get("requirement_id")
        description = value.get("description")
        source_refs = value.get("source_refs", [])
        if not isinstance(requirement_id, str) or not isinstance(
            description, str
        ):
            raise ValueError(
                "Requirement requirement_id and description must be strings"
            )
        if not isinstance(source_refs, list) or not all(
            isinstance(item, str) for item in source_refs
        ):
            raise ValueError("Requirement source_refs must be a list of strings")
        return cls(requirement_id, description, list(source_refs))


@dataclass
class ProjectConstraint:
    description: str
    constraint_type: str | None = None
    source_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectConstraint":
        if not isinstance(value, dict):
            raise ValueError("Each constraint must be an object")
        description = value.get("description")
        constraint_type = value.get("constraint_type")
        source_refs = value.get("source_refs", [])
        if not isinstance(description, str):
            raise ValueError("Constraint description must be a string")
        if constraint_type is not None and not isinstance(constraint_type, str):
            raise ValueError("Constraint type must be a string or null")
        if not isinstance(source_refs, list) or not all(
            isinstance(item, str) for item in source_refs
        ):
            raise ValueError("Constraint source_refs must be a list of strings")
        return cls(description, constraint_type, list(source_refs))


@dataclass
class Deliverable:
    deliverable_id: str
    title: str
    task_refs: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Deliverable":
        if not isinstance(value, dict):
            raise ValueError("Each deliverable must be an object")
        deliverable_id = value.get("deliverable_id")
        title = value.get("title")
        task_refs = value.get("task_refs", [])
        source_refs = value.get("source_refs", [])
        if not isinstance(deliverable_id, str) or not isinstance(title, str):
            raise ValueError(
                "Deliverable deliverable_id and title must be strings"
            )
        if any(
            not isinstance(items, list)
            or not all(isinstance(item, str) for item in items)
            for items in (task_refs, source_refs)
        ):
            raise ValueError(
                "Deliverable task_refs and source_refs must be string lists"
            )
        return cls(
            deliverable_id,
            title,
            list(task_refs),
            list(source_refs),
        )


@dataclass
class Milestone:
    name: str
    target_date: date | None = None
    task_refs: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Milestone":
        if not isinstance(value, dict):
            raise ValueError("Each milestone must be an object")
        name = value.get("name")
        if not isinstance(name, str):
            raise ValueError("Milestone name must be a string")
        refs = value.get("task_refs", [])
        sources = value.get("source_refs", [])
        if any(
            not isinstance(items, list)
            or not all(isinstance(item, str) for item in items)
            for items in (refs, sources)
        ):
            raise ValueError(
                "Milestone task_refs and source_refs must be string lists"
            )
        raw_date = value.get("target_date")
        target_date: date | None = None
        if raw_date is not None:
            if not isinstance(raw_date, str):
                raise ValueError("Milestone target_date must use ISO date text")
            try:
                target_date = date.fromisoformat(raw_date)
            except ValueError as exc:
                raise ValueError(
                    "Milestone target_date must be a valid ISO date (YYYY-MM-DD)"
                ) from exc
        return cls(name, target_date, list(refs), list(sources))


@dataclass
class Task:
    task_id: str
    title: str
    duration_days: float | None = None
    dependencies: list[str] | None = None
    parent_task_id: str | None = None
    earliest_start: date | None = None
    deadline: date | None = None
    source_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Task":
        if not isinstance(value, dict):
            raise ValueError("Each task must be an object")

        def parse_date(name: str) -> date | None:
            raw = value.get(name)
            if raw is None or raw == "":
                return None
            if not isinstance(raw, str):
                raise ValueError(f"Task {name} must use ISO date text")
            try:
                return date.fromisoformat(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Task {name} must be a valid ISO date (YYYY-MM-DD)"
                ) from exc

        raw_dependencies = value.get("dependencies")
        raw_source_refs = value.get("source_refs", [])
        if raw_dependencies is not None and (
            not isinstance(raw_dependencies, list)
            or not all(isinstance(item, str) for item in raw_dependencies)
        ):
            raise ValueError("Task dependencies must be a list of identifiers or null")
        if not isinstance(raw_source_refs, list) or not all(
            isinstance(item, str) for item in raw_source_refs
        ):
            raise ValueError("Task source_refs must be a list of strings")
        task_id = value.get("task_id")
        title = value.get("title")
        parent_task_id = value.get("parent_task_id")
        duration = value.get("duration_days")
        if not isinstance(task_id, str) or not isinstance(title, str):
            raise ValueError("Task task_id and title must be strings")
        if duration is not None and (
            isinstance(duration, bool) or not isinstance(duration, (int, float))
        ):
            raise ValueError("Task duration_days must be a number or null")
        if parent_task_id is not None and not isinstance(parent_task_id, str):
            raise ValueError("Task parent_task_id must be a string or null")
        return cls(
            task_id=task_id,
            title=title,
            duration_days=float(duration) if duration is not None else None,
            dependencies=(
                list(raw_dependencies) if raw_dependencies is not None else None
            ),
            parent_task_id=parent_task_id,
            earliest_start=parse_date("earliest_start"),
            deadline=parse_date("deadline"),
            source_refs=list(raw_source_refs),
        )


@dataclass
class ProjectPlan:
    tasks: list[Task]
    start_date: date | None = None
    requirements: list[ProjectRequirement] = field(default_factory=list)
    constraints: list[ProjectConstraint] = field(default_factory=list)
    deliverables: list[Deliverable] = field(default_factory=list)
    milestones: list[Milestone] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectPlan":
        if not isinstance(value, dict):
            raise ValueError("Project plan must be an object")
        tasks = value.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("Project plan must contain a tasks array")
        raw_requirements = value.get("requirements", [])
        raw_constraints = value.get("constraints", [])
        raw_deliverables = value.get("deliverables", [])
        raw_milestones = value.get("milestones", [])
        if any(
            not isinstance(items, list)
            for items in (
                raw_requirements,
                raw_constraints,
                raw_deliverables,
                raw_milestones,
            )
        ):
            raise ValueError(
                "requirements, constraints, deliverables, and milestones "
                "must be arrays"
            )
        start_date = value.get("start_date")
        parsed_start: date | None = None
        if start_date is not None:
            if not isinstance(start_date, str):
                raise ValueError("Project start_date must use ISO date text")
            try:
                parsed_start = date.fromisoformat(start_date)
            except ValueError as exc:
                raise ValueError(
                    "Project start_date must be a valid ISO date (YYYY-MM-DD)"
                ) from exc
        return cls(
            tasks=[Task.from_dict(item) for item in tasks],
            start_date=parsed_start,
            requirements=[
                ProjectRequirement.from_dict(item) for item in raw_requirements
            ],
            constraints=[
                ProjectConstraint.from_dict(item) for item in raw_constraints
            ],
            deliverables=[
                Deliverable.from_dict(item) for item in raw_deliverables
            ],
            milestones=[Milestone.from_dict(item) for item in raw_milestones],
        )
