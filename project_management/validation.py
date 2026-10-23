"""Deterministic validation for project task dependencies and constraints."""

from __future__ import annotations

import math
from collections import Counter

from .models import ProjectPlan


def validate_plan(plan: ProjectPlan) -> list[dict[str, str]]:
    """Return explicit validation issues; never repair missing plan data."""
    issues: list[dict[str, str]] = []
    counts = Counter(task.task_id for task in plan.tasks)
    known_ids = set(counts)
    for task in plan.tasks:
        if not task.task_id.strip():
            issues.append(_issue("missing_task_id", "Task identifier is empty"))
        if counts[task.task_id] > 1:
            issues.append(
                _issue(
                    "duplicate_task_id",
                    f"Task identifier {task.task_id!r} is not unique",
                )
            )
        if not task.title.strip():
            issues.append(
                _issue("missing_task_title", f"Task {task.task_id!r} has no title")
            )
        if task.duration_days is None:
            issues.append(
                _issue(
                    "missing_duration",
                    f"Task {task.task_id!r} has no confirmed duration",
                )
            )
        elif (
            not math.isfinite(task.duration_days)
            or task.duration_days <= 0
            or not float(task.duration_days).is_integer()
        ):
            issues.append(
                _issue(
                    "invalid_duration",
                    f"Task {task.task_id!r} duration must be a positive whole "
                    "number of calendar days",
                )
            )
        if task.earliest_start and task.deadline:
            if task.deadline < task.earliest_start:
                issues.append(
                    _issue(
                        "invalid_task_dates",
                        f"Task {task.task_id!r} deadline precedes its earliest start",
                    )
                )
        if task.parent_task_id is not None:
            if task.parent_task_id not in known_ids:
                issues.append(
                    _issue(
                        "unknown_parent_task",
                        f"Task {task.task_id!r} has unknown parent task "
                        f"{task.parent_task_id!r}",
                    )
                )
            elif task.parent_task_id == task.task_id:
                issues.append(
                    _issue(
                        "self_parent_task",
                        f"Task {task.task_id!r} cannot be its own parent",
                    )
                )
        if task.dependencies is None:
            issues.append(
                _issue(
                    "unknown_dependencies",
                    f"Task {task.task_id!r} has no confirmed dependency information",
                )
            )
        for dependency in task.dependencies or []:
            if dependency not in known_ids:
                issues.append(
                    _issue(
                        "unknown_dependency",
                        f"Task {task.task_id!r} depends on unknown task "
                        f"{dependency!r}",
                    )
                )
            if dependency == task.task_id:
                issues.append(
                    _issue(
                        "self_dependency",
                        f"Task {task.task_id!r} cannot depend on itself",
                    )
                )
    issues.extend(_cycle_issues(plan))
    issues.extend(_parent_cycle_issues(plan))
    for constraint in plan.constraints:
        if not constraint.description.strip():
            issues.append(
                _issue(
                    "invalid_constraint",
                    "Project constraint description must not be empty",
                )
            )
    for requirement in plan.requirements:
        if not requirement.requirement_id.strip() or not requirement.description.strip():
            issues.append(
                _issue(
                    "invalid_requirement",
                    "Requirements need a non-empty identifier and description",
                )
            )
    for deliverable in plan.deliverables:
        if not deliverable.deliverable_id.strip() or not deliverable.title.strip():
            issues.append(
                _issue(
                    "invalid_deliverable",
                    "Deliverables require a non-empty identifier and title",
                )
            )
        for task_ref in deliverable.task_refs:
            if task_ref not in known_ids:
                issues.append(
                    _issue(
                        "unknown_deliverable_task",
                        f"Deliverable {deliverable.deliverable_id!r} references "
                        f"unknown task {task_ref!r}",
                    )
                )
    for milestone in plan.milestones:
        if not milestone.name.strip():
            issues.append(
                _issue("invalid_milestone", "Milestone name must not be empty")
            )
        for task_ref in milestone.task_refs:
            if task_ref not in known_ids:
                issues.append(
                    _issue(
                        "unknown_milestone_task",
                        f"Milestone {milestone.name!r} references unknown task "
                        f"{task_ref!r}",
                    )
                )
    return issues


def _cycle_issues(plan: ProjectPlan) -> list[dict[str, str]]:
    graph: dict[str, list[str]] = {}
    for task in plan.tasks:
        graph.setdefault(task.task_id, []).extend(task.dependencies or [])
    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: set[tuple[str, ...]] = set()

    def visit(task_id: str) -> None:
        state[task_id] = 1
        stack.append(task_id)
        for dependency in graph.get(task_id, []):
            if dependency not in graph:
                continue
            if state.get(dependency, 0) == 0:
                visit(dependency)
            elif state.get(dependency) == 1:
                cycle_start = stack.index(dependency)
                cycle = tuple(sorted(stack[cycle_start:]))
                cycles.add(cycle)
        stack.pop()
        state[task_id] = 2

    for task_id in graph:
        if state.get(task_id, 0) == 0:
            visit(task_id)
    return [
        _issue("dependency_cycle", "Dependency cycle detected: " + " -> ".join(cycle))
        for cycle in sorted(cycles)
    ]


def _parent_cycle_issues(plan: ProjectPlan) -> list[dict[str, str]]:
    parents = {
        task.task_id: task.parent_task_id
        for task in plan.tasks
        if task.parent_task_id is not None
    }
    cycles: set[tuple[str, ...]] = set()
    for task_id in parents:
        path: list[str] = []
        current: str | None = task_id
        while current in parents:
            if current in path:
                cycles.add(tuple(sorted(path[path.index(current) :])))
                break
            path.append(current)
            current = parents[current]
    return [
        _issue("wbs_parent_cycle", "WBS parent cycle detected: " + " -> ".join(cycle))
        for cycle in sorted(cycles)
    ]


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}
