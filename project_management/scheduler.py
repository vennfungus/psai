"""Calendar-day scheduling using validated task durations and dependencies."""

from __future__ import annotations

from datetime import date, timedelta

from .models import ProjectPlan
from .validation import validate_plan


def schedule_plan(plan: ProjectPlan) -> dict[str, object]:
    """Schedule tasks deterministically; report unresolved inputs instead."""
    issues = validate_plan(plan)
    unresolved: list[str] = []
    if not plan.tasks:
        unresolved.append("No tasks were provided.")
    if plan.start_date is None:
        unresolved.append("A project start_date is required for scheduling.")
    for task in plan.tasks:
        if task.duration_days is None:
            unresolved.append(f"Task {task.task_id!r} requires a duration.")
        if task.dependencies is None:
            unresolved.append(
                f"Task {task.task_id!r} requires confirmed dependency information."
            )
    schedule_blockers = list(unresolved)
    for constraint in plan.constraints:
        unresolved.append(
            f"Project constraint requires review before relying on the schedule: "
            f"{constraint.description}"
        )
    if issues:
        return {
            "schedule": [],
            "milestones": [],
            "issues": issues,
            "unresolved": unresolved,
            "calendar": "calendar days; task finish date includes its start day",
        }
    if schedule_blockers:
        return {
            "schedule": [],
            "milestones": [],
            "issues": [],
            "unresolved": unresolved,
            "calendar": "calendar days; task finish date includes its start day",
        }

    by_id = {task.task_id: task for task in plan.tasks}
    remaining = set(by_id)
    completed: dict[str, tuple[date, date]] = {}
    output: list[dict[str, object]] = []
    while remaining:
        ready = sorted(
            task_id
            for task_id in remaining
            if all(
                dep in completed
                for dep in (by_id[task_id].dependencies or [])
            )
        )
        if not ready:
            return {
                "schedule": [],
                "milestones": [],
                "issues": [
                    {
                        "code": "unschedulable_graph",
                        "message": "Task graph could not be scheduled",
                    }
                ],
                "unresolved": unresolved,
                "calendar": "calendar days; task finish date includes its start day",
            }
        for task_id in ready:
            task = by_id[task_id]
            dependency_starts = [
                completed[dependency][1] + timedelta(days=1)
                for dependency in (task.dependencies or [])
            ]
            planned_start = max(
                [plan.start_date, *dependency_starts]
            )
            if task.earliest_start:
                planned_start = max(planned_start, task.earliest_start)
            duration_days = int(task.duration_days)
            planned_finish = planned_start + timedelta(days=duration_days - 1)
            completed[task_id] = (planned_start, planned_finish)
            entry: dict[str, object] = {
                "task_id": task.task_id,
                "title": task.title,
                "start_date": planned_start.isoformat(),
                "finish_date": planned_finish.isoformat(),
                "duration_days": duration_days,
                "dependencies": task.dependencies or [],
                "source_refs": task.source_refs,
            }
            if task.deadline and planned_finish > task.deadline:
                entry["deadline_violation"] = (
                    f"Finish date {planned_finish.isoformat()} is after deadline "
                    f"{task.deadline.isoformat()}"
                )
            output.append(entry)
            remaining.remove(task_id)
    scheduled_milestones: list[dict[str, object]] = []
    for milestone in plan.milestones:
        task_finish = (
            max(completed[task_id][1] for task_id in milestone.task_refs)
            if milestone.task_refs
            else None
        )
        if milestone.target_date is not None:
            milestone_date = milestone.target_date
        elif task_finish is not None:
            milestone_date = task_finish
        else:
            unresolved.append(
                f"Milestone {milestone.name!r} needs a target date or task references."
            )
            continue
        milestone_entry: dict[str, object] = {
            "name": milestone.name,
            "date": milestone_date.isoformat(),
            "task_refs": milestone.task_refs,
            "source_refs": milestone.source_refs,
        }
        if task_finish and task_finish > milestone_date:
            milestone_entry["date_violation"] = (
                f"Referenced tasks finish on {task_finish.isoformat()}, after "
                f"target date {milestone_date.isoformat()}"
            )
        scheduled_milestones.append(milestone_entry)
    return {
        "schedule": output,
        "milestones": scheduled_milestones,
        "issues": [],
        "unresolved": unresolved,
        "calendar": "calendar days; task finish date includes its start day",
    }
