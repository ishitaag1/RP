"""
Critical Path Method (CPM) scheduling engine for a construction project.

This module is deterministic on purpose. It never calls an LLM. Later, an
agent can *ask* this module questions through tools, but the dates always
come from the graph math below.

Calendar convention (version 1)
-------------------------------
- Durations are whole **calendar days**. Weekends and holidays count.
- An activity with duration D starting on date S occupies:
      S, S+1, ..., S+(D-1)
  That last occupied day is the **inclusive finish**.
- A successor may start on the next calendar day, which is S+D.
- Internally we count integer day offsets from PROJECT_EPOCH (2026-03-01).
  Offset 0 is 2026-03-01, offset 1 is 2026-03-02, and so on.
- baseline_start in the CSV is a "not earlier than" constraint. An activity
  will not be pulled earlier than that date, even if its predecessors finish
  early. A delay can only push dates later.
- Finish-to-start links only. No leads, lags, or SS/FF/SF relationships.
- No resource leveling: two activities may overlap even if they would need
  the same crew. Version 1 also does not suggest recovery (overtime, extra
  crews, or crashing).

Example: duration 3 starting 2026-03-01 occupies Mar 1, 2, 3.
Inclusive finish = 2026-03-03. Successor earliest start = 2026-03-04.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

PROJECT_EPOCH = date(2026, 3, 1)
REQUIRED_COLUMNS = (
    "activity_id",
    "activity_name",
    "tower_id",
    "predecessors",
    "duration_days",
    "baseline_start",
)
DATE_FORMAT = "%Y-%m-%d"


class ScheduleError(Exception):
    """Base error for anything the scheduling engine cannot compute."""


class InvalidActivityIdError(ScheduleError):
    """Raised when an activity_id is missing from the loaded schedule."""


class NegativeDelayError(ScheduleError):
    """Raised when a delay is negative. Recovery/crashing is out of scope."""


class InvalidDurationError(ScheduleError):
    """Raised when a duration is missing, zero, or negative."""


class MissingPredecessorError(ScheduleError):
    """Raised when a predecessor ID does not exist in the activity table."""


class DependencyCycleError(ScheduleError):
    """Raised when the network contains a cycle (A depends on B depends on A)."""


class DuplicateActivityIdError(ScheduleError):
    """Raised when two rows share the same activity_id."""


@dataclass
class ActivitySchedule:
    """One activity after a CPM pass: dates, float, and critical-path flag."""

    activity_id: str
    activity_name: str
    tower_id: str
    predecessors: list[str]
    duration_days: int
    baseline_start: date
    earliest_start: date
    earliest_finish: date  # inclusive last occupied day
    latest_start: date
    latest_finish: date  # inclusive
    total_float_days: int
    is_critical: bool


@dataclass
class ProjectSchedule:
    """Full CPM result for every activity, plus project-level facts."""

    activities: dict[str, ActivitySchedule]
    project_start: date
    project_finish: date
    project_duration_days: int
    critical_path: list[str]
    critical_activities: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Turn the schedule into a table that Streamlit can display later."""
        rows = []
        for item in self.activities.values():
            rows.append(
                {
                    "activity_id": item.activity_id,
                    "activity_name": item.activity_name,
                    "tower_id": item.tower_id,
                    "predecessors": ";".join(item.predecessors),
                    "duration_days": item.duration_days,
                    "baseline_start": item.baseline_start.isoformat(),
                    "earliest_start": item.earliest_start.isoformat(),
                    "earliest_finish": item.earliest_finish.isoformat(),
                    "latest_start": item.latest_start.isoformat(),
                    "latest_finish": item.latest_finish.isoformat(),
                    "total_float_days": item.total_float_days,
                    "is_critical": item.is_critical,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("activity_id").reset_index(drop=True)


def date_to_offset(day: date) -> int:
    """Convert a calendar date to a day number counted from PROJECT_EPOCH."""
    return (day - PROJECT_EPOCH).days


def offset_to_date(offset: int) -> date:
    """Convert a day number counted from PROJECT_EPOCH back to a date."""
    return PROJECT_EPOCH + timedelta(days=offset)


def parse_predecessors(raw_value: Any) -> list[str]:
    """
    Split a predecessor cell into a list of activity IDs.

    Empty cells mean "no predecessors" (a start activity).
    We accept semicolon or comma separators so the CSV stays easy to edit.
    """
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []
    text = str(raw_value).strip()
    if text == "" or text.lower() == "nan":
        return []
    parts = text.replace(",", ";").split(";")
    return [part.strip() for part in parts if part.strip()]


def parse_iso_date(raw_value: Any, activity_id: str) -> date:
    """Parse YYYY-MM-DD. Fail clearly if the date is unreadable."""
    if isinstance(raw_value, datetime):
        return raw_value.date()
    if isinstance(raw_value, date):
        return raw_value
    text = str(raw_value).strip()
    try:
        return datetime.strptime(text, DATE_FORMAT).date()
    except ValueError as exc:
        raise ScheduleError(
            f"Activity {activity_id} has an invalid baseline_start '{raw_value}'. "
            f"Use {DATE_FORMAT}."
        ) from exc


def load_activities(csv_path: str | Path) -> pd.DataFrame:
    """
    Read activities.csv and return a cleaned table.

    Each row is one unique activity. Duplicate names across towers are OK
    because we key everything on activity_id (for example T1-TIL vs T3-TIL).
    """
    path = Path(csv_path)
    if not path.exists():
        raise ScheduleError(f"Activity file not found: {path}")

    frame = pd.read_csv(path, dtype={"activity_id": str, "tower_id": str})
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ScheduleError(f"activities.csv is missing columns: {missing}")

    frame["activity_id"] = frame["activity_id"].str.strip()
    frame["activity_name"] = frame["activity_name"].astype(str).str.strip()
    frame["tower_id"] = frame["tower_id"].astype(str).str.strip()
    frame["predecessors"] = frame["predecessors"].apply(parse_predecessors)
    frame["duration_days"] = pd.to_numeric(frame["duration_days"], errors="coerce")
    frame["baseline_start"] = [
        parse_iso_date(value, activity_id)
        for value, activity_id in zip(frame["baseline_start"], frame["activity_id"])
    ]

    if frame["activity_id"].duplicated().any():
        dupes = frame.loc[frame["activity_id"].duplicated(), "activity_id"].tolist()
        raise DuplicateActivityIdError(f"Duplicate activity_id values: {dupes}")

    if frame["activity_id"].eq("").any():
        raise InvalidActivityIdError("One or more rows have an empty activity_id.")

    bad_durations = frame[frame["duration_days"].isna() | (frame["duration_days"] < 1)]
    if not bad_durations.empty:
        bad_ids = bad_durations["activity_id"].tolist()
        raise InvalidDurationError(
            f"Duration must be an integer of at least 1 day. Bad rows: {bad_ids}"
        )
    frame["duration_days"] = frame["duration_days"].astype(int)
    return frame.reset_index(drop=True)


def build_graph(frame: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph: predecessor -> successor.

    An arrow A -> B means "A must finish before B can start".
    """
    graph = nx.DiGraph()
    for _, row in frame.iterrows():
        graph.add_node(
            row["activity_id"],
            activity_name=row["activity_name"],
            tower_id=row["tower_id"],
            duration_days=int(row["duration_days"]),
            baseline_start=row["baseline_start"],
            predecessors=list(row["predecessors"]),
        )

    known_ids = set(frame["activity_id"])
    missing_links: list[str] = []
    for _, row in frame.iterrows():
        for pred_id in row["predecessors"]:
            if pred_id not in known_ids:
                missing_links.append(f"{row['activity_id']} <- {pred_id}")
            else:
                graph.add_edge(pred_id, row["activity_id"])

    if missing_links:
        raise MissingPredecessorError(
            "These predecessor IDs do not exist in the activity table: "
            + ", ".join(missing_links)
        )
    return graph


def validate_graph(graph: nx.DiGraph) -> None:
    """Reject self-loops and cycles. CPM needs a Directed Acyclic Graph."""
    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        raise DependencyCycleError(
            f"An activity cannot depend on itself: {self_loops}"
        )
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        pretty = " -> ".join(node for node, _, _ in cycle)
        pretty += f" -> {cycle[0][0]}"
        raise DependencyCycleError(
            f"Dependency cycle detected: {pretty}. "
            "Fix the predecessor list before calculating the schedule."
        )


def topological_order(graph: nx.DiGraph) -> list[str]:
    """
    Return activity IDs in an order where every predecessor appears first.

    That order is what the forward pass needs. The backward pass uses the
    reverse of this list.
    """
    return list(nx.topological_sort(graph))


def forward_pass(
    graph: nx.DiGraph,
    durations: dict[str, int],
    earliest_start_floor: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Walk the network from left to right and find the earliest dates.

    For each activity:
      ES = max(its baseline floor, max exclusive-finish of its predecessors)
      exclusive_EF = ES + duration
      inclusive finish (shown to users) = exclusive_EF - 1

    We keep exclusive EF internally because "successor starts the next day"
    is then just successor_ES >= predecessor exclusive_EF.
    """
    earliest_start: dict[str, int] = {}
    exclusive_finish: dict[str, int] = {}

    for activity_id in topological_order(graph):
        pred_finishes = [
            exclusive_finish[pred_id] for pred_id in graph.predecessors(activity_id)
        ]
        from_preds = max(pred_finishes) if pred_finishes else earliest_start_floor[activity_id]
        start = max(earliest_start_floor[activity_id], from_preds)
        earliest_start[activity_id] = start
        exclusive_finish[activity_id] = start + durations[activity_id]
    return earliest_start, exclusive_finish


def backward_pass(
    graph: nx.DiGraph,
    durations: dict[str, int],
    exclusive_finish: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Walk the network from right to left and find the latest dates.

    The project exclusive finish is the maximum exclusive EF of all activities
    (usually the last activity, but we take the max so parallel ends work).
    Sink activities (no successors) must finish by that project finish.
    Other activities must finish in time for every successor's latest start.
    """
    project_exclusive_finish = max(exclusive_finish.values())
    latest_start: dict[str, int] = {}
    exclusive_latest_finish: dict[str, int] = {}

    for activity_id in reversed(topological_order(graph)):
        successors = list(graph.successors(activity_id))
        if not successors:
            lf = project_exclusive_finish
        else:
            lf = min(latest_start[succ_id] for succ_id in successors)
        exclusive_latest_finish[activity_id] = lf
        latest_start[activity_id] = lf - durations[activity_id]
    return latest_start, exclusive_latest_finish


def compute_total_float(
    earliest_start: dict[str, int],
    latest_start: dict[str, int],
) -> dict[str, int]:
    """
    Total float = how many days this activity can slip without delaying
    the project finish. Critical activities have float 0.
    """
    return {
        activity_id: latest_start[activity_id] - earliest_start[activity_id]
        for activity_id in earliest_start
    }


def identify_critical_activities(total_float: dict[str, int]) -> list[str]:
    """Return every activity with zero total float, sorted by ID for stability."""
    return sorted(
        activity_id for activity_id, float_days in total_float.items() if float_days == 0
    )


def extract_critical_path(
    graph: nx.DiGraph,
    earliest_start: dict[str, int],
    total_float: dict[str, int],
) -> list[str]:
    """
    Build one start-to-finish chain of zero-float activities.

    If several successors are critical, we pick the one that starts earliest,
    then the one with the smaller ID. That keeps the path deterministic.
    """
    critical = {activity_id for activity_id, float_days in total_float.items() if float_days == 0}
    if not critical:
        return []

    starts = [
        activity_id
        for activity_id in critical
        if all(pred_id not in critical for pred_id in graph.predecessors(activity_id))
    ]
    starts.sort(key=lambda activity_id: (earliest_start[activity_id], activity_id))
    path = [starts[0]]
    while True:
        critical_successors = [
            succ_id for succ_id in graph.successors(path[-1]) if succ_id in critical
        ]
        if not critical_successors:
            break
        critical_successors.sort(key=lambda activity_id: (earliest_start[activity_id], activity_id))
        path.append(critical_successors[0])
    return path


def compute_cpm(frame: pd.DataFrame) -> ProjectSchedule:
    """Run validation + forward pass + backward pass on a cleaned activity table."""
    graph = build_graph(frame)
    validate_graph(graph)

    durations = {
        row["activity_id"]: int(row["duration_days"]) for _, row in frame.iterrows()
    }
    earliest_start_floor = {
        row["activity_id"]: date_to_offset(row["baseline_start"])
        for _, row in frame.iterrows()
    }

    earliest_start, exclusive_finish = forward_pass(graph, durations, earliest_start_floor)
    latest_start, exclusive_latest_finish = backward_pass(graph, durations, exclusive_finish)
    total_float = compute_total_float(earliest_start, latest_start)
    critical_ids = identify_critical_activities(total_float)
    critical_path = extract_critical_path(graph, earliest_start, total_float)

    warnings: list[str] = []
    activities: dict[str, ActivitySchedule] = {}
    for _, row in frame.iterrows():
        activity_id = row["activity_id"]
        es = earliest_start[activity_id]
        ef_excl = exclusive_finish[activity_id]
        ls = latest_start[activity_id]
        lf_excl = exclusive_latest_finish[activity_id]
        if ls < es:
            warnings.append(
                f"{activity_id} has latest start before earliest start. "
                "Check baseline_start constraints."
            )
        activities[activity_id] = ActivitySchedule(
            activity_id=activity_id,
            activity_name=row["activity_name"],
            tower_id=row["tower_id"],
            predecessors=list(row["predecessors"]),
            duration_days=durations[activity_id],
            baseline_start=row["baseline_start"],
            earliest_start=offset_to_date(es),
            earliest_finish=offset_to_date(ef_excl - 1),
            latest_start=offset_to_date(ls),
            latest_finish=offset_to_date(lf_excl - 1),
            total_float_days=total_float[activity_id],
            is_critical=total_float[activity_id] == 0,
        )

    project_start = min(item.earliest_start for item in activities.values())
    project_finish = max(item.earliest_finish for item in activities.values())
    return ProjectSchedule(
        activities=activities,
        project_start=project_start,
        project_finish=project_finish,
        project_duration_days=(project_finish - project_start).days + 1,
        critical_path=critical_path,
        critical_activities=critical_ids,
        warnings=warnings,
    )


def find_activity_ids(
    frame: pd.DataFrame,
    activity_name: str,
    tower_id: str | None = None,
) -> list[str]:
    """
    Look up unique IDs from a human name such as "Tiling".

    If the same name exists in several towers, pass tower_id (T3) to
    disambiguate. If several still match, the caller must choose an ID.
    """
    name_mask = frame["activity_name"].str.casefold() == activity_name.casefold()
    matches = frame.loc[name_mask]
    if tower_id:
        matches = matches.loc[matches["tower_id"].str.casefold() == tower_id.casefold()]
    return matches["activity_id"].tolist()


def apply_delay(
    frame: pd.DataFrame,
    activity_id: str,
    delay_days: int,
) -> tuple[pd.DataFrame, ProjectSchedule, ProjectSchedule, dict[str, Any]]:
    """
    Add delay_days to one activity's duration and recalculate CPM.

    The original table is copied, never edited in place, so the baseline
    remains available for comparison.

    Delay meaning in version 1: the activity *takes longer* by delay_days.
    Downstream activities move only if this new exclusive finish pushes
    their earliest start. Activities with enough float may not move at all.

    Returns (revised_frame, original_schedule, revised_schedule, impact).
    """
    if delay_days < 0:
        raise NegativeDelayError(
            "Negative delays are not supported in version 1. "
            "This engine does not crash or compress the schedule."
        )
    if activity_id not in set(frame["activity_id"]):
        raise InvalidActivityIdError(
            f"Unknown activity_id '{activity_id}'. "
            "Use the unique ID (for example T3-TIL), not only the name."
        )

    original_schedule = compute_cpm(frame)
    revised_frame = frame.copy(deep=True)
    mask = revised_frame["activity_id"] == activity_id
    revised_frame.loc[mask, "duration_days"] = (
        revised_frame.loc[mask, "duration_days"] + int(delay_days)
    )
    revised_schedule = compute_cpm(revised_frame)
    impact = build_impact_summary(
        delayed_activity_id=activity_id,
        delay_days=int(delay_days),
        original=original_schedule,
        revised=revised_schedule,
    )
    return revised_frame, original_schedule, revised_schedule, impact


def build_impact_summary(
    delayed_activity_id: str,
    delay_days: int,
    original: ProjectSchedule,
    revised: ProjectSchedule,
) -> dict[str, Any]:
    """Compare two CPM results and list only activities whose dates changed."""
    affected: list[dict[str, Any]] = []
    for activity_id, original_item in original.activities.items():
        revised_item = revised.activities[activity_id]
        start_slip = (revised_item.earliest_start - original_item.earliest_start).days
        finish_slip = (revised_item.earliest_finish - original_item.earliest_finish).days
        float_change = revised_item.total_float_days - original_item.total_float_days
        criticality_changed = original_item.is_critical != revised_item.is_critical
        if start_slip == 0 and finish_slip == 0 and not criticality_changed:
            continue
        affected.append(
            {
                "activity_id": activity_id,
                "activity_name": original_item.activity_name,
                "tower_id": original_item.tower_id,
                "original_start": original_item.earliest_start.isoformat(),
                "revised_start": revised_item.earliest_start.isoformat(),
                "start_slip_days": start_slip,
                "original_finish": original_item.earliest_finish.isoformat(),
                "revised_finish": revised_item.earliest_finish.isoformat(),
                "finish_slip_days": finish_slip,
                "original_float_days": original_item.total_float_days,
                "revised_float_days": revised_item.total_float_days,
                "float_change_days": float_change,
                "was_critical": original_item.is_critical,
                "is_critical": revised_item.is_critical,
            }
        )

    affected.sort(key=lambda row: (row["start_slip_days"], row["activity_id"]))
    finish_slip = (revised.project_finish - original.project_finish).days
    delayed = original.activities[delayed_activity_id]
    return {
        "delayed_activity_id": delayed_activity_id,
        "delayed_activity_name": delayed.activity_name,
        "delayed_tower_id": delayed.tower_id,
        "delay_days": delay_days,
        "delay_meaning": (
            "Duration of the delayed activity increased by delay_days. "
            "Successors move only if dependency constraints require it."
        ),
        "original_project_start": original.project_start.isoformat(),
        "original_project_finish": original.project_finish.isoformat(),
        "revised_project_finish": revised.project_finish.isoformat(),
        "project_finish_slip_days": finish_slip,
        "original_critical_path": original.critical_path,
        "revised_critical_path": revised.critical_path,
        "critical_path_changed": original.critical_path != revised.critical_path,
        "original_critical_activities": original.critical_activities,
        "revised_critical_activities": revised.critical_activities,
        "affected_activity_count": len(affected),
        "affected_activities": affected,
        "limitations": [
            "No resource leveling.",
            "No recovery/optimization (overtime, extra crews, crashing).",
            "Calendar days only; weekends are not skipped.",
            "Finish-to-start links only.",
        ],
    }


def load_and_schedule(csv_path: str | Path) -> tuple[pd.DataFrame, ProjectSchedule]:
    """Convenience helper: load the CSV and compute the baseline CPM schedule."""
    frame = load_activities(csv_path)
    return frame, compute_cpm(frame)


if __name__ == "__main__":
    csv_file = Path(__file__).parent / "data" / "activities.csv"
    table, baseline = load_and_schedule(csv_file)
    print("Baseline project finish:", baseline.project_finish.isoformat())
    print("Critical path:", " -> ".join(baseline.critical_path))
    _, _, revised, impact = apply_delay(table, "T3-TIL", 5)
    print("Revised project finish:", revised.project_finish.isoformat())
    print("Project slip (days):", impact["project_finish_slip_days"])
    print("Affected activities:", impact["affected_activity_count"])
