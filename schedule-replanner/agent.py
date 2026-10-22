"""
Optional LLM assistant that talks to the CPM engine through a fixed tool list.

Scheduling math always happens in scheduler.py. This module never recalculates
dates itself, never executes model-generated Python, and never requires a
paid API key.

Default backend: a local Ollama server. If Ollama is not running, the model
is missing, or the request fails, callers get a clear setup message instead
of a crash.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

import scheduler as sch

load_dotenv()

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "activities.csv"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"
MAX_TOOL_ROUNDS = 6

SETUP_INSTRUCTIONS = (
    "The optional AI assistant is not available right now. "
    "The scheduler and dashboard still work without it.\n\n"
    "To enable the assistant on Windows:\n"
    "1. Install Ollama from https://ollama.com/download\n"
    "2. In PowerShell run: ollama pull llama3.2\n"
    "3. Keep Ollama running, then restart the Streamlit app.\n"
    "4. Optional: copy .env.example to .env and set OLLAMA_MODEL."
)


class AgentError(Exception):
    """Base error for the optional assistant."""


class AgentUnavailableError(AgentError):
    """Raised when Ollama cannot be reached or the model is missing."""


class UnknownToolError(AgentError):
    """Raised when the model asks for a tool that is not on the allow-list."""


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_project_status",
            "description": (
                "Return baseline project start, finish, duration, activity "
                "counts, and the current critical path. Use this for questions "
                "about completion date or how many activities exist."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_critical_path",
            "description": (
                "Return the critical-path activity IDs and names, plus every "
                "zero-float activity."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_activity",
            "description": (
                "Look up unique activity IDs from a human name such as Tiling. "
                "Pass tower_id (T1, T2, T3, SITE) when the name is used in "
                "more than one tower."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_name": {
                        "type": "string",
                        "description": "Activity name, for example Tiling.",
                    },
                    "tower_id": {
                        "type": "string",
                        "description": "Optional tower filter such as T3.",
                    },
                },
                "required": ["activity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_activity_details",
            "description": "Return dates, float, predecessors, and criticality for one activity_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": "Unique ID such as T3-TIL.",
                    }
                },
                "required": ["activity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_delay",
            "description": (
                "Apply a delay to one activity_id by increasing its duration, "
                "recalculate CPM, and return the impact summary: project finish "
                "slip, critical-path change, and affected downstream activities. "
                "delay_days must be zero or a positive integer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": "Unique ID such as T3-TIL.",
                    },
                    "delay_days": {
                        "type": "integer",
                        "description": "Whole days to add to the activity duration.",
                    },
                },
                "required": ["activity_id", "delay_days"],
            },
        },
    },
]


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def ollama_model_name() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def ollama_timeout_seconds() -> int:
    raw = os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")
    try:
        return max(5, int(raw))
    except ValueError:
        return 60


def check_ollama_status(
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    Probe the local Ollama server. Never raises for a down server; returns a
    status dictionary the UI can display.
    """
    url = (base_url or ollama_base_url()).rstrip("/")
    wanted = model or ollama_model_name()
    wait = timeout if timeout is not None else min(5, ollama_timeout_seconds())
    try:
        response = requests.get(f"{url}/api/tags", timeout=wait)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {
            "available": False,
            "model_ready": False,
            "base_url": url,
            "model": wanted,
            "installed_models": [],
            "message": SETUP_INSTRUCTIONS,
            "detail": str(exc),
        }

    models = []
    for item in payload.get("models", []):
        name = str(item.get("name", ""))
        if name:
            models.append(name)
    model_ready = any(
        name == wanted or name.startswith(f"{wanted}:") for name in models
    )
    if not model_ready:
        message = (
            f"Ollama is running at {url}, but model '{wanted}' was not found.\n"
            f"Installed models: {models or 'none'}\n"
            f"In PowerShell run: ollama pull {wanted}"
        )
    else:
        message = f"Ollama is ready ({wanted} at {url})."
    return {
        "available": True,
        "model_ready": model_ready,
        "base_url": url,
        "model": wanted,
        "installed_models": models,
        "message": message if model_ready else message,
        "detail": None,
    }


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise sch.ScheduleError(f"{field_name} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise sch.ScheduleError(f"{field_name} must be an integer, got {value!r}.")


def _as_text(value: Any, field_name: str) -> str:
    if value is None:
        raise sch.ScheduleError(f"{field_name} is required.")
    text = str(value).strip()
    if not text:
        raise sch.ScheduleError(f"{field_name} is required.")
    return text


class ScheduleAssistant:
    """Loads the activity CSV once and answers only through allow-listed tools."""

    def __init__(self, csv_path: str | Path = DEFAULT_CSV) -> None:
        self.csv_path = Path(csv_path)
        self.frame, self.baseline = sch.load_and_schedule(self.csv_path)

    def get_project_status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "source": "scheduler.compute_cpm",
            "project_start": self.baseline.project_start.isoformat(),
            "project_finish": self.baseline.project_finish.isoformat(),
            "project_duration_days": self.baseline.project_duration_days,
            "activity_count": len(self.baseline.activities),
            "critical_activity_count": len(self.baseline.critical_activities),
            "critical_path": self.baseline.critical_path,
            "calendar_note": (
                "Dates are calendar days. Weekends are not skipped. "
                "This is a demo network inspired by typical tower sequences, "
                "not a live Godrej Properties programme."
            ),
        }

    def get_critical_path(self) -> dict[str, Any]:
        details = []
        for activity_id in self.baseline.critical_path:
            item = self.baseline.activities[activity_id]
            details.append(
                {
                    "activity_id": activity_id,
                    "activity_name": item.activity_name,
                    "tower_id": item.tower_id,
                    "earliest_start": item.earliest_start.isoformat(),
                    "earliest_finish": item.earliest_finish.isoformat(),
                    "total_float_days": item.total_float_days,
                }
            )
        return {
            "ok": True,
            "source": "scheduler.extract_critical_path",
            "critical_path": self.baseline.critical_path,
            "critical_activities": self.baseline.critical_activities,
            "critical_path_details": details,
        }

    def find_activity(self, activity_name: str, tower_id: str | None = None) -> dict[str, Any]:
        name = _as_text(activity_name, "activity_name")
        tower = _as_text(tower_id, "tower_id") if tower_id else None
        matches = sch.find_activity_ids(self.frame, name, tower)
        rows = []
        for activity_id in matches:
            item = self.baseline.activities[activity_id]
            rows.append(
                {
                    "activity_id": activity_id,
                    "activity_name": item.activity_name,
                    "tower_id": item.tower_id,
                    "is_critical": item.is_critical,
                    "total_float_days": item.total_float_days,
                }
            )
        return {
            "ok": True,
            "source": "scheduler.find_activity_ids",
            "query_name": name,
            "query_tower_id": tower,
            "match_count": len(rows),
            "matches": rows,
            "note": (
                "If several towers match, call analyze_delay with a unique "
                "activity_id such as T3-TIL."
            ),
        }

    def get_activity_details(self, activity_id: str) -> dict[str, Any]:
        wanted = _as_text(activity_id, "activity_id")
        item = self.baseline.activities.get(wanted)
        if item is None:
            raise sch.InvalidActivityIdError(
                f"Unknown activity_id '{wanted}'. Use the unique ID (for example T3-TIL)."
            )
        return {
            "ok": True,
            "source": "scheduler.compute_cpm",
            "activity_id": item.activity_id,
            "activity_name": item.activity_name,
            "tower_id": item.tower_id,
            "predecessors": item.predecessors,
            "duration_days": item.duration_days,
            "baseline_start": item.baseline_start.isoformat(),
            "earliest_start": item.earliest_start.isoformat(),
            "earliest_finish": item.earliest_finish.isoformat(),
            "latest_start": item.latest_start.isoformat(),
            "latest_finish": item.latest_finish.isoformat(),
            "total_float_days": item.total_float_days,
            "is_critical": item.is_critical,
        }

    def analyze_delay(self, activity_id: str, delay_days: int) -> dict[str, Any]:
        wanted = _as_text(activity_id, "activity_id")
        days = _as_int(delay_days, "delay_days")
        _, _, _, impact = sch.apply_delay(self.frame, wanted, days)
        impact["ok"] = True
        impact["source"] = "scheduler.apply_delay"
        impact["why_finish_changed"] = (
            "The project finish moved because at least one delayed activity "
            "had too little total float to absorb the extra duration. "
            "Successors moved only where finish-to-start links required it."
            if impact["project_finish_slip_days"] > 0
            else (
                "The project finish did not move. The delayed activity (and any "
                "moved successors) still had enough total float to absorb the slip."
            )
        )
        return impact

    def dispatch_tool(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """
        Run one allow-listed tool. Unknown names are rejected. Arguments are
        validated; scheduler exceptions are returned as structured errors so
        the chat loop does not crash.
        """
        handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "get_project_status": lambda: self.get_project_status(),
            "get_critical_path": lambda: self.get_critical_path(),
            "find_activity": lambda: self.find_activity(
                arguments.get("activity_name") if arguments else None,
                arguments.get("tower_id") if arguments else None,
            ),
            "get_activity_details": lambda: self.get_activity_details(
                (arguments or {}).get("activity_id")
            ),
            "analyze_delay": lambda: self.analyze_delay(
                (arguments or {}).get("activity_id"),
                (arguments or {}).get("delay_days"),
            ),
        }
        if name not in handlers:
            raise UnknownToolError(
                f"Tool '{name}' is not allowed. Permitted tools: {sorted(handlers)}."
            )
        try:
            return handlers[name]()
        except sch.ScheduleError as exc:
            return {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source": "scheduler",
            }

    def ask(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Send a question to Ollama with tools. Returns the model text plus the
        list of tools that actually ran. If no tool ran, the answer is marked
        as not grounded in the scheduler.
        """
        status = check_ollama_status()
        if not status["available"] or not status["model_ready"]:
            raise AgentUnavailableError(status["message"])

        text = question.strip()
        if not text:
            raise AgentError("Please enter a question.")

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a construction planning assistant for a demo CPM "
                    "schedule. You must call tools to read dates, the critical "
                    "path, or delay impact. Never invent activity IDs or dates. "
                    "If a tool returns ok=false, explain that error. "
                    "If several activities share a name, ask for the unique ID. "
                    "Keep answers short and interview-friendly. "
                    "Do not claim you changed the dashboard; tools only compute."
                ),
            }
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": text})

        tools_used: list[dict[str, Any]] = []
        final_text = ""
        url = f"{status['base_url']}/api/chat"

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = requests.post(
                    url,
                    json={
                        "model": status["model"],
                        "messages": messages,
                        "tools": TOOL_DEFINITIONS,
                        "stream": False,
                    },
                    timeout=ollama_timeout_seconds(),
                )
                response.raise_for_status()
                body = response.json()
            except requests.RequestException as exc:
                raise AgentUnavailableError(
                    f"Ollama request failed: {exc}\n\n{SETUP_INSTRUCTIONS}"
                ) from exc

            message = body.get("message") or {}
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_text = (message.get("content") or "").strip()
                break

            for call in tool_calls:
                function = call.get("function") or {}
                tool_name = str(function.get("name") or "")
                raw_args = function.get("arguments") or {}
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError:
                        raw_args = {}
                if not isinstance(raw_args, dict):
                    raw_args = {}
                try:
                    result = self.dispatch_tool(tool_name, raw_args)
                    error = None
                except UnknownToolError as exc:
                    result = {"ok": False, "error": str(exc)}
                    error = str(exc)
                tools_used.append(
                    {
                        "name": tool_name,
                        "arguments": raw_args,
                        "ok": bool(result.get("ok")),
                        "error": error or result.get("error"),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result, default=str),
                    }
                )
        else:
            final_text = (
                "The model kept requesting tools. Showing the last tool results "
                "instead of an invented summary."
            )

        return {
            "answer": final_text
            or "The model did not return an explanation. See the tool results.",
            "tools_used": tools_used,
            "grounded": any(item.get("ok") for item in tools_used),
            "model": status["model"],
        }
