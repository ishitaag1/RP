"""Tests for the optional assistant: allow-listed tools only, no live Ollama required."""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock, patch

import agent
import scheduler as sch
from agent import ScheduleAssistant


class TestScheduleAssistantTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assistant = ScheduleAssistant()

    def test_project_status_matches_scheduler_baseline(self) -> None:
        status = self.assistant.get_project_status()
        self.assertTrue(status["ok"])
        self.assertEqual(status["source"], "scheduler.compute_cpm")
        self.assertEqual(status["project_finish"], "2026-05-08")
        self.assertEqual(status["activity_count"], 31)

    def test_critical_path_uses_scheduler(self) -> None:
        result = self.assistant.get_critical_path()
        self.assertEqual(
            result["critical_path"],
            [
                "T3-CLR",
                "T3-EXC",
                "T3-FOU",
                "T3-STR",
                "T3-BRK",
                "T3-PLS",
                "T3-TIL",
                "T3-PNT",
                "T3-HND",
                "T0-EXT",
            ],
        )

    def test_find_activity_disambiguates_towers(self) -> None:
        all_tiling = self.assistant.find_activity("Tiling")
        self.assertEqual(all_tiling["match_count"], 3)
        t3 = self.assistant.find_activity("Tiling", "T3")
        self.assertEqual(t3["matches"][0]["activity_id"], "T3-TIL")

    def test_analyze_delay_critical_activity(self) -> None:
        impact = self.assistant.dispatch_tool(
            "analyze_delay",
            {"activity_id": "T3-TIL", "delay_days": 5},
        )
        self.assertTrue(impact["ok"])
        self.assertEqual(impact["source"], "scheduler.apply_delay")
        self.assertEqual(impact["original_project_finish"], "2026-05-08")
        self.assertEqual(impact["revised_project_finish"], "2026-05-13")
        self.assertEqual(impact["project_finish_slip_days"], 5)
        affected_ids = {row["activity_id"] for row in impact["affected_activities"]}
        self.assertIn("T3-PNT", affected_ids)
        self.assertIn("T0-EXT", affected_ids)
        self.assertNotIn("T1-HND", affected_ids)

    def test_analyze_delay_non_critical_absorbed_by_float(self) -> None:
        impact = self.assistant.dispatch_tool(
            "analyze_delay",
            {"activity_id": "T1-TIL", "delay_days": 5},
        )
        self.assertTrue(impact["ok"])
        self.assertEqual(impact["project_finish_slip_days"], 0)

    def test_invalid_activity_id_is_structured_error(self) -> None:
        result = self.assistant.dispatch_tool(
            "analyze_delay",
            {"activity_id": "DOES-NOT-EXIST", "delay_days": 1},
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "InvalidActivityIdError")

    def test_negative_delay_is_structured_error(self) -> None:
        result = self.assistant.dispatch_tool(
            "analyze_delay",
            {"activity_id": "T3-TIL", "delay_days": -2},
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "NegativeDelayError")

    def test_unknown_tool_is_rejected(self) -> None:
        with self.assertRaises(agent.UnknownToolError):
            self.assistant.dispatch_tool("os_system", {"command": "dir"})

    def test_delay_days_must_be_integer(self) -> None:
        result = self.assistant.dispatch_tool(
            "analyze_delay",
            {"activity_id": "T3-TIL", "delay_days": "five"},
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "ScheduleError")

    def test_get_activity_details(self) -> None:
        details = self.assistant.dispatch_tool(
            "get_activity_details", {"activity_id": "T3-TIL"}
        )
        self.assertTrue(details["ok"])
        self.assertEqual(details["earliest_start"], date(2026, 4, 19).isoformat())
        self.assertTrue(details["is_critical"])


class TestOllamaFallback(unittest.TestCase):
    def test_check_ollama_status_when_server_is_down(self) -> None:
        with patch("agent.requests.get", side_effect=agent.requests.ConnectionError("refused")):
            status = agent.check_ollama_status()
        self.assertFalse(status["available"])
        self.assertFalse(status["model_ready"])
        self.assertIn("ollama pull", status["message"].lower())

    def test_ask_raises_unavailable_when_model_missing(self) -> None:
        fake_status = {
            "available": True,
            "model_ready": False,
            "message": "model missing",
        }
        assistant = ScheduleAssistant()
        with patch("agent.check_ollama_status", return_value=fake_status):
            with self.assertRaises(agent.AgentUnavailableError):
                assistant.ask("What is the completion date?")

    def test_ask_does_not_claim_tools_when_none_ran(self) -> None:
        fake_status = {
            "available": True,
            "model_ready": True,
            "base_url": "http://127.0.0.1:11434",
            "model": "llama3.2",
            "message": "ready",
        }
        fake_response = Mock()
        fake_response.raise_for_status = lambda: None
        fake_response.json.return_value = {
            "message": {"role": "assistant", "content": "I guessed 1 June.", "tool_calls": []}
        }
        assistant = ScheduleAssistant()
        with patch("agent.check_ollama_status", return_value=fake_status):
            with patch("agent.requests.post", return_value=fake_response):
                result = assistant.ask("When does the project finish?")
        self.assertFalse(result["grounded"])
        self.assertEqual(result["tools_used"], [])
        self.assertIn("1 June", result["answer"])


class TestSchedulerStillOwnsMath(unittest.TestCase):
    def test_assistant_does_not_mutate_baseline_frame(self) -> None:
        assistant = ScheduleAssistant()
        before = assistant.frame.copy(deep=True)
        assistant.analyze_delay("T3-TIL", 5)
        import pandas as pd

        pd.testing.assert_frame_equal(assistant.frame, before)
        self.assertEqual(assistant.baseline.project_finish, date(2026, 5, 8))

    def test_missing_predecessor_still_comes_from_scheduler(self) -> None:
        frame, _ = sch.load_and_schedule(agent.DEFAULT_CSV)
        idx = frame.index[frame["activity_id"] == "T1-EXC"][0]
        frame.at[idx, "predecessors"] = ["MISSING"]
        with self.assertRaises(sch.MissingPredecessorError):
            sch.compute_cpm(frame)


if __name__ == "__main__":
    unittest.main()
