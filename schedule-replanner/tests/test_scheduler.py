"""
Unit tests for scheduler.py.

The first tests use a tiny 4-activity network you can check on paper.
Then we run the same rules on data/activities.csv.
"""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

import pandas as pd

import scheduler as sch

HAND_EPOCH_START = date(2026, 3, 1)


def four_activity_frame() -> pd.DataFrame:
    """
    Hand-calculated network (offsets from 2026-03-01):

        A (3 days)
       / \\
      B(2) C(4)
       \\ /
        D (2)   D waits for BOTH B and C

    Forward pass:
      A: ES=0, exclusive EF=3, inclusive finish=2  (Mar 1-3)
      B: ES=3, exclusive EF=5, inclusive finish=4  (Mar 4-5)
      C: ES=3, exclusive EF=7, inclusive finish=6  (Mar 4-7)
      D: ES=max(5,7)=7, exclusive EF=9, inclusive finish=8  (Mar 8-9)

    Backward pass (project exclusive finish = 9):
      D: LF=9, LS=7, float=0
      C: LF=7, LS=3, float=0
      B: LF=7, LS=5, float=2
      A: LF=min(5,3)=3, LS=0, float=0

    Critical path: A -> C -> D
    Project inclusive finish: 2026-03-09
    """
    return pd.DataFrame(
        {
            "activity_id": ["A", "B", "C", "D"],
            "activity_name": ["Start", "Short branch", "Long branch", "Merge"],
            "tower_id": ["T0", "T0", "T0", "T0"],
            "predecessors": [[], ["A"], ["A"], ["B", "C"]],
            "duration_days": [3, 2, 4, 2],
            "baseline_start": [
                HAND_EPOCH_START,
                date(2026, 3, 4),
                date(2026, 3, 4),
                date(2026, 3, 8),
            ],
        }
    )


class TestHandCalculatedNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = four_activity_frame()
        self.schedule = sch.compute_cpm(self.frame)

    def test_forward_and_backward_dates(self) -> None:
        a = self.schedule.activities["A"]
        b = self.schedule.activities["B"]
        c = self.schedule.activities["C"]
        d = self.schedule.activities["D"]

        self.assertEqual(a.earliest_start, date(2026, 3, 1))
        self.assertEqual(a.earliest_finish, date(2026, 3, 3))
        self.assertEqual(b.earliest_start, date(2026, 3, 4))
        self.assertEqual(b.earliest_finish, date(2026, 3, 5))
        self.assertEqual(c.earliest_start, date(2026, 3, 4))
        self.assertEqual(c.earliest_finish, date(2026, 3, 7))
        self.assertEqual(d.earliest_start, date(2026, 3, 8))
        self.assertEqual(d.earliest_finish, date(2026, 3, 9))

        self.assertEqual(a.total_float_days, 0)
        self.assertEqual(b.total_float_days, 2)
        self.assertEqual(c.total_float_days, 0)
        self.assertEqual(d.total_float_days, 0)

    def test_critical_path_is_a_c_d(self) -> None:
        self.assertEqual(self.schedule.critical_path, ["A", "C", "D"])
        self.assertEqual(self.schedule.project_finish, date(2026, 3, 9))
        self.assertEqual(self.schedule.project_duration_days, 9)

    def test_delay_inside_float_does_not_move_successor(self) -> None:
        """B has 2 days of float. Adding 1 day to B must not move D or the finish."""
        _, original, revised, impact = sch.apply_delay(self.frame, "B", 1)
        self.assertEqual(revised.activities["D"].earliest_start, date(2026, 3, 8))
        self.assertEqual(revised.project_finish, original.project_finish)
        self.assertEqual(impact["project_finish_slip_days"], 0)
        delayed_ids = {row["activity_id"] for row in impact["affected_activities"]}
        self.assertIn("B", delayed_ids)
        self.assertNotIn("D", delayed_ids)

    def test_delay_beyond_float_moves_only_what_constraints_require(self) -> None:
        """
        B duration 2 -> 5. New exclusive EF of B is 3+5=8.
        D then starts on offset 8 (Mar 9), not 7+3 blindly applied to every descendant.
        """
        _, original, revised, impact = sch.apply_delay(self.frame, "B", 3)
        self.assertEqual(revised.activities["B"].earliest_finish, date(2026, 3, 8))
        self.assertEqual(revised.activities["D"].earliest_start, date(2026, 3, 9))
        self.assertEqual(revised.activities["D"].earliest_finish, date(2026, 3, 10))
        self.assertEqual(revised.project_finish, date(2026, 3, 10))
        self.assertEqual(impact["project_finish_slip_days"], 1)
        self.assertEqual(original.critical_path, ["A", "C", "D"])
        self.assertEqual(revised.critical_path, ["A", "B", "D"])

    def test_delay_on_critical_activity_slips_project_by_same_amount(self) -> None:
        _, _, revised, impact = sch.apply_delay(self.frame, "C", 2)
        self.assertEqual(revised.project_finish, date(2026, 3, 11))
        self.assertEqual(impact["project_finish_slip_days"], 2)
        self.assertEqual(revised.activities["D"].earliest_start, date(2026, 3, 10))

    def test_original_frame_is_not_mutated(self) -> None:
        snapshot = self.frame.copy(deep=True)
        sch.apply_delay(self.frame, "C", 2)
        pd.testing.assert_frame_equal(self.frame, snapshot)


class TestValidation(unittest.TestCase):
    def test_unknown_activity_id(self) -> None:
        with self.assertRaises(sch.InvalidActivityIdError):
            sch.apply_delay(four_activity_frame(), "DOES-NOT-EXIST", 1)

    def test_negative_delay(self) -> None:
        with self.assertRaises(sch.NegativeDelayError):
            sch.apply_delay(four_activity_frame(), "A", -1)

    def test_missing_predecessor(self) -> None:
        frame = four_activity_frame()
        frame.loc[frame["activity_id"] == "B", "predecessors"] = pd.Series(
            [["MISSING"]], index=frame.loc[frame["activity_id"] == "B"].index
        )
        with self.assertRaises(sch.MissingPredecessorError):
            sch.compute_cpm(frame)

    def test_cycle_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "activity_id": ["X", "Y"],
                "activity_name": ["One", "Two"],
                "tower_id": ["T0", "T0"],
                "predecessors": [["Y"], ["X"]],
                "duration_days": [2, 2],
                "baseline_start": [HAND_EPOCH_START, HAND_EPOCH_START],
            }
        )
        with self.assertRaises(sch.DependencyCycleError):
            sch.compute_cpm(frame)

    def test_zero_duration_is_rejected(self) -> None:
        csv_text = (
            "activity_id,activity_name,tower_id,predecessors,duration_days,baseline_start\n"
            "Z,Zero,T0,,0,2026-03-01\n"
        )
        path = Path(__file__).parent / "_tmp_zero.csv"
        path.write_text(csv_text, encoding="utf-8")
        try:
            with self.assertRaises(sch.InvalidDurationError):
                sch.load_activities(path)
        finally:
            path.unlink(missing_ok=True)

    def test_duplicate_ids_rejected(self) -> None:
        csv_text = (
            "activity_id,activity_name,tower_id,predecessors,duration_days,baseline_start\n"
            "T1-TIL,Tiling,T1,,3,2026-03-01\n"
            "T1-TIL,Tiling,T1,,3,2026-03-01\n"
        )
        path = Path(__file__).parent / "_tmp_dup.csv"
        path.write_text(csv_text, encoding="utf-8")
        try:
            with self.assertRaises(sch.DuplicateActivityIdError):
                sch.load_activities(path)
        finally:
            path.unlink(missing_ok=True)


class TestProjectCsv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.csv_path = Path(__file__).resolve().parents[1] / "data" / "activities.csv"
        cls.frame, cls.baseline = sch.load_and_schedule(cls.csv_path)

    def test_same_name_different_towers_have_unique_ids(self) -> None:
        tiling_ids = sch.find_activity_ids(self.frame, "Tiling")
        self.assertEqual(sorted(tiling_ids), ["T1-TIL", "T2-TIL", "T3-TIL"])
        t3_only = sch.find_activity_ids(self.frame, "Tiling", tower_id="T3")
        self.assertEqual(t3_only, ["T3-TIL"])

    def test_baseline_matches_hand_calculated_t3_and_project_finish(self) -> None:
        t3_til = self.baseline.activities["T3-TIL"]
        t3_hnd = self.baseline.activities["T3-HND"]
        ext = self.baseline.activities["T0-EXT"]
        self.assertEqual(t3_til.earliest_start, date(2026, 4, 19))
        self.assertEqual(t3_til.earliest_finish, date(2026, 4, 26))
        self.assertEqual(t3_hnd.earliest_finish, date(2026, 5, 3))
        self.assertEqual(ext.earliest_start, date(2026, 5, 4))
        self.assertEqual(ext.earliest_finish, date(2026, 5, 8))
        self.assertEqual(self.baseline.project_finish, date(2026, 5, 8))
        self.assertTrue(t3_til.is_critical)
        self.assertEqual(
            self.baseline.critical_path,
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

    def test_t3_tiling_delay_of_five_days(self) -> None:
        original_frame = self.frame.copy(deep=True)
        revised_frame, original, revised, impact = sch.apply_delay(
            self.frame, "T3-TIL", 5
        )
        self.assertEqual(original.project_finish, date(2026, 5, 8))
        self.assertEqual(revised.project_finish, date(2026, 5, 13))
        self.assertEqual(impact["project_finish_slip_days"], 5)
        self.assertEqual(revised.activities["T3-TIL"].earliest_finish, date(2026, 5, 1))
        self.assertEqual(revised.activities["T3-PNT"].earliest_start, date(2026, 5, 2))
        self.assertEqual(revised.activities["T3-HND"].earliest_start, date(2026, 5, 7))
        self.assertEqual(revised.activities["T0-EXT"].earliest_start, date(2026, 5, 9))
        # Unrelated tower handover should not move.
        self.assertEqual(
            revised.activities["T1-HND"].earliest_finish,
            original.activities["T1-HND"].earliest_finish,
        )
        pd.testing.assert_frame_equal(self.frame, original_frame)
        self.assertEqual(int(revised_frame.loc[revised_frame["activity_id"] == "T3-TIL", "duration_days"].iloc[0]), 13)

    def test_non_critical_delay_does_not_slip_project(self) -> None:
        _, original, revised, impact = sch.apply_delay(self.frame, "T1-TIL", 5)
        self.assertEqual(revised.project_finish, original.project_finish)
        self.assertEqual(impact["project_finish_slip_days"], 0)
        self.assertEqual(revised.activities["T1-TIL"].earliest_finish, date(2026, 4, 20))
        self.assertEqual(revised.activities["T1-HND"].earliest_finish, date(2026, 4, 26))


if __name__ == "__main__":
    unittest.main()
