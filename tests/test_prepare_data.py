from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_data  # noqa: E402
from prepare_data import (  # noqa: E402
    ROOT, _validate_output_dir, assign_sessions, build_observations,
    build_online_session, visitor_split,
)


class PrepareDataTest(unittest.TestCase):
    def test_first_unseen_item_after_three_events_is_target(self) -> None:
        events = pd.DataFrame({
            "visitorid": [1, 1, 1, 1, 1, 2, 2, 2],
            "session_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "event": ["view", "view", "addtocart", "view", "view", "view", "view", "view"],
            "itemid": [10, 11, 12, 12, 99, 20, 21, 22],
            "event_time": pd.to_datetime([
                "2015-01-01T00:00:00Z", "2015-01-01T00:01:00Z",
                "2015-01-01T00:02:00Z", "2015-01-01T00:03:00Z", "2015-01-01T00:04:00Z",
                "2015-01-01T00:00:00Z", "2015-01-01T00:01:00Z", "2015-01-01T00:02:00Z",
            ]),
        })
        observed, samples = build_observations(events)
        self.assertEqual(samples["target_itemid"].tolist(), [99])
        self.assertEqual(samples["target_event"].tolist(), ["view"])
        self.assertEqual(samples["target_rank"].tolist(), [5])
        self.assertEqual(observed["itemid"].tolist(), [10, 11, 12])
        self.assertTrue(set(observed["itemid"]).isdisjoint(samples["target_itemid"]))

    def test_transaction_in_observation_window_is_excluded(self) -> None:
        events = pd.DataFrame({
            "visitorid": [1, 1, 1, 1], "session_id": [1, 1, 1, 1],
            "event": ["view", "transaction", "view", "view"],
            "itemid": [1, 2, 3, 4],
            "event_time": pd.date_range("2015-01-01", periods=4, freq="min", tz="UTC"),
        })
        _, samples = build_observations(events)
        self.assertTrue(samples.empty)

    def test_thirty_minute_boundary_and_visitor_split(self) -> None:
        events = pd.DataFrame({
            "visitorid": [1, 1, 1], "event": ["view"] * 3, "itemid": [1, 2, 3],
            "event_time": pd.to_datetime([
                "2015-01-01T00:00:00Z", "2015-01-01T00:30:00Z", "2015-01-01T01:00:01Z",
            ]),
        })
        self.assertEqual(assign_sessions(events)["session_id"].tolist(), [1, 1, 2])
        visitors = pd.Series([1, 1, 2, 2, 3, 3])
        splits = visitor_split(visitors)
        grouped = pd.DataFrame({"visitor": visitors, "split": splits}).groupby("visitor")["split"].nunique()
        self.assertTrue(grouped.eq(1).all())

    def test_online_session_does_not_require_a_target(self) -> None:
        events = pd.DataFrame({
            "event": ["view", "view", "addtocart"],
            "itemid": [1, 2, 3],
            "event_time": pd.date_range("2015-01-01", periods=3, freq="min", tz="UTC"),
        })
        result = build_online_session(events, {})
        self.assertEqual(result["itemid_3"].tolist(), [3])
        self.assertEqual(result["target_itemid"].tolist(), [-1])

    def test_online_session_uses_adjacent_thirty_minute_gaps(self) -> None:
        events = pd.DataFrame({
            "event": ["view"] * 3,
            "itemid": [1, 2, 3],
            "event_time": pd.to_datetime([
                "2015-01-01T00:00:00Z", "2015-01-01T00:25:00Z", "2015-01-01T00:50:00Z",
            ]),
        })
        self.assertEqual(build_online_session(events, {})["itemid_3"].tolist(), [3])
        events.loc[1, "event_time"] = pd.Timestamp("2015-01-01T00:31:00Z")
        with self.assertRaises(ValueError):
            build_online_session(events, {})

    def test_prepare_keeps_existing_output_when_input_is_invalid(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with patch.object(prepare_data, "ROOT", root):
                with self.assertRaises(FileNotFoundError):
                    prepare_data.prepare(root / "missing-data", output)
            self.assertTrue(marker.is_file())

    def test_source_directories_are_rejected_as_generated_output(self) -> None:
        with self.assertRaises(ValueError):
            _validate_output_dir(ROOT / "tests", ROOT.parent / "data" / "archive")
        _validate_output_dir(ROOT / "tmp" / "test-artifacts", ROOT.parent / "data" / "archive")


if __name__ == "__main__":
    unittest.main()
