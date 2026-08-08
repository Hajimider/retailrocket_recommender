from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_generation import add_source_labels, build_recall_maps, generate_candidates  # noqa: E402
from train_models import lightgbm_model, ranking_metrics  # noqa: E402


def session(sample_id: int, items: tuple[int, int, int], target: int, split: str) -> dict[str, object]:
    return {
        "sample_id": sample_id, "visitorid": sample_id, "session_id": sample_id,
        "target_itemid": target, "target_event": "view", "split": split,
        "n_view": 3, "n_cart": 0, "n_unique_items": len(set(items)),
        "duration_sec": 120.0, "mean_gap_sec": 40.0, "max_gap_sec": 60.0,
        "start_hour": 10, "weekday": 0, "is_weekend": 0,
        "event_1": "view", "event_2": "view", "event_3": "view",
        "itemid_1": items[0], "itemid_2": items[1], "itemid_3": items[2],
        "categoryid_1": -1, "categoryid_2": -1, "categoryid_3": -1,
        "gap_sec_1": 0.0, "gap_sec_2": 60.0, "gap_sec_3": 60.0,
    }


class CandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = {"item_category": {}, "category_depths": {}}
        self.train = pd.DataFrame([session(1, (1, 2, 3), 99, "train")])
        self.maps = build_recall_maps(self.train, self.catalog)

    def test_target_does_not_enter_recall_maps_or_validation_candidates(self) -> None:
        self.assertNotIn(99, self.maps["item_popularity"])
        validation = pd.DataFrame([session(2, (10, 11, 12), 999, "val")])
        candidates = generate_candidates(validation, self.maps, max_candidates=10)
        self.assertNotIn(999, candidates["candidate_itemid"].tolist())
        self.assertTrue(set(candidates["candidate_itemid"]).isdisjoint({10, 11, 12}))
        self.assertIn("candidate_categoryid", candidates.columns)
        self.assertIn("related_itemid", candidates.columns)
        self.assertEqual(int(candidates["label"].sum()), 0)

    def test_training_target_is_not_injected(self) -> None:
        candidates = generate_candidates(self.train, self.maps, max_candidates=10)
        if candidates.empty:
            self.assertTrue(candidates.empty)
        else:
            self.assertEqual(int(candidates["label"].sum()), 0)
            self.assertNotIn(99, candidates["candidate_itemid"].tolist())
            self.assertTrue(add_source_labels(candidates)["source"].str.len().gt(0).all())

    def test_ranking_metrics_count_missing_target_as_zero(self) -> None:
        frame = pd.DataFrame({
            "sample_id": [1, 1, 2, 2],
            "candidate_itemid": [10, 11, 20, 21],
            "label": [0, 1, 0, 0],
        })
        metrics = ranking_metrics(frame, np.array([0.1, 0.9, 0.8, 0.2]), {10, 11, 20, 21})
        self.assertEqual(metrics["candidate_recall"], 0.5)
        self.assertEqual(metrics["recall_at_5"], 0.5)
        self.assertEqual(metrics["mrr_at_10"], 0.5)
        self.assertEqual(metrics["coverage_at_10"], 1.0)

    def test_lightgbm_row_sampling_is_enabled(self) -> None:
        model = lightgbm_model({"subsample": 0.8}, seed=42)
        self.assertEqual(model.get_params()["subsample_freq"], 1)


if __name__ == "__main__":
    unittest.main()
