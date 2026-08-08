from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import score_api  # noqa: E402
from score_batch import load_bundle, predict_scores, recommend_session, topk_recommendations  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
GENERATED_OUTPUTS = all((ROOT / path).is_file() for path in (
    "artifacts/catalog.joblib", "outputs/model_registry.json", "outputs/recall_maps.joblib",
)) and "event_time_1" in pd.read_parquet(ROOT / "artifacts" / "recommendation_sessions.parquet").columns if (ROOT / "artifacts" / "recommendation_sessions.parquet").is_file() else False


class ScoringTest(unittest.TestCase):
    def test_predict_scores_supports_torch_bundle(self) -> None:
        candidates = pd.DataFrame({feature: [0.0, 1.0] for feature in __import__("candidate_generation").RANK_FEATURES})
        candidates["candidate_itemid"] = [10, 11]
        bundle = {
            "registry": {"model_type": "deepfm"},
            "model": type("MockBundle", (), {"predict": lambda self, frame: np.array([0.2, 0.8])})(),
        }
        np.testing.assert_allclose(predict_scores(bundle, candidates), [0.2, 0.8])

    def test_topk_schema_and_order(self) -> None:
        candidates = pd.DataFrame({
            "sample_id": [1, 1, 1], "visitorid": [1] * 3, "session_id": [1] * 3,
            "candidate_itemid": [10, 11, 12], "target_itemid": [12] * 3,
            "label": [0, 0, 1], "split": ["test"] * 3,
            "candidate_categoryid": [3, 4, 5], "related_itemid": [-1, 10, 10],
            "source_covisit": [0, 1, 1],
            "source_category": [0, 0, 0], "source_popular": [0, 0, 0],
        })
        result = topk_recommendations(candidates, np.array([0.1, 0.9, 0.2]), 2)
        self.assertEqual(result["candidate_itemid"].tolist(), [11, 12])
        self.assertEqual(result["rank"].tolist(), [1, 2])
        self.assertTrue(result["relative_score"].between(0, 1).all())

    def test_api_validates_input_and_returns_recommendations(self) -> None:
        recommendation = pd.DataFrame({
            "rank": [1], "candidate_itemid": [42], "candidate_categoryid": [7],
            "related_itemid": [11], "relative_score": [0.7], "source": ["covisit"]
        })
        bundle = {"registry": {"selected_model": "LR"}}
        with (
            patch.object(score_api.joblib, "load", return_value={"item_category": {}}),
            patch.object(score_api, "load_bundle", return_value=bundle),
            patch.object(score_api, "recommend_session", return_value=recommendation),
        ):
            client = score_api.create_app().test_client()
            examples = client.get("/examples")
            payload = {"events": [
                {"event": "view", "itemid": 1, "timestamp": "2015-01-01T00:00:00Z"},
                {"event": "view", "itemid": 2, "timestamp": "2015-01-01T00:01:00Z"},
                {"event": "addtocart", "itemid": 3, "timestamp": "2015-01-01T00:02:00Z"},
            ], "top_k": 10}
            response = client.post("/recommend", json=payload)
            invalid = client.post("/recommend", json={"events": []})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(examples.status_code, 200)
        self.assertEqual(response.get_json()["recommendations"][0]["itemid"], 42)
        self.assertEqual(response.get_json()["recommendations"][0]["categoryid"], 7)
        self.assertEqual(response.get_json()["recommendations"][0]["related_itemid"], 11)
        self.assertEqual(invalid.status_code, 400)


@unittest.skipUnless(GENERATED_OUTPUTS, "requires generated training artifacts")
class GeneratedArtifactIntegrationTest(unittest.TestCase):
    def test_real_api_matches_direct_topk_recommendation(self) -> None:
        sample = pd.read_parquet(ROOT / "artifacts" / "recommendation_sessions.parquet").loc[
            lambda frame: frame["split"].eq("test")
        ].iloc[0]
        payload = {"events": [
            {"event": str(sample[f"event_{i}"],), "itemid": int(sample[f"itemid_{i}"]),
             "timestamp": pd.Timestamp(sample[f"event_time_{i}"]).isoformat()}
            for i in (1, 2, 3)
        ], "top_k": 5}
        events, _ = score_api._event_frame(payload)
        catalog = score_api.joblib.load(ROOT / "artifacts" / "catalog.joblib")
        bundle = load_bundle(ROOT / "outputs")
        direct = recommend_session(events, catalog, bundle, 5)
        response = score_api.create_app().test_client().post("/recommend", json=payload)
        api_rows = response.get_json()["recommendations"]
        api_items = [row["itemid"] for row in api_rows]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(api_items, direct["candidate_itemid"].tolist())
        self.assertEqual(len(set(api_items)), 5)
        np.testing.assert_allclose(
            [row["score"] for row in api_rows], direct["relative_score"].to_numpy(), atol=1e-6
        )
        self.assertEqual([row["source"] for row in api_rows], direct["source"].tolist())
        self.assertTrue(set(api_items).isdisjoint({int(sample[f"itemid_{i}"]) for i in (1, 2, 3)}))


if __name__ == "__main__":
    unittest.main()
