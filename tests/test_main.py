from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import retailrocket_main  # noqa: E402


class MainEntryTest(unittest.TestCase):
    def test_preflight_rejects_data_inside_generated_directory_before_cleanup(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            data = artifacts / "raw"
            data.mkdir(parents=True)
            marker = artifacts / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with patch.object(retailrocket_main, "ROOT", root):
                with self.assertRaises(ValueError):
                    retailrocket_main.run_pipeline(data_dir=data)
            self.assertTrue(marker.is_file())

    def test_preflight_rejects_small_candidate_count_before_cleanup(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            marker = artifacts / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with patch.object(retailrocket_main, "ROOT", root):
                with self.assertRaises(ValueError):
                    retailrocket_main.run_pipeline(data_dir=root / "data", top_k=5, max_candidates=5)
            self.assertTrue(marker.is_file())

    def test_failed_preparation_keeps_existing_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            for name in ("events.csv", "item_properties_part1.csv", "item_properties_part2.csv"):
                (data / name).touch()
            markers = []
            for name in ("artifacts", "outputs", "tmp"):
                directory = root / name
                directory.mkdir()
                marker = directory / "keep.txt"
                marker.write_text("keep", encoding="utf-8")
                markers.append(marker)
            with (
                patch.object(retailrocket_main, "ROOT", root),
                patch.object(retailrocket_main, "prepare", side_effect=ValueError("invalid data")),
            ):
                with self.assertRaises(ValueError):
                    retailrocket_main.run_pipeline(data_dir=data)
            self.assertTrue(all(marker.is_file() for marker in markers))

    def test_reuse_requires_catalog_recall_maps_and_selected_model(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "artifacts/recommendation_sessions.parquet",
                "artifacts/candidate_test.parquet",
                "artifacts/catalog.joblib",
                "artifacts/manifest.json",
                "outputs/recall_maps.joblib",
                "outputs/batch_recommendations.csv",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            registry = {"model_path": "models/final.joblib", "pipeline_version": retailrocket_main.PIPELINE_VERSION}
            (root / "outputs/model_registry.json").write_text(json.dumps(registry), encoding="utf-8")
            (root / "artifacts/manifest.json").write_text(json.dumps({"max_events": None}), encoding="utf-8")
            with patch.object(retailrocket_main, "ROOT", root):
                self.assertFalse(retailrocket_main._existing_bundle_complete())
                (root / "outputs/models").mkdir()
                (root / "outputs/models/final.joblib").touch()
                registry.update({
                    "selected_model": "LightGBM Ranker",
                    "model_type": "lightgbm",
                    "max_candidates": 50,
                    "pipeline_version": retailrocket_main.PIPELINE_VERSION,
                })
                (root / "outputs/model_registry.json").write_text(json.dumps(registry), encoding="utf-8")
                self.assertTrue(retailrocket_main._existing_bundle_complete())


if __name__ == "__main__":
    unittest.main()
