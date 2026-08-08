"""Batch and online scoring for the selected recommendation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .candidate_generation import RANK_FEATURES, add_source_labels, generate_candidates_chunked
    from .prepare_data import build_online_session
except ImportError:
    from candidate_generation import RANK_FEATURES, add_source_labels, generate_candidates_chunked
    from prepare_data import build_online_session


ROOT = Path(__file__).resolve().parent
SOURCE_ZH = {
    "covisit": "相似会话共现",
    "category": "同类目热门",
    "popular": "全站热门",
}


def load_bundle(outputs_dir: Path) -> dict[str, object]:
    registry = json.loads((outputs_dir / "model_registry.json").read_text(encoding="utf-8"))
    model_path = registry.get("model_path")
    return {
        "registry": registry,
        "model": joblib.load(outputs_dir / model_path) if model_path else None,
        "recall_maps": joblib.load(outputs_dir / "recall_maps.joblib"),
    }


def predict_scores(bundle: dict[str, object], candidates: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    features = candidates[RANK_FEATURES]
    if bundle["registry"]["model_type"] == "lr":
        return np.asarray(model.predict_proba(features)[:, 1], dtype="float64")
    if bundle["registry"]["model_type"] == "popularity":
        return candidates["candidate_popularity"].to_numpy(dtype="float64")
    if bundle["registry"]["model_type"] == "hybrid":
        return candidates["recall_score"].to_numpy(dtype="float64")
    if bundle["registry"]["model_type"] in {"fm", "mlp", "deepfm"}:
        return np.asarray(model.predict(candidates), dtype="float64")
    return np.asarray(model.predict(features), dtype="float64")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def topk_recommendations(candidates: pd.DataFrame, scores: np.ndarray, top_k: int) -> pd.DataFrame:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    rows: list[pd.DataFrame] = []
    start = 0
    for _, group in candidates.groupby("sample_id", sort=False):
        end = start + len(group)
        local_scores = scores[start:end]
        order = np.argsort(-local_scores, kind="stable")[:top_k]
        selected = group.iloc[order].copy()
        selected["rank"] = np.arange(1, len(selected) + 1)
        selected["ranking_score"] = local_scores[order]
        selected["relative_score"] = _softmax(local_scores)[order]
        rows.append(selected)
        start = end
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result = add_source_labels(result)
    columns = [
        "sample_id", "visitorid", "session_id", "rank", "candidate_itemid",
        "candidate_categoryid", "related_itemid", "relative_score", "ranking_score",
        "source", "target_itemid", "label", "split",
    ]
    return result[columns]


def score_batch(
    artifacts_dir: Path = ROOT / "artifacts",
    outputs_dir: Path = ROOT / "outputs",
    output_path: Path | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    candidates = pd.read_parquet(artifacts_dir / "candidate_test.parquet")
    bundle = load_bundle(outputs_dir)
    result = topk_recommendations(candidates, predict_scores(bundle, candidates), top_k)
    result["hit"] = result["candidate_itemid"].eq(result["target_itemid"])
    if output_path is None:
        output_path = outputs_dir / "batch_recommendations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def recommend_session(
    events: pd.DataFrame,
    catalog: dict[str, object],
    bundle: dict[str, object],
    top_k: int = 10,
) -> pd.DataFrame:
    session = build_online_session(events, catalog["item_category"])
    max_candidates = int(bundle["registry"]["max_candidates"])
    candidates = generate_candidates_chunked(session, bundle["recall_maps"], max_candidates)
    return topk_recommendations(candidates, predict_scores(bundle, candidates), top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame = score_batch(args.artifacts_dir, args.outputs_dir, args.output, args.top_k)
    print(frame.head(20).to_string(index=False))
    print(f"Wrote {len(frame):,} recommendation rows.")
