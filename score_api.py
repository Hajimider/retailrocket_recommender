"""Flask API and local page for session-based unseen-item recommendation."""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

try:
    from .score_batch import SOURCE_ZH, load_bundle, recommend_session
except ImportError:
    from score_batch import SOURCE_ZH, load_bundle, recommend_session


ROOT = Path(__file__).resolve().parent


def _iso_timestamp(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def load_examples(artifacts_dir: Path, limit: int = 20) -> list[dict[str, object]]:
    """Load real test sessions without exposing their future target item."""
    path = artifacts_dir / "recommendation_sessions.parquet"
    if not path.is_file():
        return []
    sessions = pd.read_parquet(path)
    required = {"sample_id", "split", *{f"event_{i}" for i in (1, 2, 3)}, *{f"itemid_{i}" for i in (1, 2, 3)}, *{f"event_time_{i}" for i in (1, 2, 3)}}
    if not required.issubset(sessions.columns):
        return []
    sessions = sessions.loc[sessions["split"].eq("test")].head(limit)
    examples = []
    for row in sessions.itertuples(index=False):
        events = [
            {
                "event": str(getattr(row, f"event_{index}")),
                "itemid": int(getattr(row, f"itemid_{index}")),
                "timestamp": _iso_timestamp(getattr(row, f"event_time_{index}")),
            }
            for index in (1, 2, 3)
        ]
        examples.append({"sample_id": int(row.sample_id), "events": events})
    return examples


def _event_frame(payload: object) -> tuple[pd.DataFrame, int]:
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 3:
        raise ValueError("events 必须包含 3 条行为")
    top_k = int(payload.get("top_k", 10))
    if not 1 <= top_k <= 20:
        raise ValueError("top_k 必须在 1 到 20 之间")
    rows = []
    for event in events:
        if not isinstance(event, dict) or not {"event", "itemid", "timestamp"}.issubset(event):
            raise ValueError("每条行为必须包含 event、itemid 和 timestamp")
        event_type = str(event["event"])
        if event_type not in {"view", "addtocart"}:
            raise ValueError("输入行为只能是 view 或 addtocart")
        itemid = int(event["itemid"])
        if itemid < 0:
            raise ValueError("商品 ID 不能为负数")
        rows.append({
            "event": event_type,
            "itemid": itemid,
            "event_time": pd.to_datetime(event["timestamp"], utc=True, errors="raise"),
        })
    return pd.DataFrame(rows), top_k


def _source_zh(source: str) -> str:
    return "、".join(SOURCE_ZH.get(name, name) for name in source.split("+"))


def _reason_zh(row: object) -> str:
    source = str(row.source)
    related_itemid = int(row.related_itemid)
    if "covisit" in source and related_itemid >= 0:
        return f"与商品 {related_itemid} 经常出现在同一会话"
    if "category" in source:
        return "同类目中的热门商品"
    return "全站热门商品"


def create_app(
    artifacts_dir: Path = ROOT / "artifacts",
    outputs_dir: Path = ROOT / "outputs",
) -> Flask:
    catalog = joblib.load(artifacts_dir / "catalog.joblib")
    bundle = load_bundle(outputs_dir)
    model_name = str(bundle["registry"]["selected_model"])
    examples = load_examples(artifacts_dir)
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        example = examples[0] if examples else {"events": []}
        return render_template("recommender.html", model=model_name, example=example, examples=examples)

    @app.get("/health")
    def health() -> tuple[object, int]:
        return jsonify({
            "status": "ok", "model": model_name,
            "task": "next-unseen-item-recommendation",
            "examples": len(examples),
        }), 200

    @app.get("/examples")
    def get_examples() -> tuple[object, int]:
        return jsonify({"examples": examples}), 200

    @app.post("/recommend")
    def recommend() -> tuple[object, int]:
        try:
            events, top_k = _event_frame(request.get_json(silent=True))
            result = recommend_session(events, catalog, bundle, top_k)
            recommendations = [
                {
                    "rank": int(row.rank),
                    "itemid": int(row.candidate_itemid),
                    "categoryid": int(row.candidate_categoryid),
                    "related_itemid": int(row.related_itemid),
                    "score": round(float(row.relative_score), 6),
                    "source": str(row.source),
                    "source_zh": _source_zh(str(row.source)),
                    "reason_zh": _reason_zh(row),
                }
                for row in result.itertuples(index=False)
            ]
            return jsonify({
                "model": model_name, "top_k": top_k,
                "excluded_itemids": sorted(events["itemid"].astype(int).tolist()),
                "recommendations": recommendations,
            }), 200
        except (TypeError, ValueError, KeyError, OverflowError) as error:
            return jsonify({"error": str(error)}), 400

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_app(args.artifacts_dir, args.outputs_dir).run(host=args.host, port=args.port)
