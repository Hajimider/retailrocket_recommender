"""Candidate recall and candidate-level feature generation."""

from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
import pandas as pd


EVENT_CODE = {"view": 0, "addtocart": 1}
RANK_FEATURES = [
    "n_view", "n_cart", "n_unique_items", "duration_sec", "mean_gap_sec", "max_gap_sec",
    "start_hour", "weekday", "is_weekend", "event_1_code", "event_2_code", "event_3_code",
    "candidate_popularity", "category_popularity", "covisit_sum", "covisit_max", "recency_score",
    "source_covisit", "source_category", "source_popular", "category_match_count",
    "same_category_last", "candidate_category_depth",
]


def build_recall_maps(sessions: pd.DataFrame, catalog: dict[str, object], neighbors_per_item: int = 30) -> dict[str, object]:
    item_counts: Counter[int] = Counter()
    category_counts: dict[int, Counter[int]] = defaultdict(Counter)
    pair_counts: dict[int, Counter[int]] = defaultdict(Counter)
    item_category: dict[int, int] = catalog["item_category"]  # type: ignore[assignment]
    for row in sessions.itertuples(index=False):
        items = [int(row.itemid_1), int(row.itemid_2), int(row.itemid_3)]
        for item in items:
            item_counts[item] += 1
            category = int(item_category.get(item, -1))
            if category >= 0:
                category_counts[category][item] += 1
        for left_position, left in enumerate(items):
            for right_position, right in enumerate(items):
                if left == right or left_position == right_position:
                    continue
                pair_counts[left][right] += 1 / (1 + abs(left_position - right_position))
    neighbors = {
        item: counter.most_common(neighbors_per_item)
        for item, counter in pair_counts.items()
    }
    category_popular = {category: counter.most_common(30) for category, counter in category_counts.items()}
    return {
        "neighbors": neighbors,
        "popular_items": item_counts.most_common(200),
        "category_popular": category_popular,
        "item_popularity": dict(item_counts),
        "category_popularity": {category: sum(counter.values()) for category, counter in category_counts.items()},
        "item_category": item_category,
        "category_depths": catalog["category_depths"],
    }


def _source_text(row: pd.Series) -> str:
    sources = []
    if row["source_covisit"]:
        sources.append("covisit")
    if row["source_category"]:
        sources.append("category")
    if row["source_popular"]:
        sources.append("popular")
    return "+".join(sources)


def generate_candidates(
    sessions: pd.DataFrame,
    maps: dict[str, object],
    max_candidates: int = 50,
) -> pd.DataFrame:
    if max_candidates < 10:
        raise ValueError("max_candidates must be at least 10")
    neighbors: dict[int, list[tuple[int, float]]] = maps["neighbors"]  # type: ignore[assignment]
    category_popular: dict[int, list[tuple[int, int]]] = maps["category_popular"]  # type: ignore[assignment]
    popular_items: list[tuple[int, int]] = maps["popular_items"]  # type: ignore[assignment]
    item_popularity: dict[int, int] = maps["item_popularity"]  # type: ignore[assignment]
    category_popularity: dict[int, int] = maps["category_popularity"]  # type: ignore[assignment]
    item_category: dict[int, int] = maps["item_category"]  # type: ignore[assignment]
    category_depths: dict[int, int] = maps["category_depths"]  # type: ignore[assignment]
    rows: list[dict[str, object]] = []

    for session in sessions.itertuples(index=False):
        items = [int(session.itemid_1), int(session.itemid_2), int(session.itemid_3)]
        seen_items = set(items)
        categories = [int(session.categoryid_1), int(session.categoryid_2), int(session.categoryid_3)]
        candidates: dict[int, dict[str, float]] = {}

        def add(
            item: int,
            source: str,
            score: float = 0.0,
            related_item: int = -1,
            recency: float = 0.0,
        ) -> None:
            if item in seen_items:
                return
            values = candidates.setdefault(item, {
                "source_covisit": 0.0, "source_category": 0.0, "source_popular": 0.0,
                "covisit_sum": 0.0, "covisit_max": 0.0, "recency_score": 0.0,
                "related_itemid": -1.0, "related_score": 0.0,
            })
            values[f"source_{source}"] = 1.0
            if source == "covisit":
                values["covisit_sum"] += score
                values["covisit_max"] = max(values["covisit_max"], score)
                values["recency_score"] = max(values["recency_score"], recency)
                if score > values["related_score"]:
                    values["related_itemid"] = float(related_item)
                    values["related_score"] = score

        for position, item in enumerate(items):
            for neighbor, score in neighbors.get(item, [])[:15]:
                add(
                    int(neighbor), "covisit", float(score) * (position + 1) / 3,
                    related_item=item, recency=float(position + 1),
                )
        for category in set(categories) - {-1}:
            for item, _ in category_popular.get(category, [])[:8]:
                add(int(item), "category")
        for item, _ in popular_items[:25]:
            add(int(item), "popular")

        def priority(entry: tuple[int, dict[str, float]]) -> tuple[float, int]:
            item, value = entry
            score = (
                2.0 * value["source_covisit"] + value["source_category"]
                + 0.5 * value["source_popular"]
                + math.log1p(value["covisit_sum"]) + 0.1 * math.log1p(item_popularity.get(item, 0))
            )
            return score, -item

        selected = dict(sorted(candidates.items(), key=priority, reverse=True)[:max_candidates])
        target = int(session.target_itemid)
        for candidate, values in selected.items():
            category = int(item_category.get(candidate, -1))
            recall_score = (
                2.0 * values["source_covisit"] + values["source_category"]
                + 0.5 * values["source_popular"]
                + math.log1p(values["covisit_sum"])
            )
            rows.append({
                "sample_id": int(session.sample_id),
                "visitorid": int(session.visitorid),
                "session_id": int(session.session_id),
                "candidate_itemid": candidate,
                "candidate_categoryid": category,
                "related_itemid": int(values["related_itemid"]),
                "target_itemid": target,
                "label": int(candidate == target),
                "split": str(session.split),
                "n_view": int(session.n_view),
                "n_cart": int(session.n_cart),
                "n_unique_items": int(session.n_unique_items),
                "duration_sec": float(session.duration_sec),
                "mean_gap_sec": float(session.mean_gap_sec),
                "max_gap_sec": float(session.max_gap_sec),
                "start_hour": int(session.start_hour),
                "weekday": int(session.weekday),
                "is_weekend": int(session.is_weekend),
                "itemid_1": int(session.itemid_1),
                "itemid_2": int(session.itemid_2),
                "itemid_3": int(session.itemid_3),
                "categoryid_1": int(session.categoryid_1),
                "categoryid_2": int(session.categoryid_2),
                "categoryid_3": int(session.categoryid_3),
                "event_1_code": EVENT_CODE[str(session.event_1)],
                "event_2_code": EVENT_CODE[str(session.event_2)],
                "event_3_code": EVENT_CODE[str(session.event_3)],
                "candidate_popularity": math.log1p(item_popularity.get(candidate, 0)),
                "category_popularity": math.log1p(category_popularity.get(category, 0)),
                "covisit_sum": values["covisit_sum"],
                "covisit_max": values["covisit_max"],
                "recency_score": values["recency_score"],
                "source_covisit": int(values["source_covisit"]),
                "source_category": int(values["source_category"]),
                "source_popular": int(values["source_popular"]),
                "category_match_count": sum(category >= 0 and category == value for value in categories),
                "same_category_last": int(category >= 0 and category == categories[-1]),
                "candidate_category_depth": int(category_depths.get(category, -1)),
                "recall_score": recall_score,
            })
    return pd.DataFrame(rows)


def generate_candidates_chunked(
    sessions: pd.DataFrame,
    maps: dict[str, object],
    max_candidates: int = 50,
    chunk_sessions: int = 5_000,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for start in range(0, len(sessions), chunk_sessions):
        part = generate_candidates(
            sessions.iloc[start:start + chunk_sessions], maps, max_candidates
        )
        for column in RANK_FEATURES + ["recall_score"]:
            part[column] = part[column].astype("float32")
        part["label"] = part["label"].astype("int8")
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def add_source_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["source"] = result.apply(_source_text, axis=1)
    return result
