"""Prepare leakage-safe next-item recommendation samples from RetailRocket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EVENT_TYPES = {"view", "addtocart", "transaction"}
SESSION_COLUMNS = [
    "n_view", "n_cart", "n_unique_items", "duration_sec", "mean_gap_sec",
    "max_gap_sec", "start_hour", "weekday", "is_weekend",
    "event_1", "event_2", "event_3",
    "itemid_1", "itemid_2", "itemid_3",
    "categoryid_1", "categoryid_2", "categoryid_3",
    "gap_sec_1", "gap_sec_2", "gap_sec_3",
]


def default_data_dir() -> Path:
    candidates = (ROOT / "data" / "archive", ROOT.parent / "data" / "archive")
    for candidate in candidates:
        if (candidate / "events.csv").is_file():
            return candidate
    return candidates[0]


def visitor_split(visitor_ids: pd.Series, seed: int = 42) -> pd.Series:
    bucket = ((visitor_ids.astype("uint64") * np.uint64(2_654_435_761) + np.uint64(seed)) % 10).astype("int8")
    return pd.Series(np.where(bucket.eq(0), "val", np.where(bucket.eq(1), "test", "train")), index=visitor_ids.index)


def load_events(path: Path, max_events: int | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"RetailRocket events file not found: {path}")
    events = pd.read_csv(
        path,
        usecols=["timestamp", "visitorid", "event", "itemid"],
        dtype={"visitorid": "int64", "event": "string", "itemid": "int64"},
        nrows=max_events,
    )
    if events.empty:
        raise ValueError("events.csv has no rows")
    events["event_time"] = pd.to_datetime(events.pop("timestamp"), unit="ms", utc=True)
    unexpected = set(events["event"].dropna().unique()) - EVENT_TYPES
    if unexpected:
        raise ValueError(f"Unexpected event values: {sorted(unexpected)}")
    return events.sort_values(["visitorid", "event_time"], kind="stable").reset_index(drop=True)


def assign_sessions(events: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    if gap_minutes <= 0:
        raise ValueError("gap_minutes must be positive")
    result = events.copy()
    gap = result.groupby("visitorid", sort=False)["event_time"].diff()
    result["session_id"] = (gap.isna() | gap.gt(pd.Timedelta(minutes=gap_minutes))).cumsum().astype("int64")
    return result


def load_category_tree(path: Path) -> dict[int, int]:
    if not path.is_file():
        return {}
    tree = pd.read_csv(path)
    return {
        int(row.categoryid): int(row.parentid)
        for row in tree.dropna(subset=["parentid"]).itertuples(index=False)
    }


def category_depths(parents: dict[int, int]) -> dict[int, int]:
    depths: dict[int, int] = {}
    for category in set(parents) | set(parents.values()):
        current = category
        seen: set[int] = set()
        depth = 0
        while current in parents and current not in seen:
            seen.add(current)
            current = parents[current]
            depth += 1
        depths[category] = depth
    return depths


def load_item_categories(
    data_dir: Path,
    relevant_items: set[int],
    max_property_rows: int | None = None,
) -> tuple[dict[int, int], dict[str, int]]:
    parts: list[pd.DataFrame] = []
    rows_scanned = 0
    paths = [data_dir / "item_properties_part1.csv", data_dir / "item_properties_part2.csv"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"RetailRocket item property file not found: {path}")
        per_file_limit = None if max_property_rows is None else max(1, max_property_rows // len(paths))
        reader = pd.read_csv(
            path,
            usecols=["timestamp", "itemid", "property", "value"],
            dtype={"itemid": "int64", "property": "string", "value": "string"},
            chunksize=500_000,
            nrows=per_file_limit,
        )
        for chunk in reader:
            rows_scanned += len(chunk)
            chunk = chunk.loc[
                chunk["itemid"].isin(relevant_items) & chunk["property"].eq("categoryid"),
                ["timestamp", "itemid", "value"],
            ].copy()
            if chunk.empty:
                continue
            chunk["categoryid"] = pd.to_numeric(chunk.pop("value"), errors="coerce")
            parts.append(chunk.dropna(subset=["categoryid"]))
    if not parts:
        return {}, {"property_rows_scanned": rows_scanned, "category_rows_used": 0}
    history = pd.concat(parts, ignore_index=True)
    history["categoryid"] = history["categoryid"].astype("int64")
    first_known = history.sort_values(["timestamp", "itemid"], kind="stable").drop_duplicates("itemid", keep="first")
    mapping = dict(zip(first_known["itemid"].astype(int), first_known["categoryid"].astype(int)))
    return mapping, {"property_rows_scanned": rows_scanned, "category_rows_used": len(history)}


def build_observations(events: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["visitorid", "session_id"]
    ranked = events.copy()
    ranked["event_rank"] = ranked.groupby(keys, sort=False).cumcount() + 1
    first_three = ranked.loc[ranked["event_rank"].le(3)].copy()
    eligibility = first_three.assign(is_transaction=first_three["event"].eq("transaction")).groupby(
        keys, as_index=False, sort=False
    ).agg(observed_events=("event", "size"), observed_transaction=("is_transaction", "max"))
    eligibility = eligibility.loc[
        eligibility["observed_events"].eq(3) & ~eligibility["observed_transaction"], keys
    ]
    observed_items = first_three[keys + ["itemid"]].drop_duplicates().assign(_observed_item=True)
    future = ranked.loc[ranked["event_rank"].gt(3)].merge(
        eligibility, on=keys, how="inner", validate="many_to_one"
    )
    future = future.merge(
        observed_items, on=keys + ["itemid"], how="left", validate="many_to_one"
    )
    target = (
        future.loc[future["_observed_item"].isna()]
        .sort_values(keys + ["event_rank"], kind="stable")
        .drop_duplicates(keys, keep="first")
        [keys + ["event", "itemid", "event_time", "event_rank"]]
        .rename(columns={
            "event": "target_event", "itemid": "target_itemid",
            "event_time": "target_time", "event_rank": "target_rank",
        })
    )
    samples = eligibility.merge(target, on=keys, how="inner", validate="one_to_one")
    samples["split"] = visitor_split(samples["visitorid"], seed)
    samples.insert(0, "sample_id", np.arange(len(samples), dtype="int64"))
    observed = first_three.merge(
        samples[["sample_id", *keys, "split"]], on=keys, how="inner", validate="many_to_one"
    )
    return observed, samples


def aggregate_sessions(
    observed: pd.DataFrame,
    samples: pd.DataFrame,
    item_category: dict[int, int],
) -> pd.DataFrame:
    keys = ["sample_id", "visitorid", "session_id"]
    history = observed.sort_values(keys + ["event_rank"], kind="stable").copy()
    history["categoryid"] = history["itemid"].map(item_category).fillna(-1).astype("int64")
    history["is_view"] = history["event"].eq("view").astype("int8")
    history["is_cart"] = history["event"].eq("addtocart").astype("int8")
    history["gap_sec"] = history.groupby(keys, sort=False)["event_time"].diff().dt.total_seconds().fillna(0)
    grouped = history.groupby(keys, as_index=False, sort=False)
    features = grouped.agg(
        n_view=("is_view", "sum"),
        n_cart=("is_cart", "sum"),
        n_unique_items=("itemid", "nunique"),
        start_time=("event_time", "min"),
        end_time=("event_time", "max"),
        mean_gap_sec=("gap_sec", "mean"),
        max_gap_sec=("gap_sec", "max"),
    )
    features["duration_sec"] = (features.pop("end_time") - features["start_time"]).dt.total_seconds()
    features["start_hour"] = features["start_time"].dt.hour.astype("int8")
    features["weekday"] = features["start_time"].dt.dayofweek.astype("int8")
    features["is_weekend"] = features["weekday"].isin([5, 6]).astype("int8")
    features = features.drop(columns="start_time").set_index(keys)
    for position in range(1, 4):
        row = history.loc[history["event_rank"].eq(position)].set_index(keys)
        features[f"event_{position}"] = row["event"]
        features[f"itemid_{position}"] = row["itemid"]
        features[f"categoryid_{position}"] = row["categoryid"]
        features[f"gap_sec_{position}"] = row["gap_sec"]
        features[f"event_time_{position}"] = row["event_time"]
    base = samples.drop(columns="target_time", errors="ignore").merge(
        features.reset_index(), on=keys, how="inner", validate="one_to_one"
    )
    return base


def build_online_session(events: pd.DataFrame, item_category: dict[int, int]) -> pd.DataFrame:
    if len(events) != 3:
        raise ValueError("Exactly three observation events are required")
    if set(events["event"]) - {"view", "addtocart"}:
        raise ValueError("Observation events may only contain view or addtocart")
    history = events.sort_values("event_time", kind="stable").reset_index(drop=True).copy()
    if history["event_time"].diff().gt(pd.Timedelta(minutes=30)).any():
        raise ValueError("The three events must belong to one 30-minute session")
    history["sample_id"] = 0
    history["visitorid"] = 0
    history["session_id"] = 0
    history["event_rank"] = np.arange(1, 4)
    history["split"] = "online"
    sample = pd.DataFrame({
        "sample_id": [0], "visitorid": [0], "session_id": [0],
        "target_event": ["view"], "target_itemid": [-1], "split": ["online"],
    })
    return aggregate_sessions(history, sample, item_category)


def _validate_output_dir(output_dir: Path, data_dir: Path) -> None:
    output = output_dir.resolve()
    root = ROOT.resolve()
    data = data_dir.resolve()
    artifacts = (ROOT / "artifacts").resolve()
    temporary = (ROOT / "tmp").resolve()
    allowed_generated_dir = output == artifacts or output == temporary or temporary in output.parents
    unsafe = (
        not allowed_generated_dir or output == root or output == data
        or data in output.parents or output in data.parents
    )
    if unsafe:
        raise ValueError("output_dir must be a dedicated directory inside this project and outside the raw data directory")


def prepare(
    data_dir: Path,
    output_dir: Path,
    gap_minutes: int = 30,
    max_events: int | None = None,
    max_property_rows: int | None = None,
    seed: int = 42,
) -> dict[str, object]:
    _validate_output_dir(output_dir, data_dir)
    print("[data] Loading events and building sessions...", flush=True)
    events = assign_sessions(load_events(data_dir / "events.csv", max_events), gap_minutes)
    observed, samples = build_observations(events, seed)
    relevant_items = set(observed["itemid"].unique()) | set(samples["target_itemid"].unique())
    print("[data] Scanning item categories...", flush=True)
    item_category, property_stats = load_item_categories(data_dir, relevant_items, max_property_rows)
    parents = load_category_tree(data_dir / "category_tree.csv")
    depths = category_depths(parents)
    sessions = aggregate_sessions(observed, samples, item_category)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(output_dir / "recommendation_sessions.parquet", index=False)
    joblib.dump({"item_category": item_category, "category_depths": depths}, output_dir / "catalog.joblib")
    manifest = {
        "events": len(events),
        "visitors": int(events["visitorid"].nunique()),
        "samples": len(sessions),
        "target_event_counts": {name: int(count) for name, count in sessions["target_event"].value_counts().items()},
        "split_counts": {name: int(count) for name, count in sessions["split"].value_counts().items()},
        "unique_target_items": int(sessions["target_itemid"].nunique()),
        "gap_minutes": gap_minutes,
        "target_definition": "first_unseen_item_after_three_events",
        "max_events": max_events,
        **property_stats,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[data] Built {len(sessions):,} next-item samples.", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--gap-minutes", type=int, default=30)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-property-rows", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(prepare(args.data_dir, args.output_dir, args.gap_minutes, args.max_events, args.max_property_rows), ensure_ascii=False, indent=2))
