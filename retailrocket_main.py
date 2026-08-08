"""Run the complete RetailRocket product recommendation workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

try:
    from .prepare_data import default_data_dir, prepare
    from .score_batch import score_batch
    from .train_models import run_baselines
    from .tune_model import PIPELINE_VERSION, run_tuning
except ImportError:
    from prepare_data import default_data_dir, prepare
    from score_batch import score_batch
    from train_models import run_baselines
    from tune_model import PIPELINE_VERSION, run_tuning


ROOT = Path(__file__).resolve().parent


def _divider(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}", flush=True)


def _print_metrics(frame) -> None:
    columns = ["model", "candidate_recall", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10", "seconds"]
    display = frame.loc[:, columns].copy()
    display.columns = ["Model", "Candidate R", "Recall@5", "Recall@10", "MRR@10", "NDCG@10", "Time(s)"]
    for column in display.columns[1:-1]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display["Time(s)"] = display["Time(s)"].map(lambda value: f"{value:.1f}")
    print(display.to_string(index=False), flush=True)


def _preflight(data_dir: Path, top_k: int, max_candidates: int) -> Path:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    if max_candidates < 10:
        raise ValueError("max_candidates must be at least 10")
    if max_candidates < top_k:
        raise ValueError("max_candidates must be at least top_k")
    data = data_dir.resolve()
    for name in ("artifacts", "outputs", "tmp"):
        generated = (ROOT / name).resolve()
        if data == generated or data in generated.parents or generated in data.parents:
            raise ValueError("data_dir must be outside this project's generated directories")
    required = ("events.csv", "item_properties_part1.csv", "item_properties_part2.csv")
    missing = [name for name in required if not (data / name).is_file()]
    if missing:
        raise FileNotFoundError(f"RetailRocket data files not found: {', '.join(missing)}")
    return data


def _existing_bundle_complete(max_candidates: int | None = None, quick: bool = False) -> bool:
    required = (
        ROOT / "artifacts" / "recommendation_sessions.parquet",
        ROOT / "artifacts" / "candidate_test.parquet",
        ROOT / "artifacts" / "catalog.joblib",
        ROOT / "artifacts" / "manifest.json",
        ROOT / "outputs" / "model_registry.json",
        ROOT / "outputs" / "recall_maps.joblib",
        ROOT / "outputs" / "batch_recommendations.csv",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        registry = json.loads((ROOT / "outputs" / "model_registry.json").read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    model_types = {
        "Popularity": "popularity",
        "Hybrid Recall": "hybrid",
        "LR": "lr",
        "LightGBM Ranker": "lightgbm",
        "FM": "fm",
        "MLP": "mlp",
        "DeepFM": "deepfm",
    }
    selected_model = registry.get("selected_model")
    model_type = registry.get("model_type")
    if registry.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if selected_model not in model_types or model_types[selected_model] != model_type:
        return False
    if not isinstance(registry.get("max_candidates"), int) or registry["max_candidates"] < 10:
        return False
    if max_candidates is not None and registry["max_candidates"] != max_candidates:
        return False
    expected_max_events = 250_000 if quick else None
    if manifest.get("max_events") != expected_max_events:
        return False
    model_path = registry.get("model_path")
    if model_type in {"lr", "lightgbm", "fm", "mlp", "deepfm"}:
        return isinstance(model_path, str) and (ROOT / "outputs" / model_path).is_file()
    return model_path is None


def run_pipeline(
    data_dir: Path | None = None,
    top_k: int = 10,
    max_candidates: int = 50,
    seed: int = 42,
    quick: bool = False,
    reuse_existing: bool = False,
) -> None:
    started = time.perf_counter()
    if reuse_existing and _existing_bundle_complete(max_candidates, quick):
        recommendations = score_batch(ROOT / "artifacts", ROOT / "outputs", top_k=top_k)
        print("[reuse] Existing model bundle reused; batch recommendations refreshed.")
        print(f"Rows: {len(recommendations):,} | Output: outputs/batch_recommendations.csv", flush=True)
        return
    selected_data_dir = _preflight(data_dir or default_data_dir(), top_k, max_candidates)

    max_events = 250_000 if quick else None
    max_property_rows = 500_000 if quick else None
    effective_candidates = min(max_candidates, 30) if quick else max_candidates
    train_candidates = 15 if quick else 20
    candidate_limit = 1 if quick else None

    _divider("[1/4] Session and target-item preparation")
    manifest = prepare(
        selected_data_dir, ROOT / "artifacts",
        max_events=max_events, max_property_rows=max_property_rows, seed=seed,
    )
    print(
        f"Samples: {manifest['samples']:,} | Unique targets: {manifest['unique_target_items']:,} | "
        f"Property rows scanned: {manifest['property_rows_scanned']:,}", flush=True,
    )
    for name in ("outputs", "tmp"):
        path = ROOT / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    _divider("[2/4] Recall and ML/DL ranking baselines")
    baseline = run_baselines(
        ROOT / "artifacts", ROOT / "outputs" / "baseline",
        effective_candidates, train_candidates, seed, quick=quick,
    )
    _print_metrics(baseline)

    _divider("[3/4] LightGBM tuning and final evaluation")
    tuning, registry, test_metrics = run_tuning(
        ROOT / "artifacts", ROOT / "outputs" / "tuning", ROOT / "outputs" / "baseline",
        effective_candidates, train_candidates, seed, candidate_limit,
    )
    _print_metrics(tuning)
    print(f"Selected final model: {registry['selected_model']}", flush=True)
    print(
        f"Test Recall@10: {test_metrics['recall_at_10']:.4f} | "
        f"Test NDCG@10: {test_metrics['ndcg_at_10']:.4f}", flush=True,
    )

    _divider("[4/4] Batch Top-K recommendation")
    recommendations = score_batch(ROOT / "artifacts", ROOT / "outputs", top_k=top_k)
    preview = recommendations[["sample_id", "rank", "candidate_itemid", "relative_score", "source", "hit"]].head(top_k)
    print(preview.to_string(index=False, formatters={"relative_score": lambda value: f"{value:.4f}"}), flush=True)
    print(f"Rows: {len(recommendations):,} | Output: outputs/batch_recommendations.csv", flush=True)
    _divider("Completed")
    print(f"Total time: {time.perf_counter() - started:.1f} seconds", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.data_dir, args.top_k, args.max_candidates, args.seed, args.quick, args.reuse_existing)
