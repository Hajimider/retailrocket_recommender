"""Tune LightGBM Ranker and fit the final recommendation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .candidate_generation import RANK_FEATURES, build_recall_maps, generate_candidates_chunked
    from .deep_models import fit_torch_ranker
    from .train_models import group_sizes, lightgbm_model, ranking_metrics, result_row
except ImportError:
    from candidate_generation import RANK_FEATURES, build_recall_maps, generate_candidates_chunked
    from deep_models import fit_torch_ranker
    from train_models import group_sizes, lightgbm_model, ranking_metrics, result_row


ROOT = Path(__file__).resolve().parent
PIPELINE_VERSION = "unseen-item-deep-v2"
LIGHTGBM_CANDIDATES = [
    {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 60, "subsample": 0.85, "colsample_bytree": 0.85},
    {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40, "subsample": 0.85, "colsample_bytree": 0.85},
    {"n_estimators": 180, "learning_rate": 0.08, "num_leaves": 63, "min_child_samples": 60, "subsample": 0.8, "colsample_bytree": 0.9},
]


def _best(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(rows, key=lambda row: (float(row["ndcg_at_10"]), -float(row["seconds"])))


def run_tuning(
    artifacts_dir: Path,
    output_dir: Path,
    baseline_dir: Path,
    max_candidates: int = 50,
    train_candidates: int = 20,
    seed: int = 42,
    candidate_limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(artifacts_dir / "candidate_train.parquet")
    validation = pd.read_parquet(artifacts_dir / "candidate_val.parquet")
    selection_maps = joblib.load(artifacts_dir / "selection_recall_maps.joblib")
    x_train, y_train = train[RANK_FEATURES], train["label"].to_numpy(dtype="int8")
    x_val = validation[RANK_FEATURES]
    catalog_items = set(selection_maps["item_popularity"])
    configs = LIGHTGBM_CANDIDATES[:candidate_limit] if candidate_limit else LIGHTGBM_CANDIDATES
    tuned_rows: list[dict[str, object]] = []

    for index, config in enumerate(configs, 1):
        print(f"[tune] LightGBM candidate {index}/{len(configs)}...", flush=True)
        started = time.perf_counter()
        # Keep the seed fixed so validation differences come from hyperparameters.
        model = lightgbm_model(config, seed)
        model.fit(x_train, y_train, group=group_sizes(train))
        scores = model.predict(x_val)
        tuned_rows.append({
            "model": "LightGBM Ranker", "config": json.dumps(config, sort_keys=True),
            **ranking_metrics(validation, scores, catalog_items),
            "seconds": round(time.perf_counter() - started, 2),
        })
    best_lgbm = _best(tuned_rows)
    baseline = pd.read_csv(baseline_dir / "baseline_results.csv")
    deployment_candidates = [
        baseline.loc[baseline["model"].eq(name)].iloc[0].to_dict()
        for name in ("Popularity", "Hybrid Recall", "LR", "FM", "MLP", "DeepFM")
    ] + [best_lgbm]
    selected = _best(deployment_candidates)
    selected_name = str(selected["model"])
    selected_type = {
        "Popularity": "popularity", "Hybrid Recall": "hybrid",
        "LR": "lr", "LightGBM Ranker": "lightgbm",
        "FM": "fm", "MLP": "mlp", "DeepFM": "deepfm",
    }[selected_name]

    sessions = pd.read_parquet(artifacts_dir / "recommendation_sessions.parquet")
    catalog = joblib.load(artifacts_dir / "catalog.joblib")
    train_val = sessions.loc[sessions["split"].isin(["train", "val"])].reset_index(drop=True)
    test_sessions = sessions.loc[sessions["split"].eq("test")].reset_index(drop=True)
    print("[final] Rebuilding recall maps with train and validation sessions...", flush=True)
    final_maps = build_recall_maps(train_val, catalog)
    raw_final_train = generate_candidates_chunked(train_val, final_maps, train_candidates)
    final_recalled_ids = raw_final_train.groupby("sample_id", sort=False)["label"].max()
    final_recalled_ids = final_recalled_ids.index[final_recalled_ids.eq(1)]
    final_train = raw_final_train.loc[raw_final_train["sample_id"].isin(final_recalled_ids)].reset_index(drop=True)
    final_test = generate_candidates_chunked(test_sessions, final_maps, max_candidates)
    final_test.to_parquet(artifacts_dir / "candidate_test.parquet", index=False)
    joblib.dump(final_maps, output_dir.parent / "recall_maps.joblib")

    started = time.perf_counter()
    if selected_type == "lightgbm":
        selected_config = json.loads(str(best_lgbm["config"]))
        final_model = lightgbm_model(selected_config, seed)
        final_model.fit(final_train[RANK_FEATURES], final_train["label"], group=group_sizes(final_train))
        test_scores = final_model.predict(final_test[RANK_FEATURES])
        model_path = "models/lightgbm_ranker.joblib"
    elif selected_type in {"fm", "mlp", "deepfm"}:
        torch_config_path = baseline_dir / "torch_configs.json"
        if not torch_config_path.is_file():
            raise FileNotFoundError(f"PyTorch baseline configuration not found: {torch_config_path}")
        torch_configs = json.loads(torch_config_path.read_text(encoding="utf-8"))
        selected_config = dict(torch_configs[selected_type])
        # 深度模型只在验证集胜出后用 train+val 候选重新训练，测试集保持独立。
        final_model = fit_torch_ranker(final_train, selected_type, selected_config, seed)
        test_scores = final_model.predict(final_test)
        model_path = f"models/{selected_type}.joblib"
    else:
        if selected_type == "lr":
            selected_config = {"loss": "log_loss", "max_iter": 80, "average": True}
            final_model = Pipeline([
                ("scale", StandardScaler()),
                ("model", SGDClassifier(
                    loss="log_loss", class_weight="balanced", max_iter=80,
                    tol=1e-3, random_state=seed, average=True,
                )),
            ])
            final_model.fit(final_train[RANK_FEATURES], final_train["label"])
            test_scores = final_model.predict_proba(final_test[RANK_FEATURES])[:, 1]
            model_path = "models/lr.joblib"
        elif selected_type == "popularity":
            selected_config = {"score": "candidate_popularity"}
            final_model = None
            test_scores = final_test["candidate_popularity"].to_numpy()
            model_path = None
        else:
            selected_config = {"score": "recall_score"}
            final_model = None
            test_scores = final_test["recall_score"].to_numpy()
            model_path = None
    models_dir = output_dir.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if model_path is not None:
        joblib.dump(final_model, output_dir.parent / model_path)
    test_metrics = ranking_metrics(final_test, np.asarray(test_scores), set(final_maps["item_popularity"]))
    final_row = result_row("final", selected_name, test_metrics, time.perf_counter() - started)

    registry: dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "selection_metric": "validation_ndcg_at_10",
        "selection_scope": ["Popularity", "Hybrid Recall", "LR", "LightGBM Ranker", "FM", "MLP", "DeepFM"],
        "selected_model": selected_name,
        "model_type": selected_type,
        "model_path": model_path,
        "features": RANK_FEATURES,
        "max_candidates": max_candidates,
        "top_k": 10,
        "config": selected_config,
    }
    (output_dir.parent / "model_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    tuning = pd.DataFrame(tuned_rows)
    tuning.to_csv(output_dir / "tuning_results.csv", index=False)
    pd.concat([baseline, pd.DataFrame([final_row])], ignore_index=True).to_csv(output_dir.parent / "final_results.csv", index=False)
    (output_dir.parent / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    return tuning, registry, test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "tuning")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "outputs" / "baseline")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--train-candidates", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tuning, registry, metrics = run_tuning(
        args.artifacts_dir, args.output_dir, args.baseline_dir,
        args.max_candidates, args.train_candidates, candidate_limit=args.candidate_limit,
    )
    print(tuning.to_string(index=False))
    print(json.dumps({"registry": registry, "test_metrics": metrics}, ensure_ascii=False, indent=2))
