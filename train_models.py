"""Build candidates and compare recommendation ranking baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .candidate_generation import RANK_FEATURES, build_recall_maps, generate_candidates_chunked
    from .deep_models import fit_torch_ranker
except ImportError:
    from candidate_generation import RANK_FEATURES, build_recall_maps, generate_candidates_chunked
    from deep_models import fit_torch_ranker


ROOT = Path(__file__).resolve().parent


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("sample_id", sort=False).size().to_numpy(dtype="int32")


def ranking_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    catalog_items: set[int],
) -> dict[str, float]:
    if len(frame) != len(scores):
        raise ValueError("score count does not match candidate rows")
    recalls_5: list[float] = []
    recalls_10: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    candidate_hits: list[float] = []
    recommended: set[int] = set()
    start = 0
    for size in group_sizes(frame):
        end = start + int(size)
        labels = frame["label"].to_numpy()[start:end]
        items = frame["candidate_itemid"].to_numpy()[start:end]
        order = np.argsort(-scores[start:end], kind="stable")
        ranked_labels = labels[order]
        positive = np.flatnonzero(ranked_labels == 1)
        rank = int(positive[0]) + 1 if len(positive) else 0
        candidate_hits.append(float(rank > 0))
        recalls_5.append(float(0 < rank <= 5))
        recalls_10.append(float(0 < rank <= 10))
        reciprocal_ranks.append(1 / rank if 0 < rank <= 10 else 0.0)
        ndcgs.append(1 / np.log2(rank + 1) if 0 < rank <= 10 else 0.0)
        recommended.update(int(item) for item in items[order[:10]] if int(item) in catalog_items)
        start = end
    return {
        "candidate_recall": float(np.mean(candidate_hits)),
        "recall_at_5": float(np.mean(recalls_5)),
        "recall_at_10": float(np.mean(recalls_10)),
        "mrr_at_10": float(np.mean(reciprocal_ranks)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "coverage_at_10": len(recommended) / max(1, len(catalog_items)),
    }


def result_row(stage: str, model: str, metrics: dict[str, float], seconds: float) -> dict[str, object]:
    return {"stage": stage, "model": model, **metrics, "seconds": round(seconds, 2)}


def lightgbm_model(config: dict[str, object], seed: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        random_state=seed,
        subsample_freq=1,
        n_jobs=-1,
        verbosity=-1,
        **config,
    )


def prepare_selection_candidates(
    artifacts_dir: Path,
    max_candidates: int,
    train_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    sessions = pd.read_parquet(artifacts_dir / "recommendation_sessions.parquet")
    catalog = joblib.load(artifacts_dir / "catalog.joblib")
    train = sessions.loc[sessions["split"].eq("train")].reset_index(drop=True)
    validation = sessions.loc[sessions["split"].eq("val")].reset_index(drop=True)
    maps = build_recall_maps(train, catalog)
    raw_train = generate_candidates_chunked(train, maps, train_candidates)
    recalled_train_ids = raw_train.groupby("sample_id", sort=False)["label"].max()
    recalled_train_ids = recalled_train_ids.index[recalled_train_ids.eq(1)]
    train_frame = raw_train.loc[raw_train["sample_id"].isin(recalled_train_ids)].reset_index(drop=True)
    val_frame = generate_candidates_chunked(validation, maps, max_candidates)
    train_frame.to_parquet(artifacts_dir / "candidate_train.parquet", index=False)
    val_frame.to_parquet(artifacts_dir / "candidate_val.parquet", index=False)
    joblib.dump(maps, artifacts_dir / "selection_recall_maps.joblib")
    return train_frame, val_frame, maps


def run_baselines(
    artifacts_dir: Path,
    output_dir: Path,
    max_candidates: int = 50,
    train_candidates: int = 20,
    seed: int = 42,
    quick: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[recall] Building train-only candidate maps...", flush=True)
    train, validation, maps = prepare_selection_candidates(artifacts_dir, max_candidates, train_candidates)
    catalog_items = set(maps["item_popularity"])
    rows: list[dict[str, object]] = []

    started = time.perf_counter()
    rows.append(result_row(
        "baseline", "Popularity",
        ranking_metrics(validation, validation["candidate_popularity"].to_numpy(), catalog_items),
        time.perf_counter() - started,
    ))
    started = time.perf_counter()
    rows.append(result_row(
        "baseline", "Hybrid Recall",
        ranking_metrics(validation, validation["recall_score"].to_numpy(), catalog_items),
        time.perf_counter() - started,
    ))

    x_train = train[RANK_FEATURES]
    y_train = train["label"].to_numpy(dtype="int8")
    x_val = validation[RANK_FEATURES]

    print("[model] Training LR ranking baseline...", flush=True)
    started = time.perf_counter()
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("model", SGDClassifier(
            loss="log_loss", class_weight="balanced", max_iter=80,
            tol=1e-3, random_state=seed, average=True,
        )),
    ])
    lr.fit(x_train, y_train)
    lr_score = lr.predict_proba(x_val)[:, 1]
    rows.append(result_row("baseline", "LR", ranking_metrics(validation, lr_score, catalog_items), time.perf_counter() - started))
    joblib.dump(lr, output_dir / "lr.joblib")

    lgbm_config = {
        "n_estimators": 250, "learning_rate": 0.05, "num_leaves": 31,
        "min_child_samples": 40, "subsample": 0.85, "colsample_bytree": 0.85,
    }
    print("[model] Training LightGBM Ranker baseline...", flush=True)
    started = time.perf_counter()
    lgbm = lightgbm_model(lgbm_config, seed)
    lgbm.fit(x_train, y_train, group=group_sizes(train))
    lgbm_score = lgbm.predict(x_val)
    rows.append(result_row(
        "baseline", "LightGBM Ranker",
        ranking_metrics(validation, lgbm_score, catalog_items),
        time.perf_counter() - started,
    ))
    joblib.dump(lgbm, output_dir / "lightgbm_ranker.joblib")

    torch_configs = {
        "fm": {"embedding_dim": 8, "learning_rate": 0.001, "weight_decay": 1e-5, "epochs": 2 if quick else 6, "batch_size": 4096},
        "mlp": {"embedding_dim": 8, "hidden_units": [64, 32], "dropout": 0.15, "learning_rate": 0.001, "weight_decay": 1e-5, "epochs": 2 if quick else 6, "batch_size": 4096},
        "deepfm": {"embedding_dim": 8, "hidden_units": [64, 32], "dropout": 0.15, "learning_rate": 0.001, "weight_decay": 1e-5, "epochs": 2 if quick else 6, "batch_size": 4096},
    }
    for kind, config in torch_configs.items():
        display_name = kind.upper() if kind != "deepfm" else "DeepFM"
        print(f"[model] Training {display_name} with PyTorch...", flush=True)
        started = time.perf_counter()
        bundle = fit_torch_ranker(train, kind, config, seed)
        scores = bundle.predict(validation)
        rows.append(result_row(
            "baseline", display_name,
            ranking_metrics(validation, scores, catalog_items),
            time.perf_counter() - started,
        ))
        joblib.dump(bundle, output_dir / f"{kind}.joblib")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "baseline_results.csv", index=False)
    (output_dir / "baseline_config.json").write_text(json.dumps(lgbm_config, indent=2), encoding="utf-8")
    (output_dir / "torch_configs.json").write_text(json.dumps(torch_configs, indent=2), encoding="utf-8")
    stats = {
        "train_candidate_rows": len(train),
        "validation_candidate_rows": len(validation),
        "validation_candidate_recall": float(validation.groupby("sample_id")["label"].max().mean()),
        "training_recalled_sessions": int(train["sample_id"].nunique()),
        "catalog_items": len(catalog_items),
    }
    (output_dir / "candidate_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "baseline")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--train-candidates", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run_baselines(args.artifacts_dir, args.output_dir, args.max_candidates, args.train_candidates, quick=args.quick).to_string(index=False))
