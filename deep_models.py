"""CPU PyTorch rankers for sparse recommendation fields and dense features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .candidate_generation import RANK_FEATURES
except ImportError:
    from candidate_generation import RANK_FEATURES


SPARSE_FIELDS = [
    "itemid_1", "itemid_2", "itemid_3", "candidate_itemid",
    "categoryid_1", "categoryid_2", "categoryid_3", "candidate_categoryid",
    "event_1_code", "event_2_code", "event_3_code",
]
DENSE_FEATURES = list(RANK_FEATURES)


@dataclass
class SparseFeatureEncoder:
    """Fit categorical vocabularies and dense normalization on training rows only."""

    field_names: list[str]
    mappings: dict[str, dict[int, int]] | None = None
    dense_mean: np.ndarray | None = None
    dense_scale: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "SparseFeatureEncoder":
        self.mappings = {}
        for field in self.field_names:
            values = pd.to_numeric(frame[field], errors="coerce").fillna(-1).astype("int64")
            self.mappings[field] = {int(value): index for index, value in enumerate(sorted(values.unique()), 1)}
        dense = frame[DENSE_FEATURES].astype("float32").to_numpy()
        self.dense_mean = dense.mean(axis=0)
        self.dense_scale = dense.std(axis=0)
        self.dense_scale[self.dense_scale < 1e-6] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.mappings is None or self.dense_mean is None or self.dense_scale is None:
            raise RuntimeError("SparseFeatureEncoder must be fitted before transform")
        sparse = []
        for field in self.field_names:
            mapping = self.mappings[field]
            values = pd.to_numeric(frame[field], errors="coerce").fillna(-1).astype("int64")
            sparse.append(values.map(mapping).fillna(0).to_numpy(dtype="int64"))
        dense = frame[DENSE_FEATURES].astype("float32").to_numpy()
        dense = (dense - self.dense_mean) / self.dense_scale
        return np.column_stack(sparse), dense.astype("float32")

    def field_dims(self) -> list[int]:
        if self.mappings is None:
            raise RuntimeError("SparseFeatureEncoder must be fitted before field_dims")
        return [max(mapping.values(), default=0) + 1 for mapping in self.mappings.values()]


class _FeatureEmbedding(nn.Module):
    def __init__(self, field_dims: list[int], embedding_dim: int) -> None:
        super().__init__()
        self.field_dims = field_dims
        offsets = np.cumsum([0, *field_dims[:-1]], dtype="int64")
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)

    def forward(self, sparse: torch.Tensor) -> torch.Tensor:
        return self.embedding(sparse + self.offsets)


class MLPRecommender(nn.Module):
    def __init__(self, field_dims: list[int], dense_dim: int, embedding_dim: int, hidden_units: Iterable[int], dropout: float) -> None:
        super().__init__()
        self.features = _FeatureEmbedding(field_dims, embedding_dim)
        layers: list[nn.Module] = []
        input_dim = len(field_dims) * embedding_dim + dense_dim
        for hidden in hidden_units:
            layers.extend([nn.Linear(input_dim, int(hidden)), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = int(hidden)
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        embedded = self.features(sparse).flatten(start_dim=1)
        return self.mlp(torch.cat([embedded, dense], dim=1)).squeeze(1)


class FMRecommender(nn.Module):
    def __init__(self, field_dims: list[int], dense_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.features = _FeatureEmbedding(field_dims, embedding_dim)
        self.first_order = _FeatureEmbedding(field_dims, 1)
        self.dense_linear = nn.Linear(dense_dim, 1)

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        embedded = self.features(sparse)
        summed = embedded.sum(dim=1)
        pairwise = 0.5 * (summed.pow(2) - embedded.pow(2).sum(dim=1)).sum(dim=1)
        first = self.first_order(sparse).sum(dim=1).squeeze(1)
        return first + self.dense_linear(dense).squeeze(1) + pairwise


class DeepFMRecommender(nn.Module):
    def __init__(self, field_dims: list[int], dense_dim: int, embedding_dim: int, hidden_units: Iterable[int], dropout: float) -> None:
        super().__init__()
        self.features = _FeatureEmbedding(field_dims, embedding_dim)
        self.first_order = _FeatureEmbedding(field_dims, 1)
        self.dense_linear = nn.Linear(dense_dim, 1)
        layers: list[nn.Module] = []
        input_dim = len(field_dims) * embedding_dim + dense_dim
        for hidden in hidden_units:
            layers.extend([nn.Linear(input_dim, int(hidden)), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = int(hidden)
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        embedded = self.features(sparse)
        summed = embedded.sum(dim=1)
        pairwise = 0.5 * (summed.pow(2) - embedded.pow(2).sum(dim=1)).sum(dim=1)
        first = self.first_order(sparse).sum(dim=1).squeeze(1)
        deep = self.deep(torch.cat([embedded.flatten(start_dim=1), dense], dim=1)).squeeze(1)
        return first + self.dense_linear(dense).squeeze(1) + pairwise + deep


def _build_model(kind: str, field_dims: list[int], dense_dim: int, config: dict[str, object]) -> nn.Module:
    embedding_dim = int(config.get("embedding_dim", 8))
    hidden_units = config.get("hidden_units", [64, 32])
    dropout = float(config.get("dropout", 0.15))
    if kind == "fm":
        return FMRecommender(field_dims, dense_dim, embedding_dim)
    if kind == "mlp":
        return MLPRecommender(field_dims, dense_dim, embedding_dim, hidden_units, dropout)
    if kind == "deepfm":
        return DeepFMRecommender(field_dims, dense_dim, embedding_dim, hidden_units, dropout)
    raise ValueError(f"Unsupported PyTorch ranker: {kind}")


@dataclass
class TorchRankerBundle:
    kind: str
    encoder: SparseFeatureEncoder
    model: nn.Module
    config: dict[str, object]

    def predict(self, frame: pd.DataFrame, batch_size: int = 4096) -> np.ndarray:
        sparse, dense = self.encoder.transform(frame)
        dataset = TensorDataset(torch.from_numpy(sparse), torch.from_numpy(dense))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for sparse_batch, dense_batch in loader:
                predictions.append(self.model(sparse_batch, dense_batch).cpu().numpy())
        return np.concatenate(predictions) if predictions else np.empty(0, dtype="float64")


def fit_torch_ranker(
    train_frame: pd.DataFrame,
    kind: str,
    config: dict[str, object],
    seed: int = 42,
) -> TorchRankerBundle:
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = SparseFeatureEncoder(SPARSE_FIELDS).fit(train_frame)
    sparse, dense = encoder.transform(train_frame)
    labels = train_frame["label"].to_numpy(dtype="float32")
    dataset = TensorDataset(
        torch.from_numpy(sparse), torch.from_numpy(dense), torch.from_numpy(labels),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 4096)), shuffle=True)
    model = _build_model(kind, encoder.field_dims(), len(DENSE_FEATURES), config)
    positive = max(1.0, float(labels.sum()))
    negative = max(1.0, float(len(labels) - labels.sum()))
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    model.train()
    for _ in range(int(config.get("epochs", 6))):
        for sparse_batch, dense_batch, label_batch in loader:
            optimizer.zero_grad()
            batch_loss = loss(model(sparse_batch, dense_batch), label_batch)
            batch_loss.backward()
            optimizer.step()
    return TorchRankerBundle(kind, encoder, model, dict(config))
