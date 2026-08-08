from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_generation import RANK_FEATURES  # noqa: E402
from deep_models import DENSE_FEATURES, SPARSE_FIELDS, SparseFeatureEncoder, fit_torch_ranker  # noqa: E402


def candidate_frame(rows: int = 12) -> pd.DataFrame:
    frame = pd.DataFrame({feature: np.arange(rows, dtype="float32") for feature in RANK_FEATURES})
    frame["label"] = np.array([0, 1] * (rows // 2), dtype="int8")
    for field in SPARSE_FIELDS:
        frame[field] = np.arange(rows, dtype="int64") % 4
    return frame


class DeepModelTest(unittest.TestCase):
    def test_encoder_uses_training_vocab_and_unknown_bucket(self) -> None:
        train = candidate_frame(4)
        validation = candidate_frame(2)
        validation["candidate_itemid"] = 9999
        encoder = SparseFeatureEncoder(SPARSE_FIELDS).fit(train)
        sparse, dense = encoder.transform(validation)
        self.assertEqual(sparse.shape, (2, len(SPARSE_FIELDS)))
        self.assertEqual(dense.shape, (2, len(DENSE_FEATURES)))
        self.assertTrue(np.all(sparse[:, SPARSE_FIELDS.index("candidate_itemid")] == 0))

    def test_torch_rankers_train_predict_and_reload(self) -> None:
        train = candidate_frame()
        config = {
            "embedding_dim": 4,
            "hidden_units": [8, 4],
            "dropout": 0.1,
            "learning_rate": 0.01,
            "weight_decay": 1e-5,
            "epochs": 1,
            "batch_size": 4,
        }
        for kind in ("fm", "mlp", "deepfm"):
            with self.subTest(kind=kind):
                bundle = fit_torch_ranker(train, kind, config, seed=42)
                scores = bundle.predict(train)
                self.assertEqual(scores.shape, (len(train),))
                self.assertTrue(np.isfinite(scores).all())
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / f"{kind}.joblib"
                    joblib.dump(bundle, path)
                    restored = joblib.load(path)
                    np.testing.assert_allclose(scores, restored.predict(train), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
