"""Synthetic baseline checks; not run by the assistant."""

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logistic_regression.evaluation import evaluate
from logistic_regression.train import load_training


def test_good_predictions_beat_constant_benchmark():
    metrics, bands = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], training_rate=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.025)
    assert metrics["constant_benchmark_brier"] == pytest.approx(0.25)
    assert metrics["brier_skill_vs_training_rate"] == pytest.approx(0.9)
    assert metrics["observed_expected_ratio"] == pytest.approx(1.0)
    assert metrics["log_loss"] < metrics["constant_benchmark_log_loss"]
    assert sum(band["loans"] for band in bands) == 4
    assert bands[0]["mean_pd"] == pytest.approx(0.9)


def test_unknown_target_or_invalid_probability_is_rejected():
    with pytest.raises(ValueError):
        evaluate([0, 2], [0.2, 0.5], 0.1)
    with pytest.raises(ValueError):
        evaluate([0, 1], [0.2, np.nan], 0.1)


def test_single_class_does_not_invent_an_auc():
    metrics, _ = evaluate([0, 0], [0.1, 0.2], training_rate=0.1)
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["top_10pct_event_capture"] is None


def test_sparse_loading_does_not_include_validation_or_test_rows(tmp_path):
    features = ["fico_scaled", "fico_missing", "state_unknown", "state_category_01", "state_category_02"]
    rows = []
    for index, split in enumerate(["development", "development", "development", "validation", "test"]):
        rows.append({"loan_id": str(index),
                     "observation_month": date(2012 if split == "development" else 2018 if split == "validation" else 2020, 1, 1),
                     "candidate_default_12m": index % 2 if split != "test" else 999,
                     "proposed_split": split, "fico_scaled": float(index), "fico_missing": 0,
                     "state_unknown": 0, "state_category_01": 1, "state_category_02": 0})
    path = tmp_path / "model_inputs.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    files = [(path, {"development": 3, "validation": 1, "test": 1})]
    matrix, target, stats = load_training(files, features, expected_rows=3, max_nonzero=3,
                                         batch_size=2, memory_gb=1)
    assert matrix.shape == (3, len(features))
    np.testing.assert_array_equal(target, [0, 1, 0])
    np.testing.assert_array_equal(matrix[:, 0].toarray().ravel(), [0, 1, 2])
    assert stats["events"] == 1
    assert stats["last_observation"] == "2012-01-01"
