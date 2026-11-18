"""Small synthetic diagnostics checks; never read project datasets or fit the PD model."""

from datetime import date
from pathlib import Path
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logistic_regression.preprocessing import NUMERIC, CATEGORICAL, projection
from model_validation.diagnostics import (auc_interval, calibration_diagnostics, date_audit,
    drift, month_resampling, plot_calibration, summary_groups)
from model_validation.events import audit_events
from model_validation.pipeline import CACHE_COLUMNS, cache_predictions, diagnose, setup_views
from model_validation.report import write_report


def test_calibration_recovers_known_log_odds_relationship():
    truth = np.repeat([0.02, 0.05, 0.10, 0.20, 0.40], 1000)
    y = np.concatenate([np.r_[np.ones(int(p*1000)), np.zeros(1000-int(p*1000))]
                        for p in [0.02, 0.05, 0.10, 0.20, 0.40]])
    correct = calibration_diagnostics(y, truth)
    assert correct["converged"]
    assert correct["citl"] == pytest.approx(0, abs=1e-7)
    assert correct["slope"] == pytest.approx(1, abs=1e-5)
    distorted = expit((logit(truth)-0.4)/0.7)
    fitted = calibration_diagnostics(y, distorted)
    assert fitted["converged"]
    assert fitted["intercept"] == pytest.approx(0.4, abs=1e-5)
    assert fitted["slope"] == pytest.approx(0.7, abs=1e-5)
    assert calibration_diagnostics(y, truth/2)["citl"] > 0


def test_degenerate_calibration_and_auc_are_reported_not_invented():
    assert calibration_diagnostics([0, 0], [0.1, 0.2])["slope"] is None
    assert calibration_diagnostics([0, 1], [0.1, 0.1])["slope"] is None
    assert auc_interval([0, 0], [0.1, 0.2])["low"] is None
    interval = auc_interval([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert interval["low"] < 0.75 < interval["high"]
    with pytest.raises(ValueError):
        calibration_diagnostics([0, 1], [0.1, 1.2])


def test_month_resampling_is_deterministic_and_identical_drift_is_zero():
    months = [{"dimension": "month", "events": 20, "expected_events": 10} for _ in range(12)]
    interval = month_resampling(months, repetitions=100)
    assert interval["low"] == interval["high"] == 2
    assert interval == month_resampling(months, repetitions=100)
    assert month_resampling(months[:1])["low"] is None
    profile = [{"dimension": "dti", "segment": "Missing", "loans": 20},
               {"dimension": "dti", "segment": "<=30", "loans": 80}]
    result = drift(profile, profile)[0]
    assert result["psi"] == result["total_variation"] == 0
    assert result["missing_training"] == 0.2


def candidate_records():
    records = []
    for index in range(11):
        year = 2020 if index < 10 else 2021
        positive = index == 0
        records.append({"loan_id": f"loan_{index:02d}", "observation_month": date(year, 1, 1),
            "candidate_default_12m": int(positive), "assumed_label_available_month": date(year+1, 2, 1),
            "target_policy": "verified_repayment_v2", "origination_cohort": 2020,
            "horizon_end": date(year+1, 1, 1), "event_in_window": date(year, 2, 1) if positive else None,
            "event_report_month": date(year, 2, 1) if positive else None, "first_exit_month": None,
            "future_months": 12, "future_unknown_months": 0, "internal_gap_months": 0,
            "reclassified_as_repaid": False, "payoff_confirmation_month": None,
            **{name: 100.0 for name in NUMERIC}, **{name: "known" for name in CATEGORICAL}})
        records[-1]["borrowers_code"] = 1
        records[-1]["property_state"] = "CA"
        if index % 2:
            records[-1]["dti"] = None
    return records


def test_frozen_scoring_cache_groups_and_dates_exclude_reserved_years(tmp_path):
    candidates = tmp_path / "candidates.parquet"
    pq.write_table(pa.Table.from_pylist(candidate_records()), candidates)
    splits = tmp_path / "assignments.parquet"
    pq.write_table(pa.table({"loan_id": pa.array([], type=pa.string()),
                            "is_holdout": pa.array([], type=pa.bool_())}), splits)
    fitted = {"numeric": {name: {"median": 100.0, "mean": 100.0, "scale": 1.0} for name in NUMERIC},
              "categorical": {name: ["single" if name == "borrowers_code" else "CA" if name == "property_state" else "known"]
                              for name in CATEGORICAL}}
    expressions, features = projection(fitted)
    expected = {"loans": 10, "bad_events": 1, "mean_predicted_pd": 0.1}
    target = tmp_path / "predictions.parquet"
    with duckdb.connect() as connection:
        setup_views(connection, [candidates], [splits])
        cached = cache_predictions(connection, "oot_2020", target, None, expected,
                                   expressions, features, np.zeros(len(features)), float(logit(0.1)))
        assert pq.read_schema(target).names == CACHE_COLUMNS
        assert pq.read_metadata(target).num_rows == 10
        assert "loan_10" not in pq.read_table(target, columns=["loan_id"])["loan_id"].to_pylist()
        assert cached == cache_predictions(connection, "oot_2020", target, cached, expected,
                                           expressions, features, np.zeros(len(features)), float(logit(0.1)))
        connection.execute(f"CREATE TEMP VIEW evaluation AS SELECT * FROM read_parquet('{target.as_posix()}')").fetchall()
        grouped = summary_groups(connection, "evaluation")
        for dimension in {row["dimension"] for row in grouped}:
            selected = [row for row in grouped if row["dimension"] == dimension]
            assert sum(row["loans"] for row in selected) == 10
            assert sum(row["events"] for row in selected) == 1
            assert sum(row["expected_events"] for row in selected) == pytest.approx(1)
        assert [row for row in grouped if row["dimension"] == "dti_missing" and row["segment"] == "Missing"][0]["loans"] == 5
        checks = date_audit(connection)
        assert all(value == 0 for value in checks.values())
        diagnostic = diagnose(connection, "oot_2020", target,
                              summary_groups(connection, "evaluation", outcomes=False), 0.1)
        assert diagnostic["metrics"]["brier_score"] == pytest.approx(0.09)
        assert diagnostic["metrics"]["roc_auc"] == 0.5
        assert diagnostic["event_calendar"] == [{"event_month": "2020-02", "loans": 1}]
        plot_calibration(tmp_path / "calibration.png", diagnostic["bands"], "Synthetic check")
        state = {"status": "SYNTHETIC", "prepared_run": "test", "model_experiment": "synthetic",
                 "diagnostics": {"oot_2020": diagnostic}, "signature": {"focus": ["oot_2020"]},
                 "plots": {"oot_2020": "calibration.png"}, "event_source": {"status": "SKIPPED", "files": {}}}
        write_report(tmp_path / "report.md", state)
        assert (tmp_path / "calibration.png").stat().st_size > 0
        text = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "calibration.png" in text and "SKIPPED" in text and "2021-2023" in text
        connection.execute(f"""CREATE OR REPLACE TEMP VIEW evaluation AS
            SELECT * REPLACE (CASE WHEN loan_id='loan_01' THEN DATE '2020-02-01'
                ELSE event_in_window END AS event_in_window) FROM read_parquet('{target.as_posix()}')
        """).fetchall()
        assert date_audit(connection)["zero_with_recorded_bad_event"] == 1
        with pytest.raises(ValueError, match="reproduce"):
            cache_predictions(connection, "oot_2020", tmp_path / "wrong.parquet", None, expected,
                              expressions, features, np.zeros(len(features)), float(logit(0.2)))
        assert not (tmp_path / "wrong.parquet").exists()


def test_source_events_reconstruct_ties_backdating_and_unmatched_cases(tmp_path):
    source = pa.table({
        "LOAN IDENTIFIER": ["A", "A", "B", "B", "C"],
        "PERIOD": ["202003", "202004", "202003", "202004", "202003"],
        "CURRENT LOAN DELINQUENCY STATUS": ["3", "4", "3", "4", "0"],
        "ZERO BALANCE CODE": [None, "03", None, "03", "15"],
        "ZERO BALANCE EFFECTIVE DATE": [None, "202003", None, "202002", "202003"],
    })
    folder = tmp_path / "data/prepared/test/2020"
    folder.mkdir(parents=True)
    files = []
    for quarter in range(1, 5):
        path = folder / f"perf_2020Q{quarter}.parquet"
        pq.write_table(source if quarter == 1 else source.slice(0, 0), path)
        files.append({"file": path.name, "rows_after": pq.read_metadata(path).num_rows,
                      "bytes_after": path.stat().st_size})
    selected = pa.Table.from_pylist([
        {"loan_id": loan, "group_name": "oot_2020", "origination_cohort": 2020,
         "observation_month": date(2020, 1, 1), "horizon_end": date(2021, 1, 1),
         "candidate_default_12m": 1, "event_in_window": event, "event_report_month": date(2020, 3, 1)}
        for loan, event in (("A", date(2020, 3, 1)), ("B", date(2020, 2, 1)), ("C", date(2020, 3, 1)))])
    predictions = tmp_path / "selected.parquet"
    pq.write_table(selected, predictions)
    prepared = {"run_id": "test", "results": {"2020": {"files": files}}}
    state = {"files": {}}
    with duckdb.connect() as connection:
        result = audit_events(connection, [predictions], prepared, tmp_path, tmp_path / "cache", state, lambda: None)
        assert result["checks"]["positive_loans"] == 3
        assert result["checks"]["no_source_event"] == 1
        assert result["checks"]["event_month_mismatch"] == 1
        assert {row["trigger_mask"] for row in result["types"]} == {0, 8, 9}
        assert sum(row["loans"] for row in result["types"]) == 3
        assert len(result["files"]) == 4
        assert audit_events(connection, [predictions], prepared, tmp_path, tmp_path / "cache", state, lambda: None)["checks"] == result["checks"]
