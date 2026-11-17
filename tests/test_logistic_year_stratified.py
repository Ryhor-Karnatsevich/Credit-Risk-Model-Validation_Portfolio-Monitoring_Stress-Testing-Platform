"""Synthetic checks only; no project data or model-training pipeline is run."""

from datetime import date
from pathlib import Path
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logistic_regression.preprocessing import CATEGORICAL, NUMERIC, fit_preprocessing
from logistic_regression.walk_forward import create_source
from logistic_regression.train import digest
from logistic_regression.year_stratified import YEARS, assignments, counts, pool_filter, split_indices


def test_split_preserves_class_proportions_and_is_reproducible():
    targets = np.r_[np.zeros(3600, dtype=np.int8), np.ones(400, dtype=np.int8)]
    train, holdout = split_indices(targets, 42)
    again_train, again_holdout = split_indices(targets, 42)
    assert len(train) == 3600 and len(holdout) == 400
    assert targets[train].sum() == 360 and targets[holdout].sum() == 40
    assert not set(train).intersection(holdout)
    assert set(train).union(holdout) == set(range(4000))
    np.testing.assert_array_equal(train, again_train)
    np.testing.assert_array_equal(holdout, again_holdout)
    old_train, old_holdout = train_test_split(np.arange(4000), test_size=0.25,
                                             stratify=targets, random_state=42)
    assert set(old_train).issubset(train)
    assert set(holdout).issubset(old_holdout)


def test_each_year_is_split_separately_and_holdouts_never_fit_preprocessing(tmp_path):
    records = []
    for year in YEARS:
        for index in range(100):
            records.append({
                "loan_id": f"{year}_{index:03d}", "observation_month": date(year, 1, 1),
                "assumed_label_available_month": date(year + 1, 2, 1),
                "candidate_default_12m": int(index < 20),
                **{name: float(600 + index) for name in NUMERIC},
                **{name: "CA" for name in CATEGORICAL},
            })
    # These do not qualify for the training-era population.
    for loan_id, observation, available in (
        ("later_2020", date(2020, 1, 1), date(2021, 2, 1)),
        ("late_training_outcome", date(2017, 12, 1), date(2020, 1, 1)),
    ):
        records.append({**records[0], "loan_id": loan_id, "observation_month": observation,
                        "assumed_label_available_month": available})
    with duckdb.connect() as connection:
        connection.register("input_rows", pa.Table.from_pylist(records))
        connection.execute("CREATE TEMP TABLE loan_rows AS SELECT * FROM input_rows")
        connection.execute("CREATE TEMP VIEW source AS SELECT * FROM loan_rows")
        state = {"splits": {}}
        reference_folder = tmp_path / "old_75_25"
        reference_folder.mkdir()
        reference = {"splits": {}}
        for year in YEARS:
            labels = np.asarray([int(index < 20) for index in range(100)])
            _, old_holdout = train_test_split(np.arange(100), test_size=0.25,
                                              stratify=labels, random_state=42)
            mask = np.zeros(100, dtype=bool)
            mask[old_holdout] = True
            path = reference_folder / f"split_{year}.parquet"
            pq.write_table(pa.table({"loan_id": [f"{year}_{index:03d}" for index in range(100)],
                                     "is_holdout": mask}), path)
            reference["splits"][str(year)] = {"sha256": digest(path), "holdout": 25,
                                              "holdout_events": 5}
        new_folder = tmp_path / "new_90_10"
        assignments(connection, new_folder, state, 42, lambda: None, (reference, reference_folder))
        for year in YEARS:
            item = state["splits"][str(year)]
            assert item["eligible"] == 100
            assert item["training"] == 90 and item["holdout"] == 10
            assert item["training_events"] == 18 and item["holdout_events"] == 2
            previous_mask = pq.read_table(reference_folder / f"split_{year}.parquet")["is_holdout"].to_numpy(zero_copy_only=False)
            current_mask = pq.read_table(new_folder / f"split_{year}.parquet")["is_holdout"].to_numpy(zero_copy_only=False)
            assert np.all(~current_mask | previous_mask)
        assert connection.execute("SELECT count(*) FROM population").fetchall()[0][0] == 600
        assert connection.execute("SELECT count(*) FROM population WHERE loan_id IN "
                                  "('later_2020', 'late_training_outcome')").fetchall()[0][0] == 0
        assert connection.execute("SELECT count(*) FROM (SELECT loan_id FROM assignments "
                                  "GROUP BY loan_id HAVING count(*) != 1)").fetchall()[0][0] == 0
        # Deliberately change held-out features; training settings must not learn them.
        connection.execute("UPDATE loan_rows SET fico = 999, property_state = 'NY' WHERE loan_id IN "
                           "(SELECT loan_id FROM assignments WHERE is_holdout)")
        connection.execute("CREATE TEMP VIEW eligible AS SELECT * FROM population WHERE proposed_split = 'development'")
        fitted = fit_preprocessing(connection)
        assert fitted["numeric"]["fico"]["mean"] < 700
        assert fitted["categorical"]["property_state"] == ["CA"]
        # Reuse must retain precisely the same assignments.
        connection.execute("DROP VIEW eligible")
        connection.execute("DROP VIEW population")
        connection.execute("DROP VIEW assignments")
        previous = {key: value["sha256"] for key, value in state["splits"].items()}
        assignments(connection, new_folder, state, 42, lambda: None, (reference, reference_folder))
        assert previous == {key: value["sha256"] for key, value in state["splits"].items()}


def test_source_cutoff_excludes_2021_and_later_observations(tmp_path):
    records = []
    for year in (2020, 2021, 2023):
        record = {"loan_id": str(year), "observation_month": date(year, 1, 1),
                  "candidate_default_12m": 0, "assumed_label_available_month": date(year + 1, 2, 1),
                  "target_policy": "verified_repayment_v2",
                  **{name: 100.0 for name in NUMERIC}, **{name: "CA" for name in CATEGORICAL}}
        record["borrowers_code"] = 1
        records.append(record)
    path = tmp_path / "candidates.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    with duckdb.connect() as connection:
        create_source(connection, [path], cutoff="2021-01-01")
        # create_source's validation COUNT must be exhausted before Arrow starts.
        with connection.sql("SELECT loan_id FROM source").to_arrow_reader(batch_size=16) as reader:
            assert reader.read_all()["loan_id"].to_pylist() == ["2020"]
        assert connection.execute("SELECT count(*) FROM source WHERE " + pool_filter()).fetchall()[0][0] == 0


def test_counts_finishes_result_before_next_arrow_reader():
    with duckdb.connect() as connection:
        connection.execute("""
            CREATE TEMP VIEW eligible AS
            SELECT 1 AS candidate_default_12m, DATE '2017-01-01' AS observation_month,
                DATE '2018-02-01' AS assumed_label_available_month
        """)
        assert counts(connection)["rows"] == 1
        with connection.sql("SELECT candidate_default_12m FROM eligible").to_arrow_reader(batch_size=16) as reader:
            assert reader.read_all().num_rows == 1


def test_unknown_or_too_rare_labels_are_not_silently_sampled():
    with pytest.raises(ValueError, match="binary"):
        split_indices([0, 0, None, 1], 42)
    with pytest.raises(ValueError, match="enough"):
        split_indices([0, 0, 0, 1], 42)
