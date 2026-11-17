"""Synthetic time-boundary/leakage checks; do not read project datasets."""

from datetime import date
from pathlib import Path
import sys

import duckdb
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logistic_regression.preprocessing import CATEGORICAL, NUMERIC, fit_preprocessing, projection
from logistic_regression.walk_forward import create_fold_views, fold_filters


def register_source(connection, observations, availability, targets, scores=None, states=None):
    count = len(observations)
    data = {
        "loan_id": [f"loan_{index}" for index in range(count)],
        "observation_month": [date.fromisoformat(value) for value in observations],
        "assumed_label_available_month": [date.fromisoformat(value) if value else None for value in availability],
        "candidate_default_12m": pa.array(targets, type=pa.int8()),
        # Previously withheld by the global baseline; folds must use new dates.
        "proposed_split_eligible": [False] * count,
    }
    data.update({name: [100.0] * count for name in NUMERIC})
    data.update({name: ["known"] * count for name in CATEGORICAL})
    if scores is not None:
        data["fico"] = scores
    if states is not None:
        data["property_state"] = states
    connection.register("source", pa.table(data))


def test_training_needs_known_outcomes_before_year_and_restores_2017_evaluation():
    with duckdb.connect() as connection:
        register_source(connection,
            ["2012-01-01", "2013-01-01", "2015-12-01", "2017-01-01", "2020-01-01", "2017-02-01"],
            ["2013-02-01", "2014-02-01", "2017-01-01", "2018-02-01", "2021-02-01", None],
            [0, 1, 0, 1, 1, None])
        counts = create_fold_views(connection, 2017)
        rows = dict(connection.execute("SELECT loan_id, proposed_split FROM eligible").fetchall())
        assert counts["development"]["rows"] == 2
        assert counts["validation"]["rows"] == 1
        assert rows == {"loan_0": "development", "loan_1": "development", "loan_3": "validation"}


def test_2018_evaluation_requires_outcome_availability_before_final_test():
    with duckdb.connect() as connection:
        register_source(connection,
            ["2012-01-01", "2013-01-01", "2018-11-01", "2018-12-01", "2019-01-01", "2020-01-01"],
            ["2013-02-01", "2014-02-01", "2019-12-01", "2020-01-01", "2020-02-01", "2021-02-01"],
            [0, 1, 0, 1, 1, 1])
        counts = create_fold_views(connection, 2018)
        assert counts["validation"]["rows"] == 1
        assert counts["validation"]["last_observation"] == "2018-11-01"
        assert connection.execute("SELECT max(observation_month) FROM eligible").fetchone()[0] < date(2020, 1, 1)


def test_preprocessing_does_not_learn_later_median_or_category():
    with duckdb.connect() as connection:
        register_source(connection,
            ["2012-01-01", "2013-01-01", "2016-01-01"],
            ["2013-02-01", "2014-02-01", "2017-02-01"],
            [0, 1, 1], scores=[600.0, 700.0, 999.0], states=["CA", "CA", "NY"])
        create_fold_views(connection, 2016)
        fitted = fit_preprocessing(connection)
        assert fitted["numeric"]["fico"]["median"] == 650.0
        assert fitted["numeric"]["fico"]["mean"] == 650.0
        assert fitted["categorical"]["property_state"] == ["CA"]
        expressions, _ = projection(fitted)
        row = connection.sql("SELECT " + ", ".join(expressions)
                             + " FROM eligible WHERE proposed_split = 'validation'").fetchone()
        names = [column[0] for column in connection.sql("SELECT " + ", ".join(expressions)
                 + " FROM eligible LIMIT 0").description]
        assert row[names.index("property_state_unknown")] == 1


def test_final_test_cannot_be_selected_as_a_fold():
    with pytest.raises(ValueError, match="predefined"):
        fold_filters(2020)
