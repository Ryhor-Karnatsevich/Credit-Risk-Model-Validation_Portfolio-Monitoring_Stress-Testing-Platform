"""Small synthetic checks; no project loan data or model training is needed.

Written but not run by the assistant. Optional: pytest tests/test_repayment_targets.py
"""

from datetime import date
from pathlib import Path
import sys

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_preparation.repayment_targets import POLICY, refine_targets


def test_repayments_preserve_bad_events_and_unknowns(tmp_path):
    loans = ["paid", "bad_then_paid", "sale", "gap", "unknown_status",
             "blank_at_payoff", "late_confirmation", "already_good", "conflicting_exit"]
    rows = []
    for loan in loans:
        label = 1 if loan == "bad_then_paid" else (0 if loan == "already_good" else None)
        reason = ("observed_default_12m" if label == 1 else
                  "observed_nondefault_12m" if label == 0 else "early_exit_censored")
        rows.append({"loan_id": loan, "observation_month": date(2012, 1, 1),
                     "horizon_end": date(2013, 1, 1), "first_exit_month": date(2012, 3, 1),
                     "candidate_default_12m": label, "outcome_reason": reason,
                     "assumed_label_available_month": date(2013, 2, 1),
                     "proposed_split": "development", "proposed_split_eligible": label is not None})
    source = tmp_path / "base.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    performance = []

    def add(loan, month, status="00", code=None, effective=None):
        performance.append({"LOAN IDENTIFIER": loan, "PERIOD": month,
                            "CURRENT LOAN DELINQUENCY STATUS": status,
                            "ZERO BALANCE CODE": code, "ZERO BALANCE EFFECTIVE DATE": effective})

    for loan in loans:
        add(loan, "201201")
        if loan != "gap":
            add(loan, "201202", status=None if loan == "unknown_status" else
                "03" if loan == "bad_then_paid" else "00")
        add(loan, "201203", status=None if loan == "blank_at_payoff" else "00",
            code="15" if loan == "sale" else None if loan == "late_confirmation" else "01",
            effective="201203")
        if loan == "late_confirmation":
            add(loan, "201901", code="01", effective="201203")
        if loan == "conflicting_exit":
            add(loan, "201204", code="15", effective="201203")
    perf = tmp_path / "perf.parquet"
    pq.write_table(pa.Table.from_pylist(performance), perf)
    target, checkpoint = tmp_path / "corrected.parquet", tmp_path / "targets.json"
    with duckdb.connect() as connection:
        saved = refine_targets(connection, source, [perf], target, checkpoint, lag_months=1)
        result = {row["loan_id"]: row for row in pq.read_table(target).to_pylist()}
        assert result["paid"]["candidate_default_12m"] == 0
        assert result["paid"]["proposed_split_eligible"] is True
        assert result["blank_at_payoff"]["candidate_default_12m"] == 0
        assert result["bad_then_paid"]["candidate_default_12m"] == 1
        assert result["already_good"]["candidate_default_12m"] == 0
        for loan in ("sale", "gap", "unknown_status", "conflicting_exit"):
            assert result[loan]["candidate_default_12m"] is None
            assert result[loan]["proposed_split_eligible"] is False
        assert result["late_confirmation"]["candidate_default_12m"] == 0
        assert result["late_confirmation"]["assumed_label_available_month"] == date(2019, 2, 1)
        assert result["late_confirmation"]["proposed_split_eligible"] is False
        assert all(row["target_policy"] == POLICY for row in result.values())
        assert saved["stats"]["defaults_before"] == saved["stats"]["defaults_after"] == 1
        assert saved["stats"]["changed_known_labels"] == 0
        assert saved["stats"]["unsupported_zero_labels"] == 0
        assert len(result) == len(rows)
        assert {row["loan_id"]: row["candidate_default_12m"] for row in pq.read_table(source).to_pylist()} == {
            row["loan_id"]: row["candidate_default_12m"] for row in rows}
        before_mtime = target.stat().st_mtime_ns
        reused = refine_targets(connection, source, [perf], target, checkpoint, lag_months=1)
        assert reused == saved
        assert target.stat().st_mtime_ns == before_mtime
