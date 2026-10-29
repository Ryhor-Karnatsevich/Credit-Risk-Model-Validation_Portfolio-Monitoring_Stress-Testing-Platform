"""Refine saved candidate labels without repeating normalization or full history grouping.

Called by logistic_regression/preprocessing.py. Verified voluntary payoff (01)
with complete, unambiguous history through payoff becomes 0. Existing observed
bad events remain 1. Sales/disappearance/uncertain histories remain unknown.
Old candidate files and all cleaned Parquets are read-only.
"""

import json
from pathlib import Path
import tempfile
from uuid import uuid4

import pyarrow.parquet as pq

from .rules import CREDIT_EXIT_CODES, identifier as q, literal, month_date


POLICY = "verified_repayment_v2"
EXTRA_COLUMNS = ["original_outcome_reason", "original_candidate_default_12m",
                 "reclassified_as_repaid", "payoff_confirmation_month", "target_policy"]
PERFORMANCE_COLUMNS = ["LOAN IDENTIFIER", "PERIOD", "CURRENT LOAN DELINQUENCY STATUS",
                       "ZERO BALANCE CODE", "ZERO BALANCE EFFECTIVE DATE"]


def fingerprint(path):
    info = path.stat()
    return {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def atomic_json(path, value):
    temporary = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".targets_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def refine_targets(connection, source, performance, target, checkpoint, lag_months):
    """One narrow performance scan for early-exit candidates; versioned outputs."""
    source, target, checkpoint = map(Path, (source, target, checkpoint))
    before = pq.read_metadata(source)
    required = {"loan_id", "observation_month", "horizon_end", "first_exit_month",
                "candidate_default_12m", "outcome_reason", "assumed_label_available_month",
                "proposed_split", "proposed_split_eligible"}
    if not required.issubset(before.schema.names) or set(EXTRA_COLUMNS).intersection(before.schema.names):
        raise ValueError(f"Unexpected base candidate schema: {source}")
    inputs = [fingerprint(source)] + [fingerprint(path) for path in performance]
    signature = {"policy": POLICY, "inputs": inputs, "lag_months": lag_months}
    expected_names = before.schema.names + EXTRA_COLUMNS
    if target.exists():
        if not checkpoint.is_file():
            raise ValueError(f"Uncheckpointed target file; will not overwrite: {target}")
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        metadata = pq.read_metadata(target)
        if (saved.get("signature") != signature or metadata.schema.names != expected_names
                or metadata.num_rows != before.num_rows
                or target.stat().st_size != saved["bytes"]):
            raise ValueError(f"Saved targets are inconsistent with their inputs: {target}")
        print(f"Reusing corrected labels: {target.parent.name}", flush=True)
        return saved
    for path in performance:
        if not set(PERFORMANCE_COLUMNS).issubset(pq.read_metadata(path).schema.names):
            raise ValueError(f"Missing repayment-evidence columns: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
    try:
        connection.execute(f"CREATE TEMP VIEW repayment_base AS SELECT * FROM read_parquet({literal(source.as_posix())})")
        connection.execute("""
            CREATE TEMP TABLE repayment_candidates AS
            SELECT loan_id, observation_month, horizon_end, first_exit_month
            FROM repayment_base
            WHERE candidate_default_12m IS NULL AND outcome_reason = 'early_exit_censored'
              AND first_exit_month > observation_month AND first_exit_month <= horizon_end
        """)
        candidate_count = connection.execute("SELECT count(*) FROM repayment_candidates").fetchone()[0]
        print(f"Checking payoff evidence for {candidate_count:,} early-exit loans: {target.parent.name}", flush=True)
        if candidate_count:
            paths = "[" + ", ".join(literal(path.as_posix()) for path in performance) + "]"
            reporting = month_date(q("PERIOD"))
            effective = month_date(q("ZERO BALANCE EFFECTIVE DATE"))
            credit = ", ".join(map(literal, CREDIT_EXIT_CODES))
            # Only five monthly columns are read, and only early-exit IDs are grouped.
            connection.execute(f"""
                CREATE TEMP TABLE repayment_evidence AS
                WITH parsed AS (
                    SELECT {q('LOAN IDENTIFIER')} AS loan_id, {reporting} AS month,
                        {q('CURRENT LOAN DELINQUENCY STATUS')} AS status,
                        CASE WHEN regexp_full_match({q('CURRENT LOAN DELINQUENCY STATUS')}, '[0-9]{{1,2}}')
                            THEN try_cast({q('CURRENT LOAN DELINQUENCY STATUS')} AS INTEGER) END AS delinquency,
                        {q('ZERO BALANCE CODE')} AS exit_code,
                        coalesce({effective}, {reporting}) AS exit_month
                    FROM read_parquet({paths})
                ), matched AS (
                    SELECT p.*, c.observation_month, c.first_exit_month,
                        p.month BETWEEN c.observation_month AND c.first_exit_month AS in_history,
                        p.exit_code = '01' AND p.exit_month = c.first_exit_month
                            AND p.month >= c.first_exit_month AS confirms_payoff
                    FROM parsed p JOIN repayment_candidates c USING (loan_id)
                )
                SELECT loan_id,
                    count(*) FILTER (WHERE in_history) AS history_rows,
                    count(DISTINCT month) FILTER (WHERE in_history) AS history_months,
                    count(*) FILTER (WHERE in_history AND (delinquency IS NULL OR status = 'RA')
                        AND NOT coalesce(month = first_exit_month AND confirms_payoff, FALSE)) AS unknown_rows,
                    count(*) FILTER (WHERE in_history AND (delinquency >= 3 OR status = 'RA')) AS bad_rows,
                    count(*) FILTER (WHERE exit_code IS NOT NULL AND exit_code != '01'
                        AND exit_month <= first_exit_month) AS conflicting_exits,
                    count(*) FILTER (WHERE exit_code IN ({credit}) AND exit_month <= first_exit_month) AS credit_exits,
                    max(month) FILTER (WHERE confirms_payoff) AS confirmation_month
                FROM matched GROUP BY loan_id
            """)
        else:
            connection.execute("""
                CREATE TEMP TABLE repayment_evidence AS
                SELECT loan_id, 0::BIGINT AS history_rows, 0::BIGINT AS history_months,
                    0::BIGINT AS unknown_rows, 0::BIGINT AS bad_rows,
                    0::BIGINT AS conflicting_exits, 0::BIGINT AS credit_exits,
                    NULL::DATE AS confirmation_month
                FROM repayment_candidates
            """)
        connection.execute("""
            CREATE TEMP VIEW repayment_review AS
            SELECT c.loan_id, e.confirmation_month,
                coalesce(e.confirmation_month IS NOT NULL AND e.conflicting_exits = 0
                    AND e.credit_exits = 0 AND e.bad_rows = 0 AND e.unknown_rows = 0
                    AND e.history_rows = e.history_months
                    AND e.history_months = date_diff('month', c.observation_month, c.first_exit_month) + 1,
                    FALSE) AS verified
            FROM repayment_candidates c LEFT JOIN repayment_evidence e USING (loan_id)
        """)
        # Outcome availability remains conservatively horizon-based; it cannot
        # precede the payoff confirmation plus the existing reporting lag.
        connection.execute(f"""
            CREATE TEMP VIEW repayment_updated AS
            WITH revised AS (
                SELECT b.* REPLACE (
                    CASE WHEN coalesce(r.verified, FALSE) THEN 0 ELSE b.candidate_default_12m END AS candidate_default_12m,
                    CASE WHEN coalesce(r.verified, FALSE) THEN 'repaid_without_default_12m'
                        ELSE b.outcome_reason END AS outcome_reason,
                    CASE WHEN coalesce(r.verified, FALSE)
                        THEN CAST(greatest(b.assumed_label_available_month,
                            b.horizon_end + INTERVAL '{lag_months} months',
                            r.confirmation_month + INTERVAL '{lag_months} months') AS DATE)
                        ELSE b.assumed_label_available_month END AS assumed_label_available_month
                ),
                    b.outcome_reason AS original_outcome_reason,
                    b.candidate_default_12m AS original_candidate_default_12m,
                    coalesce(r.verified, FALSE) AS reclassified_as_repaid,
                    r.confirmation_month AS payoff_confirmation_month,
                    {literal(POLICY)} AS target_policy
                FROM repayment_base b LEFT JOIN repayment_review r USING (loan_id)
            )
            SELECT * REPLACE (
                CASE WHEN reclassified_as_repaid THEN
                    CASE WHEN proposed_split = 'development' THEN assumed_label_available_month < DATE '2018-01-01'
                         WHEN proposed_split = 'validation' THEN assumed_label_available_month < DATE '2020-01-01'
                         WHEN proposed_split = 'test' THEN TRUE ELSE FALSE END
                    ELSE proposed_split_eligible END AS proposed_split_eligible
            ) FROM revised
        """)
        result = connection.execute("""
            SELECT count(*) AS "rows",
                count(*) FILTER (WHERE reclassified_as_repaid) AS repaid_changed_to_zero,
                count(*) FILTER (WHERE reclassified_as_repaid AND proposed_split_eligible) AS repaid_now_split_eligible,
                count(*) FILTER (WHERE candidate_default_12m = 1) AS defaults_after,
                count(*) FILTER (WHERE original_candidate_default_12m = 1) AS defaults_before,
                count(*) FILTER (WHERE original_candidate_default_12m = 1
                    AND candidate_default_12m IS DISTINCT FROM 1) AS lost_default_labels,
                count(*) FILTER (WHERE original_candidate_default_12m IS NOT NULL
                    AND candidate_default_12m IS DISTINCT FROM original_candidate_default_12m) AS changed_known_labels,
                count(*) FILTER (WHERE original_candidate_default_12m IS NULL AND candidate_default_12m = 0
                    AND NOT reclassified_as_repaid) AS unsupported_zero_labels,
                count(*) FILTER (WHERE outcome_reason = 'early_exit_censored') AS early_exits_still_unknown
            FROM repayment_updated
        """)
        stats = dict(zip([item[0] for item in result.description], result.fetchone()))
        if (stats["rows"] != before.num_rows or stats["lost_default_labels"]
                or stats["changed_known_labels"] or stats["unsupported_zero_labels"]):
            raise ValueError("Repayment relabelling failed row/default/known-label preservation checks")
        connection.execute(f"COPY (SELECT * FROM repayment_updated) TO {literal(temporary.as_posix())} "
                           "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)")
        after = pq.read_metadata(temporary)
        if after.num_rows != before.num_rows or after.schema.names != expected_names:
            raise ValueError("Corrected-target output failed row/schema checks")
        if inputs != [fingerprint(source)] + [fingerprint(path) for path in performance]:
            raise ValueError("A target-construction input changed during processing")
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
        saved = {"signature": signature, "file": str(target), "bytes": target.stat().st_size, "stats": stats}
        atomic_json(checkpoint, saved)
        print(f"  Retained {stats['repaid_changed_to_zero']:,} verified repayments as 0; "
              f"preserved {stats['defaults_after']:,} bad-event labels as 1.", flush=True)
        return saved
    finally:
        if temporary.exists():
            temporary.unlink()
        for name in ("repayment_updated", "repayment_review"):
            connection.execute(f"DROP VIEW IF EXISTS {name}")
        for name in ("repayment_evidence", "repayment_candidates"):
            connection.execute(f"DROP TABLE IF EXISTS {name}")
        connection.execute("DROP VIEW IF EXISTS repayment_base")
