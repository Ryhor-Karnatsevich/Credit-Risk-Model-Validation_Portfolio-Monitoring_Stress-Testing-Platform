"""Loan-history diagnostics and a provisional, explicitly flagged review cohort."""

if __package__:
    from .cleaning import export_parquet, file_list, query_rows
    from .rules import CREDIT_EXIT_CODES, FEATURES, KNOWN_EXIT_CODES, RANGES, identifier as q, literal, month_date
else:
    from cleaning import export_parquet, file_list, query_rows
    from rules import CREDIT_EXIT_CODES, FEATURES, KNOWN_EXIT_CODES, RANGES, identifier as q, literal, month_date


def prepare_cohort(connection, orig_files, perf_files, output, year, lag_months=1, blank_ids=None):
    connection.execute(f"CREATE OR REPLACE TEMP VIEW orig AS SELECT * FROM read_parquet({file_list(orig_files)})")
    connection.execute(f"CREATE OR REPLACE TEMP VIEW perf AS SELECT * FROM read_parquet({file_list(perf_files)})")
    credit = ", ".join(map(literal, CREDIT_EXIT_CODES))
    exits = ", ".join(map(literal, KNOWN_EXIT_CODES))
    month = month_date(q("PERIOD"))
    exit_month = month_date(q("ZERO BALANCE EFFECTIVE DATE"))
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW p AS
        SELECT {q('LOAN IDENTIFIER')} AS loan_id, {month} AS month,
            try_cast({q('CURRENT ACTUAL UPB')} AS DOUBLE) AS balance,
            {q('CURRENT LOAN DELINQUENCY STATUS')} AS status,
            CASE WHEN regexp_full_match({q('CURRENT LOAN DELINQUENCY STATUS')}, '[0-9]{{1,2}}')
                 THEN try_cast({q('CURRENT LOAN DELINQUENCY STATUS')} AS INTEGER) END AS delinquency,
            {q('ZERO BALANCE CODE')} AS exit_code,
            CASE WHEN {q('ZERO BALANCE CODE')} IS NOT NULL
                 THEN coalesce({exit_month}, {month}) END AS exit_month
        FROM perf
    """)
    print("Grouping loan-month histories ...", flush=True)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE monthly_all AS
        SELECT loan_id, month, count(*) AS records,
            count(*) FILTER (WHERE exit_code IS NOT NULL AND exit_code NOT IN ({exits})) AS unknown_exit_rows,
            max(delinquency) AS delinquency,
            max(balance) AS balance,
            bool_or(status IS NULL OR (delinquency IS NULL AND status != 'RA')) AS unknown_status,
            min(least(CASE WHEN delinquency >= 3 OR status = 'RA' THEN month END,
                      CASE WHEN exit_code IN ({credit}) THEN exit_month END)) AS event_month,
            min(exit_month) AS exit_month
        FROM p WHERE loan_id IS NOT NULL GROUP BY loan_id, month
    """)
    # Parse/read performance columns once. Invalid dates stay in a NULL-month
    # group for diagnostics but are not treated as observed calendar months.
    connection.execute("CREATE OR REPLACE TEMP VIEW monthly AS SELECT * FROM monthly_all WHERE month IS NOT NULL")
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE perf_flags AS
        SELECT loan_id, sum(records) AS perf_rows,
            coalesce(sum(records) FILTER (WHERE month IS NULL), 0) AS invalid_month_rows,
            sum(unknown_exit_rows) AS unknown_exit_rows
        FROM monthly_all GROUP BY loan_id
    """)
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE baseline AS
        SELECT loan_id, min(month) AS observation_month FROM monthly GROUP BY loan_id
    """)
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE history AS
        SELECT m.loan_id, b.observation_month,
            CAST(b.observation_month + INTERVAL '12 months' AS DATE) AS horizon_end,
            max(m.month) AS last_month, count(*) AS months_observed,
            date_diff('month', min(m.month), max(m.month)) + 1 - count(*) AS internal_gap_months,
            sum(m.records - 1) AS duplicate_month_rows,
            sum(m.records - 1) FILTER (WHERE m.month BETWEEN b.observation_month
                AND b.observation_month + INTERVAL '12 months') AS duplicate_window_rows,
            bool_or(m.month = b.observation_month AND m.delinquency = 0
                AND NOT m.unknown_status AND m.records = 1
                AND isfinite(m.balance) AND m.balance > 0) AS baseline_current,
            min(m.exit_month) AS first_exit_month,
            count(*) FILTER (WHERE m.month > b.observation_month
                AND m.month <= b.observation_month + INTERVAL '12 months') AS future_months,
            count(*) FILTER (WHERE m.month > b.observation_month
                AND m.month <= b.observation_month + INTERVAL '12 months'
                AND m.unknown_status) AS future_unknown_months,
            min(m.event_month) FILTER (WHERE m.event_month > b.observation_month
                AND m.event_month <= b.observation_month + INTERVAL '12 months') AS event_in_window,
            min(m.month) FILTER (WHERE m.event_month > b.observation_month
                AND m.event_month <= b.observation_month + INTERVAL '12 months') AS event_report_month,
            min(m.event_month) AS first_event_month
        FROM monthly m JOIN baseline b USING (loan_id)
        GROUP BY m.loan_id, b.observation_month
    """)
    features = []
    for field, (alias, dtype) in FEATURES.items():
        value = f"try_cast({q(field)} AS {dtype})"
        if dtype != "VARCHAR":
            condition = f"isfinite({value})"
            if field in RANGES:
                low, high = RANGES[field]
                condition += f" AND {value} >= {low}"
                if high is not None:
                    condition += f" AND {value} <= {high}"
            value = f"CASE WHEN {condition} THEN {value} END"
        features.append(f"CASE WHEN count(*) = 1 THEN max({value}) END AS {q(alias)}")
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE orig_profile AS
        SELECT {q('LOAN IDENTIFIER')} AS loan_id, count(*) AS orig_records,
            CASE WHEN count(*) = 1 THEN max({q('AMORTIZATION TYPE')}) END AS amortization_type,
            {', '.join(features)}
        FROM orig WHERE {q('LOAN IDENTIFIER')} IS NOT NULL GROUP BY {q('LOAN IDENTIFIER')}
    """)
    feature_names = ", ".join(f"o.{q(alias)}" for alias, _ in FEATURES.values())
    print("Building provisional review cohort ...", flush=True)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE cohort_base AS
        WITH ids AS (SELECT loan_id FROM orig_profile UNION SELECT loan_id FROM perf_flags)
        SELECT ids.loan_id, {year} AS origination_cohort, {feature_names},
            o.amortization_type,
            coalesce(o.orig_records, 0) AS orig_records,
            coalesce(pf.perf_rows, 0) AS perf_rows,
            coalesce(pf.invalid_month_rows, 0) AS invalid_month_rows,
            coalesce(pf.unknown_exit_rows, 0) AS unknown_exit_rows,
            h.* EXCLUDE (loan_id),
            CASE
                WHEN o.loan_id IS NULL THEN 'missing_origination'
                WHEN o.orig_records != 1 THEN 'duplicate_origination_id'
                WHEN o.amortization_type IS NULL OR o.amortization_type != 'FRM' THEN 'not_confirmed_fixed_rate'
                WHEN pf.loan_id IS NULL THEN 'missing_performance'
                WHEN pf.invalid_month_rows > 0 THEN 'invalid_reporting_month'
                WHEN coalesce(h.duplicate_window_rows, 0) > 0 THEN 'duplicate_loan_month_in_window'
                WHEN NOT coalesce(h.baseline_current, FALSE)
                  OR h.first_event_month <= h.observation_month
                  OR h.first_exit_month <= h.observation_month THEN 'baseline_not_current_or_already_exited'
                WHEN h.event_in_window IS NOT NULL THEN 'observed_default_12m'
                WHEN h.first_exit_month <= h.horizon_end THEN 'early_exit_censored'
                WHEN h.last_month < h.horizon_end THEN 'insufficient_followup'
                WHEN h.future_months != 12 THEN 'missing_window_months'
                WHEN h.future_unknown_months > 0 THEN 'unknown_status_in_window'
                ELSE 'observed_nondefault_12m'
            END AS outcome_reason
        FROM ids LEFT JOIN orig_profile o USING (loan_id)
            LEFT JOIN perf_flags pf USING (loan_id) LEFT JOIN history h USING (loan_id)
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE cohort AS
        WITH labeled AS (
            SELECT *, CASE WHEN outcome_reason = 'observed_default_12m' THEN 1
                           WHEN outcome_reason = 'observed_nondefault_12m' THEN 0 END AS candidate_default_12m,
                CAST(greatest(horizon_end, coalesce(event_report_month, horizon_end))
                    + INTERVAL '{lag_months} months' AS DATE) AS assumed_label_available_month,
                CASE WHEN observation_month < DATE '2018-01-01' THEN 'development'
                     WHEN observation_month < DATE '2020-01-01' THEN 'validation'
                     WHEN observation_month < DATE '2024-01-01' THEN 'test'
                     ELSE 'reserve' END AS proposed_split
            FROM cohort_base
        )
        SELECT *, CASE
            WHEN candidate_default_12m IS NULL THEN FALSE
            WHEN proposed_split = 'development' THEN assumed_label_available_month < DATE '2018-01-01'
            WHEN proposed_split = 'validation' THEN assumed_label_available_month < DATE '2020-01-01'
            WHEN proposed_split = 'test' THEN TRUE ELSE FALSE END AS proposed_split_eligible,
            TRUE AS research_rules_require_review
        FROM labeled
    """)
    export_parquet(connection, "SELECT * FROM cohort", output / "cohort_candidates.parquet")
    metrics = query_rows(connection, """
        SELECT count(*) AS loan_ids,
            count(*) FILTER (WHERE orig_records = 0) AS performance_only_ids,
            count(*) FILTER (WHERE perf_rows = 0) AS origination_only_ids,
            count(*) FILTER (WHERE orig_records > 1) AS duplicate_origination_ids,
            count(*) FILTER (WHERE duplicate_month_rows > 0) AS loans_with_duplicate_months,
            count(*) FILTER (WHERE internal_gap_months > 0) AS loans_with_internal_gaps,
            coalesce(sum(internal_gap_months), 0) AS internal_gap_months,
            coalesce(sum(invalid_month_rows), 0) AS invalid_reporting_rows,
            coalesce(sum(unknown_exit_rows), 0) AS unrecognized_exit_rows,
            count(*) FILTER (WHERE candidate_default_12m = 1) AS provisional_defaults,
            count(*) FILTER (WHERE candidate_default_12m = 0) AS provisional_nondefaults,
            count(*) FILTER (WHERE candidate_default_12m IS NULL) AS unlabelled,
            min(observation_month) AS first_observation_month,
            max(last_month) AS last_reporting_month
        FROM cohort
    """)[0]
    for kind, view in (("origination", "orig"), ("performance", "perf")):
        metrics[f"blank_{kind}_ids"] = (
            blank_ids[kind] if blank_ids is not None else connection.execute(
                f"SELECT count(*) FROM {view} WHERE {q('LOAN IDENTIFIER')} IS NULL"
            ).fetchone()[0]
        )
    outcomes = query_rows(connection, "SELECT outcome_reason, count(*) AS loans FROM cohort GROUP BY outcome_reason ORDER BY outcome_reason")
    splits = query_rows(connection, """
        SELECT proposed_split, count(*) AS all_candidates,
            count(*) FILTER (WHERE proposed_split_eligible) AS eligible_candidates,
            count(*) FILTER (WHERE candidate_default_12m IS NOT NULL AND NOT proposed_split_eligible)
                AS labelled_but_not_split_eligible
        FROM cohort GROUP BY proposed_split ORDER BY proposed_split
    """)
    # Free large year-specific tables before the next year's work.
    connection.execute("DROP VIEW monthly")
    for table in ("cohort", "cohort_base", "orig_profile", "history", "baseline", "monthly_all", "perf_flags"):
        connection.execute(f"DROP TABLE {table}")
    return {"metrics": metrics, "outcomes": outcomes, "splits": splits}
