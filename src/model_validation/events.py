"""Narrow, cached source-history check of positive labels in the focus groups."""

from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq

from data_preparation.rules import identifier as q, literal, month_date


PERFORMANCE_FIELDS = ["LOAN IDENTIFIER", "PERIOD", "CURRENT LOAN DELINQUENCY STATUS",
                      "ZERO BALANCE CODE", "ZERO BALANCE EFFECTIVE DATE"]
TRIGGERS = {1: "90+ days delinquent", 2: "REO status (RA)", 4: "Credit exit 02",
            8: "Credit exit 03", 16: "Credit exit 09"}


def trigger_name(mask):
    return "; ".join(name for bit, name in TRIGGERS.items() if mask & bit) or "No source trigger matched saved event"


def fingerprint(path):
    info = path.stat()
    return {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def rows(connection, query):
    result = connection.execute(query)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def event_query(source, year):
    reporting = month_date(q("PERIOD"))
    effective = month_date(q("ZERO BALANCE EFFECTIVE DATE"))
    return f"""
        WITH matched AS (
            SELECT p.{q('LOAN IDENTIFIER')} AS loan_id, p.{q('PERIOD')},
                p.{q('CURRENT LOAN DELINQUENCY STATUS')}, p.{q('ZERO BALANCE CODE')},
                p.{q('ZERO BALANCE EFFECTIVE DATE')},
                s.observation_month, s.horizon_end, s.event_in_window AS expected_event
            FROM read_parquet({literal(source.as_posix())}) p JOIN positive s
                ON p.{q('LOAN IDENTIFIER')} = s.loan_id
            WHERE s.origination_cohort = {year}
        ), parsed AS (
            SELECT *, {reporting} AS report_month,
                coalesce({effective}, {reporting}) AS effective_month,
                CASE WHEN regexp_full_match({q('CURRENT LOAN DELINQUENCY STATUS')}, '[0-9]{{1,2}}')
                    THEN try_cast({q('CURRENT LOAN DELINQUENCY STATUS')} AS INTEGER) END AS delinquency
            FROM matched
        ), expanded AS (
            SELECT p.loan_id, p.report_month, p.expected_event, v.event_month, v.trigger
            FROM parsed p CROSS JOIN LATERAL (VALUES
                (CASE WHEN p.delinquency >= 3 THEN p.report_month END, 1),
                (CASE WHEN p.{q('CURRENT LOAN DELINQUENCY STATUS')} = 'RA' THEN p.report_month END, 2),
                (CASE WHEN p.{q('ZERO BALANCE CODE')} IN ('02','03','09') THEN p.effective_month END,
                    CASE p.{q('ZERO BALANCE CODE')} WHEN '02' THEN 4 WHEN '03' THEN 8 WHEN '09' THEN 16 ELSE 0 END)
            ) AS v(event_month, trigger)
            WHERE v.event_month > p.observation_month AND v.event_month <= p.horizon_end
        )
        SELECT loan_id, min(event_month) AS first_event,
            min(report_month) AS first_event_report,
            coalesce(bit_or(trigger) FILTER (WHERE event_month=expected_event), 0)::INTEGER AS trigger_mask
        FROM expanded GROUP BY loan_id
    """


def audit_events(connection, prediction_paths, prepared, root, data_folder, state, save):
    files = "[" + ", ".join(literal(path.as_posix()) for path in prediction_paths) + "]"
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW positive AS
        SELECT loan_id, group_name, origination_cohort, observation_month, horizon_end,
            event_in_window, event_report_month
        FROM read_parquet({files}) WHERE candidate_default_12m=1
    """).fetchall()
    selected = rows(connection, "SELECT count(*) AS loans, count(DISTINCT loan_id) AS ids FROM positive")[0]
    if selected["loans"] != selected["ids"]:
        raise ValueError("Duplicate positive loan IDs in the event-source audit")
    if not selected["loans"]:
        return {"status": "NO POSITIVE LOANS", "selected_loans": 0, "types": [], "checks": {}}
    years = [row[0] for row in connection.execute("SELECT DISTINCT origination_cohort FROM positive ORDER BY 1").fetchall()]
    planned = []
    for year in years:
        if year < 2012 or year > 2020:
            raise ValueError("Reserved/later origination data reached the event audit")
        cleaned = {item["file"]: item for item in prepared["results"][str(year)]["files"]}
        for quarter in range(1, 5):
            name = f"perf_{year}Q{quarter}.parquet"
            source = root / "data" / "prepared" / prepared["run_id"] / str(year) / name
            metadata = pq.read_metadata(source)
            saved = cleaned[name]
            if (metadata.num_rows != saved["rows_after"] or source.stat().st_size != saved["bytes_after"]
                    or not set(PERFORMANCE_FIELDS).issubset(metadata.schema.names)):
                raise ValueError(f"Cleaned performance file differs from preparation: {source}")
            planned.append((year, source, metadata.num_rows))
    state.update(status="RUNNING", selected_loans=selected["loans"], planned_files=len(planned),
                 planned_monthly_rows=sum(item[2] for item in planned))
    state.setdefault("files", {})
    save()
    print(f"Event-source audit: {selected['loans']:,} positive loans; {len(planned)} files; "
          f"up to {state['planned_monthly_rows']:,} monthly records, five columns only.", flush=True)
    print("Completed file checks are cached. --skip-event-history runs only the smaller model/target diagnostics.", flush=True)
    outputs = []
    for index, (year, source, _) in enumerate(planned, 1):
        key = source.stem
        target = data_folder / f"events_{key}.parquet"
        before = fingerprint(source)
        cached = state["files"].get(key)
        if cached is not None:
            if (cached["source"] != before or not target.is_file()
                    or fingerprint(target) != cached["output"]
                    or pq.read_metadata(target).num_rows != cached["rows"]):
                raise ValueError(f"Changed/unavailable event-audit cache: {target}")
            print(f"Event-source audit {index}/{len(planned)}: reusing {source.name}", flush=True)
        else:
            if target.exists():
                raise ValueError(f"Uncheckpointed event-audit output: {target}")
            print(f"Event-source audit {index}/{len(planned)}: {source.name}", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
            try:
                connection.execute("COPY (" + event_query(source, year) + ") TO " + literal(temporary.as_posix())
                                   + " (FORMAT PARQUET, COMPRESSION ZSTD)").fetchall()
                if fingerprint(source) != before:
                    raise ValueError("Performance source changed while being audited")
                temporary.rename(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            state["files"][key] = {"source": before, "output": fingerprint(target),
                                   "rows": pq.read_metadata(target).num_rows}
            save()
        outputs.append(target)
    paths = "[" + ", ".join(literal(path.as_posix()) for path in outputs) + "]"
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW reconstructed AS
        SELECT loan_id, min(first_event) AS first_event,
            min(first_event_report) AS first_report, bit_or(trigger_mask) AS trigger_mask
        FROM read_parquet({paths}) GROUP BY loan_id
    """).fetchall()
    checks = rows(connection, """
        SELECT count(*) AS positive_loans,
            count(*) FILTER (WHERE r.first_event IS NULL) AS no_source_event,
            count(*) FILTER (WHERE r.first_event IS DISTINCT FROM p.event_in_window) AS event_month_mismatch,
            count(*) FILTER (WHERE r.first_report IS DISTINCT FROM p.event_report_month) AS reporting_month_mismatch,
            count(*) FILTER (WHERE coalesce(r.trigger_mask,0)=0) AS no_matching_saved_event_trigger
        FROM positive p LEFT JOIN reconstructed r USING (loan_id)
    """)[0]
    types = rows(connection, """
        SELECT p.group_name, coalesce(r.trigger_mask,0) AS trigger_mask, count(*) AS loans
        FROM positive p LEFT JOIN reconstructed r USING (loan_id)
        GROUP BY p.group_name, coalesce(r.trigger_mask,0) ORDER BY 1,2
    """)
    for item in types:
        item["event_type"] = trigger_name(item["trigger_mask"])
    state.update(status="COMPLETE", checks=checks, types=types)
    save()
    return state
