"""Parquet-to-Parquet normalization with before/after counts."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__:
    from .rules import DATE_FIELDS, MISSING_CODES, RANGES, identifier as q, literal, month_date
else:
    from rules import DATE_FIELDS, MISSING_CODES, RANGES, identifier as q, literal, month_date


def normalized(column):
    value = f"trim({q(column)})"
    codes = MISSING_CODES.get(column, ())
    if codes:
        return f"CASE WHEN {value} IN ({', '.join(map(literal, codes))}) THEN NULL ELSE nullif({value}, '') END"
    return f"nullif({value}, '')"


def query_rows(connection, sql):
    cursor = connection.execute(sql)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def export_parquet(connection, query, target):
    """Only write new outputs in the unique run directory."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    connection.execute(
        f"COPY ({query}) TO {literal(target.as_posix())} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
    )


def clean_file(connection, source, target, columns, batch_size=131_072):
    """Normalize, count changes and write bounded batches in one source scan."""
    import tempfile
    import time

    import pyarrow.compute as pc

    expected = pa.schema([(column, pa.string()) for column in columns])
    before = pq.read_metadata(source)
    if not before.schema.to_arrow_schema().equals(expected):
        raise ValueError(f"Unexpected Stage 1 schema: {source.name}")
    stat_before = source.stat()
    started = last_progress = time.monotonic()
    # Explicit layers share trimmed/cleaned values across output and counters.
    raw_projection, clean_projection, output_projection = [], [], []
    metrics = []
    for index, column in enumerate(columns):
        raw, trimmed, clean = f"r{index}", f"t{index}", f"c{index}"
        raw_projection += [f"{q(column)} AS {raw}", f"trim({q(column)}) AS {trimmed}"]
        codes = MISSING_CODES.get(column, ())
        condition = f"{trimmed} IN ({', '.join(map(literal, codes))})" if codes else "FALSE"
        expression = (f"CASE WHEN {condition} THEN NULL ELSE nullif({trimmed}, '') END"
                      if codes else f"nullif({trimmed}, '')")
        clean_projection.append(f"{expression} AS {clean}")
        output_projection.append(f"{clean} AS {q(column)}")
        flags = {
            "null_before": f"{raw} IS NULL",
            "trimmed": f"{raw} IS NOT NULL AND {raw} != {trimmed}",
            "blank_to_null": f"{raw} IS NOT NULL AND {trimmed} = ''",
            "changed_cells": f"{raw} IS DISTINCT FROM {clean}",
        }
        if codes:
            flags["code_to_null"] = condition
        if column in RANGES:
            low, high = RANGES[column]
            numeric = f"try_cast({clean} AS DOUBLE)"
            bad = f"{numeric} IS NULL OR NOT isfinite({numeric}) OR {numeric} < {low}"
            if high is not None:
                bad += f" OR {numeric} > {high}"
            flags["invalid_flagged"] = f"{clean} IS NOT NULL AND ({bad})"
        elif column in DATE_FIELDS:
            flags["invalid_flagged"] = f"{clean} IS NOT NULL AND {month_date(clean)} IS NULL"
        for key, expression in flags.items():
            alias = f"__audit_{index}_{key}"
            metrics.append((index, key, alias))
            output_projection.append(f"({expression}) AS {alias}")

    # Keep the cleaned columns first, followed by transient Boolean counters.
    # Counters never appear in the saved dataset.
    clean_outputs = [f"c{index} AS {q(column)}" for index, column in enumerate(columns)]
    flag_outputs = [item for item in output_projection if " AS __audit_" in item]
    query = (
        f"WITH trimmed AS (SELECT {', '.join(raw_projection)} "
        f"FROM read_parquet({literal(source.as_posix())})), "
        f"cleaned AS (SELECT *, {', '.join(clean_projection)} FROM trimmed) "
        f"SELECT {', '.join(clean_outputs + flag_outputs)} FROM cleaned"
    )
    field_changes = [
        {"file": source.name, "column": column, "rows": before.num_rows,
         "null_before": 0, "trimmed": 0, "blank_to_null": 0, "code_to_null": 0,
         "null_after": 0, "changed_cells": 0, "invalid_flagged": 0}
        for column in columns
    ]
    if target.exists():
        raise FileExistsError(target)
    temporary = None
    rows = 0
    print(f"Cleaning, counting and writing: {source.name}", flush=True)
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".partial", delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with connection.sql(query).to_arrow_reader(batch_size=batch_size) as reader:
            output_schema = pa.schema([reader.schema.field(index) for index in range(len(columns))])
            with pq.ParquetWriter(
                temporary, output_schema, compression="zstd", compression_level=3,
                use_dictionary=True,
            ) as writer:
                for batch in reader:
                    arrays = [batch.column(index) for index in range(len(columns))]
                    for index, array in enumerate(arrays):
                        field_changes[index]["null_after"] += array.null_count
                    for index, key, alias in metrics:
                        flag = batch.column(batch.schema.get_field_index(alias))
                        field_changes[index][key] += pc.sum(flag.cast(pa.int64())).as_py() or 0
                    writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=output_schema))
                    rows += batch.num_rows
                    now = time.monotonic()
                    if now - last_progress >= 5:
                        print(f"  {source.name}: {rows:,}/{before.num_rows:,} rows "
                              f"({100 * rows / max(before.num_rows, 1):.1f}%)", flush=True)
                        last_progress = now
        after = pq.read_metadata(temporary)
        if rows != before.num_rows or after.num_rows != rows or after.schema.names != columns:
            raise ValueError(f"Row/column preservation failed: {source.name}")
        stat_after = source.stat()
        if (stat_after.st_size, stat_after.st_mtime_ns) != (stat_before.st_size, stat_before.st_mtime_ns):
            raise ValueError(f"Source changed during preparation: {source.name}")
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    elapsed = time.monotonic() - started
    print(f"  Completed {source.name}: {rows:,} rows in {elapsed:.1f}s", flush=True)
    return {
        "file": source.name, "rows_before": before.num_rows, "rows_after": rows,
        "columns": len(columns), "bytes_before": stat_before.st_size,
        "bytes_after": target.stat().st_size,
        "changed_cells": sum(row["changed_cells"] for row in field_changes),
        "seconds": round(elapsed, 2), "source_passes": 1,
    }, field_changes


def file_list(paths):
    return "[" + ", ".join(literal(path.as_posix()) for path in paths) + "]"
