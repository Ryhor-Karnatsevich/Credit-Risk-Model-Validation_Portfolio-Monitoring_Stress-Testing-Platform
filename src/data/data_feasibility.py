"""Validate the raw files collected for the data-feasibility stage."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FREDDIE_FILES = {
    "origination header": RAW_DATA_DIR
    / "freddie_mac"
    / "headers"
    / "origination_data_file_header.txt",
    "performance header": RAW_DATA_DIR
    / "freddie_mac"
    / "headers"
    / "performance_data_file_header.txt",
    "origination sample": RAW_DATA_DIR
    / "freddie_mac"
    / "sample"
    / "Sample Files"
    / "origination_sample_file.txt",
    "performance sample": RAW_DATA_DIR
    / "freddie_mac"
    / "sample"
    / "Sample Files"
    / "performance_sample_file.txt",
    "file layout": RAW_DATA_DIR / "freddie_mac" / "file_layout_july_2026.xlsx",
}

FRED_FILES = {
    "UNRATE": RAW_DATA_DIR / "fred" / "UNRATE.csv",
    "USSTHPI": RAW_DATA_DIR / "fred" / "USSTHPI.csv",
    "MORTGAGE30US": RAW_DATA_DIR / "fred" / "MORTGAGE30US.csv",
    "FEDFUNDS": RAW_DATA_DIR / "fred" / "FEDFUNDS.csv",
}


def require_files(files: dict[str, Path]) -> None:
    """Raise a clear error when a required raw file is missing or empty."""
    missing = [name for name, path in files.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty files: {', '.join(missing)}")


def pipe_file_summary(path: Path, loan_id_index: int) -> tuple[int, int]:
    """Return row count and unique loan count for a pipe-delimited loan file."""
    row_count = 0
    loan_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file, delimiter="|"):
            if not row:
                continue
            row_count += 1
            if len(row) <= loan_id_index:
                raise ValueError(
                    f"Row in {path.name} has {len(row)} columns; "
                    f"loan ID index {loan_id_index} is unavailable"
                )
            loan_ids.add(row[loan_id_index])
    return row_count, len(loan_ids)


def fred_summary(path: Path, series_id: str) -> tuple[int, str, str]:
    """Return usable observation count and date range for a FRED CSV file."""
    dates: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        expected_columns = {"observation_date", series_id}
        if not reader.fieldnames or not expected_columns.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected columns in {path.name}: {reader.fieldnames}")
        for row in reader:
            value = row[series_id].strip()
            if value not in {"", "."}:
                dates.append(row["observation_date"])
    if not dates:
        raise ValueError(f"No usable observations in {path.name}")
    return len(dates), min(dates), max(dates)


def main() -> None:
    """Run the Stage 0 raw-data checks and print a compact inventory."""
    require_files(FREDDIE_FILES)
    require_files(FRED_FILES)

    origination_rows, origination_loans = pipe_file_summary(
        FREDDIE_FILES["origination sample"], loan_id_index=19
    )
    performance_rows, performance_loans = pipe_file_summary(
        FREDDIE_FILES["performance sample"], loan_id_index=0
    )

    print("Freddie Mac sample")
    print(f"  Origination: {origination_rows:,} rows, {origination_loans:,} loans")
    print(f"  Performance: {performance_rows:,} rows, {performance_loans:,} loans")

    print("FRED macro series")
    for series_id, path in FRED_FILES.items():
        observations, start_date, end_date = fred_summary(path, series_id)
        print(
            f"  {series_id}: {observations:,} observations, "
            f"{start_date} to {end_date}"
        )

    print("DATA_FEASIBILITY_FILES=PASS")


if __name__ == "__main__":
    main()
