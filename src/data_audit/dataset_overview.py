"""Create a basic Markdown overview of the downloaded 2012 mortgage cohort.

Run: python src/data_audit/dataset_overview.py
Output: results/data_audit/dataset_overview_2012.md
Reads all available 2012 quarterly text files, not sample files. No downloads.
"""

from collections import Counter
from pathlib import Path
import sys

import pandas as pd

# Also supports running this file directly from PyCharm.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.schema import get_headers

YEAR = 2012
SOURCE = PROJECT_ROOT / "data" / "raw" / "freddie_mac"
OUTPUT = PROJECT_ROOT / "results" / "data_audit" / f"dataset_overview_{YEAR}.md"
CHUNK_SIZE = 100_000

# Only documented field-specific missing codes are excluded from summaries.
# Zero is a real value, not a general missing-value marker.
MISSING_CODES = {
    "CLASSIC FICO": {"9999"},
    "VANTAGESCORE 4.0": {"9999"},
    "MORTGAGE INSURANCE PERCENTAGE (MI %)": {"999"},
    "NUMBER OF UNITS": {"99"},
    "OCCUPANCY STATUS": {"9"},
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)": {"999"},
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": {"999"},
    "ORIGINAL LOAN-TO-VALUE (LTV)": {"999"},
    "NUMBER OF BORROWERS": {"99"},
    "FIRST TIME HOMEBUYER INDICATOR": {"9"},
    "CHANNEL": {"9"},
    "PROPERTY TYPE": {"99"},
    "LOAN PURPOSE": {"9"},
    "ESTIMATED LOAN-TO-VALUE (ELTV)": {"999"},
}
LOAN_FIELDS = {
    "CLASSIC FICO": "Borrower FICO score",
    "ORIGINAL UPB": "Initial unpaid principal (USD)",
    "ORIGINAL INTEREST RATE": "Initial annual interest rate (%)",
    "ORIGINAL LOAN-TO-VALUE (LTV)": "Initial loan / property value (%)",
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": "Initial debt payments / income (%)",
    "ORIGINAL LOAN TERM": "Initial loan term (months)",
}


def scan(kind, files):
    """Read each file once in bounded chunks; retain IDs and aggregate counts."""
    headers = get_headers(kind)
    result = {
        "rows": 0, "ids": set(), "blank_ids": 0, "files": [],
        "blank": Counter(), "sentinel": Counter(), "codes": Counter(),
        "first_month": None, "last_month": None, "invalid_months": 0,
        "numeric": {name: {"n": 0, "sum": 0.0, "min": None, "max": None}
                    for name in LOAN_FIELDS} if kind == "orig" else {},
    }
    for path in files:
        print(f"Reading {path.name} ...", flush=True)
        file_rows = 0
        for frame in pd.read_csv(
            path, sep="|", header=None, dtype="string[pyarrow]",
            keep_default_na=False, encoding="utf-8-sig",
            chunksize=CHUNK_SIZE, on_bad_lines="error",
        ):
            if frame.shape[1] != len(headers):
                raise ValueError(
                    f"{path.name}: expected {len(headers)} columns, "
                    f"found {frame.shape[1]}"
                )
            frame.columns = headers
            for name in headers:
                frame[name] = frame[name].str.strip()
                result["blank"][name] += int(frame[name].eq("").sum())
                result["sentinel"][name] += int(
                    frame[name].isin(MISSING_CODES.get(name, set())).sum()
                )
            ids = frame["LOAN IDENTIFIER"]
            result["blank_ids"] += int(ids.eq("").sum())
            result["ids"].update(ids[ids.ne("")].unique().tolist())
            if kind == "perf":
                raw_months = frame["PERIOD"]
                valid = raw_months.str.fullmatch(r"[0-9]{6}", na=False)
                valid &= raw_months.str[:4].between("1990", "2100")
                valid &= raw_months.str[4:].between("01", "12")
                result["invalid_months"] += int((~valid).sum())
                months = raw_months[valid]
                if not months.empty:
                    first, last = str(months.min()), str(months.max())
                    result["first_month"] = min(result["first_month"] or first, first)
                    result["last_month"] = max(result["last_month"] or last, last)
                result["codes"].update(
                    frame["CURRENT LOAN DELINQUENCY STATUS"].value_counts().to_dict()
                )
            else:
                for name, stats in result["numeric"].items():
                    values = frame[name]
                    usable = values.ne("") & ~values.isin(MISSING_CODES.get(name, set()))
                    numbers = pd.to_numeric(values[usable], errors="coerce").dropna()
                    if numbers.empty:
                        continue
                    stats["n"] += len(numbers)
                    stats["sum"] += float(numbers.sum())
                    low, high = float(numbers.min()), float(numbers.max())
                    stats["min"] = low if stats["min"] is None else min(stats["min"], low)
                    stats["max"] = high if stats["max"] is None else max(stats["max"], high)
            file_rows += len(frame)
        result["rows"] += file_rows
        result["files"].append((path.name, path.stat().st_size, file_rows))
        print(f"  {file_rows:,} rows", flush=True)
    return result


def month_label(value):
    return f"{value[:4]}-{value[4:]}" if value else "Not available"


def create_markdown(orig, perf):
    """Save dataset facts as simple Markdown tables."""
    archive = SOURCE / "zips" / f"historical_data_{YEAR}.zip"
    files = orig["files"] + perf["files"]
    quarters = sorted({Path(name).stem.rsplit("Q", 1)[-1] for name, _, _ in files})
    lines = [
        f"# Freddie Mac {YEAR} dataset", "", "## Coverage and files", "",
        "| Item | Value |", "|---|---|",
        f"| Origination cohort | {YEAR} |",
        f"| Available quarters | {', '.join('Q' + q for q in quarters)} |",
        f"| Reporting period | {month_label(perf['first_month'])} to {month_label(perf['last_month'])} |",
        f"| Extracted size | {sum(size for _, size, _ in files) / 2**30:,.2f} GiB |",
    ]
    if archive.exists():
        lines.append(f"| ZIP size | {archive.stat().st_size / 2**30:,.2f} GiB |")
    lines += ["", "| File | Rows | Columns | Size (MiB) |", "|---|---:|---:|---:|"]
    for kind, data in (("orig", orig), ("perf", perf)):
        lines += [
            f"| {name} | {rows:,} | {len(get_headers(kind))} | {size / 2**20:,.2f} |"
            for name, size, rows in data["files"]
        ]
    lines += ["", "## Loans and basic checks", "", "| Measure | Value |", "|---|---:|"]
    measures = [
        ("Unique origination loan IDs", f"{len(orig['ids']):,}"),
        ("Unique performance loan IDs", f"{len(perf['ids']):,}"),
        ("Monthly records per loan, mean", f"{perf['rows'] / len(perf['ids']):,.2f}"),
        ("Origination IDs without performance", f"{len(orig['ids'] - perf['ids']):,}"),
        ("Performance IDs without origination", f"{len(perf['ids'] - orig['ids']):,}"),
        ("Duplicate origination ID rows", f"{orig['rows'] - orig['blank_ids'] - len(orig['ids']):,}"),
        ("Blank origination loan IDs", f"{orig['blank_ids']:,}"),
        ("Blank performance loan IDs", f"{perf['blank_ids']:,}"),
        ("Invalid or blank reporting months", f"{perf['invalid_months']:,}"),
    ]
    lines += [f"| {label} | {value} |" for label, value in measures]
    lines += [
        "", "## Original loan characteristics", "",
        "| Field | Minimum | Mean | Maximum |", "|---|---:|---:|---:|",
    ]
    for name, label in LOAN_FIELDS.items():
        stats = orig["numeric"][name]
        values = (f"{stats['min']:,.2f} | {stats['sum'] / stats['n']:,.2f} | {stats['max']:,.2f}"
                  if stats["n"] else "N/A | N/A | N/A")
        lines.append(f"| {label} | {values} |")
    lines += [
        "", "## Missing values", "",
        "| Dataset | Blank cells | Recognized missing-code cells |", "|---|---:|---:|",
    ]
    for kind, data in (("Origination", orig), ("Performance", perf)):
        lines.append(f"| {kind} | {sum(data['blank'].values()):,} | {sum(data['sentinel'].values()):,} |")
    lines += [
        "", "## Delinquency status", "",
        "| Source code | Monthly records | Share |", "|---|---:|---:|",
    ]
    for code, count in sorted(perf["codes"].items()):
        lines.append(f"| {code or '(blank)'} | {count:,} | {100 * count / perf['rows']:.2f}% |")
    lines += [
        "", "Numeric summaries exclude blanks, recognized missing codes and non-numeric values. "
        "Real zeros are retained. Missing-code counts cover configured fields only.", "",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    folder = SOURCE / "extracted" / str(YEAR)
    files = {kind: sorted(folder.glob(f"{kind}_{YEAR}Q[1-4].txt")) for kind in ("orig", "perf")}
    if not all(files.values()):
        raise FileNotFoundError(f"Full origination and performance files are required in {folder}")
    quarters = {kind: {path.stem.rsplit('Q', 1)[-1] for path in paths} for kind, paths in files.items()}
    if quarters["orig"] != quarters["perf"]:
        raise ValueError("Origination and performance quarterly files do not match")
    orig = scan("orig", files["orig"])
    perf = scan("perf", files["perf"])
    if not orig["rows"] or not perf["rows"]:
        raise ValueError("Source files contain no origination or performance records")
    create_markdown(orig, perf)
    print(f"Markdown saved to: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
