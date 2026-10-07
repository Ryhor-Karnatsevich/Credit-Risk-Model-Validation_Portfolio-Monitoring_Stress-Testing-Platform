"""Show the origination record and complete monthly history for one mortgage."""

import argparse
import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "freddie_mac"
RESULTS_DIR = PROJECT_ROOT / "results"


READABLE_NAMES = {
    "LOAN IDENTIFIER": "Loan ID",
    "CLASSIC FICO": "Borrower FICO credit score",
    "VANTAGESCORE 4.0": "Borrower VantageScore 4.0",
    "FIRST PAYMENT DATE": "First scheduled payment month",
    "FIRST TIME HOMEBUYER INDICATOR": "First-time homebuyer code",
    "MATURITY DATE": "Scheduled final payment month",
    "METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION": "Metropolitan area code",
    "MORTGAGE INSURANCE PERCENTAGE (MI %)": "Mortgage insurance coverage (%)",
    "NUMBER OF UNITS": "Property units",
    "OCCUPANCY STATUS": "Property occupancy code",
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)": "Initial total secured debt / property value (%)",
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": "Initial debt payments / income (%)",
    "ORIGINAL UPB": "Initial unpaid principal ($)",
    "ORIGINAL LOAN-TO-VALUE (LTV)": "Initial mortgage / property value (%)",
    "ORIGINAL INTEREST RATE": "Initial annual interest rate (%)",
    "CHANNEL": "Origination channel code",
    "PREPAYMENT PENALTY INDICATOR": "Early repayment penalty code",
    "AMORTIZATION TYPE": "Loan rate type code",
    "PROPERTY STATE": "Property state",
    "PROPERTY TYPE": "Property type code",
    "POSTAL CODE": "Property postal code prefix",
    "LOAN PURPOSE": "Loan purpose code",
    "ORIGINAL LOAN TERM": "Initial loan term (months)",
    "NUMBER OF BORROWERS": "Borrowers on the loan",
    "SUPER CONFORMING FLAG": "Super-conforming loan code",
    "PRE-HARP LOAN SEQUENCE NUMBER": "Previous HARP loan ID",
    "SPECIAL ELIGIBILITY PROGRAM": "Special programme code",
    "HARP INDICATOR": "HARP programme code",
    "PROPERTY VALUATION METHOD": "Property valuation method code",
    "INTEREST ONLY (I/O) INDICATOR": "Interest-only loan code",
    "PERIOD": "Reporting month",
    "CURRENT ACTUAL UPB": "Remaining unpaid principal ($)",
    "CURRENT LOAN DELINQUENCY STATUS": "Payment delinquency code",
    "LOAN AGE": "Loan age (months)",
    "REMAINING MONTHS TO LEGAL MATURITY": "Months until scheduled maturity",
    "UNDERWRITING DEFECT AND MAJOR SERVICING DEFECT SETTLEMENT DATE": "Loan defect settlement date",
    "MODIFICATION FLAG": "Loan modification code",
    "ZERO BALANCE CODE": "Loan termination reason code",
    "ZERO BALANCE EFFECTIVE DATE": "Loan termination month",
    "CURRENT INTEREST RATE": "Current annual interest rate (%)",
    "CURRENT NON-INTEREST BEARING UPB": "Principal not accruing interest ($)",
    "DUE DATE OF LAST PAID INSTALLMENT (DDLPI)": "Due date of last paid instalment",
    "MI RECOVERIES": "Mortgage insurance recoveries ($)",
    "NET SALES PROCEEDS": "Net property sale proceeds ($)",
    "NON MI RECOVERIES": "Other recoveries ($)",
    "TOTAL EXPENSES": "Total reported expenses ($)",
    "LEGAL COSTS": "Legal costs ($)",
    "MAINTENANCE AND PRESERVATION COSTS": "Property maintenance costs ($)",
    "TAXES AND INSURANCE": "Taxes and insurance expenses ($)",
    "MISCELLANEOUS EXPENSES": "Other expenses ($)",
    "ACTUAL LOSS": "Reported actual loss ($)",
    "CUMULATIVE MODIFICATION COSTS": "Total modification costs to date ($)",
    "INTEREST RATE STEP INDICATOR": "Stepped interest rate code",
    "PAYMENT DEFERRAL FLAG": "Payment deferral code",
    "ESTIMATED LOAN-TO-VALUE (ELTV)": "Estimated current loan / property value (%)",
    "ZERO BALANCE REMOVAL UPB": "Principal at loan termination ($)",
    "DELINQUENT ACCRUED INTEREST": "Accrued overdue interest ($)",
    "DELINQUENCY DUE TO DISASTER": "Disaster-related delinquency code",
    "BORROWER ASSISTANCE PLAN": "Borrower assistance code",
    "CURRENT PERIOD MODIFICATION COSTS": "This period's modification costs ($)",
    "CURRENT INTEREST BEARING UPB": "Principal accruing interest ($)",
    "MORTGAGE INSURANCE CANCELLATION INDICATOR": "Mortgage insurance cancellation code",
    "BANKRUPTCY CRAMDOWN COSTS": "Bankruptcy debt-reduction costs ($)",
}


def prepare_display_table(table, name):
    """Simplify this loan's exports only; do not change the source dataset."""
    removed = []
    for column in table.columns:
        values = table[column].str.strip()
        if values.eq("").all():
            removed.append((column, "entirely empty"))
            continue
        # Missing values are not zeros. A mixture of zero and missing is retained.
        numbers = pd.to_numeric(values, errors="coerce")
        if numbers.notna().all() and numbers.eq(0).all():
            removed.append((column, "zero in every row"))
    print(f"\n{name}: hidden columns")
    for column, reason in removed:
        print(f"  {READABLE_NAMES.get(column, column.title())}: {reason}")
    result = table.drop(columns=[column for column, _ in removed]).copy()
    # YYYYMM dates become readable; categorical codes remain as supplied.
    for column in ("FIRST PAYMENT DATE", "MATURITY DATE", "PERIOD", "ZERO BALANCE EFFECTIVE DATE"):
        if column in result:
            result[column] = result[column].map(
                lambda value: value[:4] + "-" + value[4:]
                if len(value) == 6 and value.isdigit() else value
            )
    return result.rename(columns=lambda column: READABLE_NAMES.get(column, column.title()))


def load_headers(filename):
    return (SOURCE_DIR / "headers" / filename).read_text(
        encoding="utf-8-sig"
    ).strip().split("|")


def find_origination(files, headers, loan_id):
    id_index = headers.index("LOAN IDENTIFIER")
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.reader(file, delimiter="|"):
                if len(row) <= id_index or row[id_index] != loan_id:
                    continue
                if len(row) != len(headers):
                    raise ValueError(f"Unexpected column count in {path.name}")
                return row
    raise ValueError(f"Origination record not found for {loan_id}")


def find_performance(files, headers, loan_id):
    # Scan on disk and retain only this loan's rows in memory.
    prefix = loan_id.encode("utf-8") + b"|"
    rows = []
    for path in files:
        print(f"Reading history: {path.name}", flush=True)
        with path.open("rb") as file:
            for line in file:
                if not line.startswith(prefix):
                    continue
                row = next(csv.reader(
                    [line.decode("utf-8-sig").rstrip("\r\n")], delimiter="|"
                ))
                if len(row) != len(headers):
                    raise ValueError(f"Unexpected column count in {path.name}")
                rows.append(row)
    if not rows:
        raise ValueError(f"No performance records found for {loan_id}")
    period_index = headers.index("PERIOD")
    return sorted(rows, key=lambda row: row[period_index])


def show_loan(year=2017, loan_id="F17Q10000469"):
    folder = SOURCE_DIR / "extracted" / str(year)
    orig_files = sorted(folder.glob("orig_*.txt"))
    perf_files = sorted(folder.glob("perf_*.txt"))
    if not orig_files or not perf_files:
        raise FileNotFoundError(f"Origination or performance files missing in {folder}")

    orig_headers = load_headers("origination_data_file_header.txt")
    perf_headers = load_headers("performance_data_file_header.txt")
    orig_row = find_origination(orig_files, orig_headers, loan_id)
    perf_rows = find_performance(perf_files, perf_headers, loan_id)

    # Keep values as source strings: identifiers and codes retain leading zeros.
    orig = pd.DataFrame([orig_row], columns=orig_headers)
    perf = pd.DataFrame(perf_rows, columns=perf_headers)
    orig = prepare_display_table(orig, "Original loan")
    perf = prepare_display_table(perf, "Monthly history")

    # Export simplified tables before adding empty-field display placeholders.
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_csv = output_dir / f"orig_{year}_{loan_id}.csv"
    perf_csv = output_dir / f"perf_{year}_{loan_id}.csv"
    orig.to_csv(orig_csv, index=False, encoding="utf-8-sig")
    perf.to_csv(perf_csv, index=False, encoding="utf-8-sig")
    print(f"Origination CSV saved to: {orig_csv}")
    print(f"Performance CSV saved to: {perf_csv}")

    orig = orig.replace("", "-")
    perf = perf.replace("", "-")

    print(f"\nLoan {loan_id} | Origination cohort {year}")
    print("'-' means an empty source field. Codes and sentinel values are preserved.")
    print("Entirely empty and all-zero columns are hidden only in these loan exports.")

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.max_colwidth", None,
        "display.width", 240,
        "display.expand_frame_repr", True,
    ):
        print("\n" + "=" * 80)
        print(f"TABLE 1 — ORIGINAL LOAN RECORD ({len(orig)} row, {len(orig.columns)} columns)")
        print("=" * 80)
        print(orig.to_string(index=False))

        print("\n" + "=" * 80)
        print(f"TABLE 2 — MONTHLY HISTORY ({len(perf)} rows, {len(perf.columns)} columns)")
        print("=" * 80)
        print(perf.to_string(index=False))

    # A local HTML view fits all columns in horizontally scrollable tables.
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"loan_{year}_{loan_id}.html"
    from html import escape
    page = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Mortgage history</title>
<style>
body { font: 14px system-ui, sans-serif; background: #f4f6fa; color: #203047;
       margin: 32px; }
h1 { font-size: 26px; } h2 { font-size: 18px; }
section { background: white; border: 1px solid #dce3ec; border-radius: 12px;
          padding: 20px; margin-top: 24px; }
.scroll { overflow-x: auto; max-height: 70vh; }
table { border-collapse: collapse; white-space: nowrap; font-size: 13px; }
th { background: #203b60; color: white; position: sticky; top: 0;
     text-align: left; padding: 12px; }
td { padding: 9px 12px; border-bottom: 1px solid #e3e8ef; }
tr:nth-child(even) td { background: #f2f6fb; }
tr:hover td { background: #e4effd; }
.note { color: #596b80; }
</style></head><body>
"""
    page += f"<h1>Mortgage {escape(loan_id)}</h1>"
    page += f"<p>Origination cohort: {year}</p>"
    page += '<p class="note">Scroll horizontally to see every column. All monthly rows are included. '
    page += '"-" means an empty source field; codes and sentinel values are preserved. '
    page += 'Entirely empty and all-zero columns are hidden. The raw files are unchanged.</p>'
    page += '<section><h2>Original loan record — 1 row</h2><div class="scroll">'
    page += orig.to_html(index=False, border=0, escape=True)
    page += '</div></section><section>'
    page += f'<h2>Monthly history — {len(perf)} rows</h2><div class="scroll">'
    page += perf.to_html(index=False, border=0, escape=True)
    page += '</div></section></body></html>'
    output_path.write_text(page, encoding="utf-8")
    print(f"\nReadable browser tables saved to: {output_path}")
    print("Open that HTML file in your browser for the formatted view.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--loan-id", default="F17Q10000469")
    args = parser.parse_args()
    if not args.loan_id or any(character in args.loan_id for character in '/\\:*?"<>|'):
        parser.error("--loan-id must be a valid identifier without path characters")
    show_loan(args.year, args.loan_id)
