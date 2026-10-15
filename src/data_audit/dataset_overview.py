"""Create a basic PDF overview of the full downloaded 2026 mortgage cohort.

Run: python src/data_audit/dataset_overview.py
Output: results/data_audit/dataset_overview_2026.pdf
Reads all available 2026 quarterly text files, not sample files. No downloads.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle,
    )
except ImportError as error:
    raise SystemExit(
        "Install the PDF dependency in your project environment:\n"
        "python -m pip install reportlab==4.4.9"
    ) from error

# Also supports running this file directly from PyCharm.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.schema import get_headers

YEAR = 2026
SOURCE = PROJECT_ROOT / "data" / "raw" / "freddie_mac"
OUTPUT = PROJECT_ROOT / "results" / "data_audit" / f"dataset_overview_{YEAR}.pdf"
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


def create_pdf(orig, perf):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Cell", fontName="Helvetica", fontSize=8, leading=11,
        textColor=colors.HexColor("#24354A"), spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="HeaderCell", parent=styles["Cell"],
        fontName="Helvetica-Bold", textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        alignment=TA_CENTER, textColor=colors.HexColor("#163A5F"),
    ))
    for name in ("BodyText", "Normal"):
        styles[name].fontSize = 9
        styles[name].leading = 13
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 3 * mm))

    def heading(text):
        story.append(Paragraph(escape(text), styles["Heading2"]))

    def table(rows, widths):
        cells = [
            [Paragraph(escape(str(value)), styles["HeaderCell" if index == 0 else "Cell"])
             for value in row]
            for index, row in enumerate(rows)
        ]
        grid = LongTable(cells, colWidths=[width * mm for width in widths], repeatRows=1)
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163A5F")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#163A5F")),
        ]))
        story.extend([grid, Spacer(1, 4 * mm)])

    paragraph(f"Mortgage Dataset Overview | {YEAR}", "ReportTitle")
    paragraph(f"Freddie Mac Standard dataset | Generated {datetime.now():%Y-%m-%d %H:%M}")
    paragraph(
        "Scope: all locally available full-data files for the 2026 origination cohort. "
        "This is an inventory and descriptive overview, not a completed data-quality "
        "audit or a model-readiness approval."
    )
    heading("1. What these records mean")
    paragraph(
        "Origination (orig): original loan and borrower characteristics, normally "
        "one row per mortgage. Performance (perf): monthly snapshots of loan status, "
        "not individual payment transactions. The cohort year describes when the "
        "loan originated; monthly histories can extend beyond that year."
    )
    archive = SOURCE / "zips" / f"historical_data_{YEAR}.zip"
    all_files = orig["files"] + perf["files"]
    quarters = sorted({Path(name).stem.split("Q")[-1] for name, _, _ in all_files})
    summary = [
        ["Measure", "Observed value"],
        ["Quarters present locally", ", ".join("Q" + quarter for quarter in quarters)],
        ["Extracted text files", str(len(all_files))],
        ["Extracted size", f"{sum(size for _, size, _ in all_files) / 2**20:,.2f} MiB"],
        ["Annual ZIP size", f"{archive.stat().st_size / 2**20:,.2f} MiB" if archive.exists() else "Not found locally"],
        ["Origination rows / unique non-empty loan IDs", f"{orig['rows']:,} / {len(orig['ids']):,}"],
        ["Performance rows / unique non-empty loan IDs", f"{perf['rows']:,} / {len(perf['ids']):,}"],
        ["Origination / performance column counts", f"{len(get_headers('orig'))} / {len(get_headers('perf'))}"],
        ["Observed monthly reporting range", f"{month_label(perf['first_month'])} to {month_label(perf['last_month'])}"],
        ["Performance rows per observed loan (average)", f"{perf['rows'] / len(perf['ids']):,.2f}" if perf["ids"] else "Not available"],
        ["Origination IDs without performance", f"{len(orig['ids'] - perf['ids']):,}"],
        ["Performance IDs without origination", f"{len(perf['ids'] - orig['ids']):,}"],
        ["Blank origination / performance loan IDs", f"{orig['blank_ids']:,} / {perf['blank_ids']:,}"],
        ["Invalid or blank performance reporting months", f"{perf['invalid_months']:,}"],
    ]
    heading("2. General information")
    table(summary, [95, 79])
    paragraph(
        "Recent cohorts have short follow-up. Absence of a recorded default so far "
        "does not establish a non-default outcome over the next 12 months. "
        "This report does not assign default labels or check complete loan-level "
        "12-month coverage, internal gaps, or duplicate loan-months."
    )
    story.append(PageBreak())
    heading("3. Source file inventory")
    table([["File", "Size (MiB)", "Rows"]] + [
        [name, f"{size / 2**20:,.2f}", f"{rows:,}"] for name, size, rows in all_files
    ], [90, 38, 46])
    heading("4. Original loan characteristics")
    rows = [["Characteristic", "Valid numeric rows", "Minimum", "Average", "Maximum"]]
    for name, label in LOAN_FIELDS.items():
        stats = orig["numeric"][name]
        if stats["n"]:
            values = [f"{stats['min']:,.2f}", f"{stats['sum'] / stats['n']:,.2f}", f"{stats['max']:,.2f}"]
        else:
            values = ["N/A"] * 3
        rows.append([label, f"{stats['n']:,}", *values])
    table(rows, [62, 28, 28, 28, 28])
    paragraph(
        "These are unweighted row-level summaries. Empty fields and the recognized "
        "missing codes below are excluded, as are unparseable numeric values. "
        "Other outliers are retained; plausible-value validation is a later audit check."
    )
    heading("5. Reported delinquency codes")
    table([["Source code", "Monthly rows", "Share of performance rows"]] + [
        [code or "(empty)", f"{count:,}", f"{100 * count / perf['rows']:.2f}%"]
        for code, count in sorted(perf["codes"].items())
    ], [54, 54, 66])
    paragraph(
        "This counts monthly records, not distinct defaulted loans. Codes are shown "
        "as supplied by the source; no default-event definition is applied."
    )
    for kind, data, title in (
        ("orig", orig, "6. Origination columns and missing fields"),
        ("perf", perf, "7. Performance columns and missing fields"),
    ):
        story.append(PageBreak())
        heading(title)
        paragraph(
            "Blank and recognized missing-code counts are separate. Missing (%) "
            "combines both. Real zeros are retained. The missing-code register is "
            "limited to the fields listed here; a blank code column means no sentinel "
            "check was configured, not that no such code exists."
        )
        rows = [["Column", "Blank", "Missing code", "Code count", "Missing (%)"]]
        for name in get_headers(kind):
            blank, sentinel = data["blank"][name], data["sentinel"][name]
            percent = f"{100 * (blank + sentinel) / data['rows']:.2f}" if data["rows"] else "N/A"
            rows.append([name, f"{blank:,}", ", ".join(sorted(MISSING_CODES.get(name, set()))), f"{sentinel:,}", percent])
        table(rows, [87, 23, 23, 23, 18])
    heading("Source and limitations")
    paragraph("Local source: " + str((SOURCE / "extracted" / str(YEAR)).relative_to(PROJECT_ROOT)))
    paragraph("Column order: src/data/schema.py, copied from the official July 2026 format headers.")
    paragraph("Official dataset documentation: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset")
    paragraph(
        "Counts describe these local files, not all US mortgages. The script does "
        "not certify download completeness, historical accuracy, or representative "
        "sampling. Preserve the source files unchanged."
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#627184"))
        canvas.drawString(18 * mm, 12 * mm, f"Freddie Mac {YEAR} | Basic dataset overview")
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {document.page}")
        canvas.restoreState()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # A failed build must not replace an existing report with a partial PDF.
    temporary = OUTPUT.with_suffix(".building.pdf")
    try:
        document = SimpleDocTemplate(
            str(temporary), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
            topMargin=17 * mm, bottomMargin=20 * mm,
            title=f"Freddie Mac {YEAR} - Dataset Overview", author="Credit Risk Research",
        )
        document.build(story, onFirstPage=footer, onLaterPages=footer)
        temporary.replace(OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    create_pdf(orig, perf)
    print(f"PDF saved to: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
