"""Plain Markdown change report; no PDF generation or decorative formatting."""

if __package__:
    from .rules import FEATURES, USER_GUIDE
else:
    from rules import FEATURES, USER_GUIDE


def display(value):
    if value is None:
        return "-"
    return str(value).replace("|", "/").replace("\n", " ")


def table(rows, columns):
    if not rows:
        return ["None.", ""]
    result = ["| " + " | ".join(label for _, label in columns) + " |",
              "|" + "|".join("---" for _ in columns) + "|"]
    result += ["| " + " | ".join(display(row.get(key)) for key, _ in columns) + " |" for row in rows]
    return result + [""]


def write_report(path, run):
    lines = [
        "# Data preparation results", "", f"Status: {run['status']}",
        f"Run: {run['run_id']}", f"Years: {', '.join(map(str, run['years']))}", "",
        "## What is changed", "",
        "- Source Parquets are read-only. Cleaned copies are written to a separate run folder.",
        "- Surrounding spaces are trimmed; blanks and configured missing codes become null.",
        "- No source rows or columns are intentionally removed. Zero remains a valid value.",
        "- Duplicates, gaps and implausible values are flagged, not automatically repaired.",
        "- No imputation, winsorization, scaling, encoding or model fitting is performed.",
        "- Row order is not guaranteed; join and compare records using loan ID and month.", "",
        "Cleaning counters and output are produced in one source scan. Reused files retain their original counters/timings, not a new benchmark.", "",
    ]
    for year, result in run["results"].items():
        lines += [f"## {year}", "", f"Outputs: {result['output']}", "", "### Row preservation", ""]
        lines += table(result["files"], [
            ("file", "File"), ("rows_before", "Before rows"), ("rows_after", "After rows"),
            ("columns", "Columns"), ("changed_cells", "Changed cells"),
            ("seconds", "Seconds"),
        ])
        if result.get("reused_files"):
            lines += [f"Reused completed cleaned files: {result['reused_files']}.", ""]
        lines += ["### Changed cells and flagged fields", "",
                  "Counts below are per file and column. Trimming can overlap with null conversion; do not add those columns together.", ""]
        changes = [row for row in result["changes"] if row["changed_cells"] or row["invalid_flagged"]]
        lines += table(changes, [
            ("file", "File"), ("column", "Column"), ("trimmed", "Trimmed"),
            ("blank_to_null", "Blank to null"), ("code_to_null", "Missing code to null"),
            ("null_before", "Null before"), ("null_after", "Null after"),
            ("changed_cells", "Changed cells"), ("invalid_flagged", "Invalid flagged, retained"),
        ])
        if "cohort" in result:
            cohort = result["cohort"]
            lines += ["### Loan-level checks", ""]
            lines += table([{"measure": key, "value": value} for key, value in cohort["metrics"].items()],
                           [("measure", "Measure"), ("value", "Value")])
            lines += ["### Provisional 12-month outcome", ""]
            lines += table(cohort["outcomes"], [("outcome_reason", "Reason"), ("loans", "Loan IDs")])
            lines += ["### Provisional chronological split", ""]
            lines += table(cohort["splits"], [
                ("proposed_split", "Split"), ("all_candidates", "Candidates"),
                ("eligible_candidates", "Eligible under provisional rules"),
                ("labelled_but_not_split_eligible", "Labelled but withheld"),
            ])
    lines += [
        "## Research decisions still requiring review", "",
        "The cohort is a review table, not an approved final training dataset.", "",
        "- Candidate population: confirmed fixed-rate loans (FRM). Observation date: first valid monthly record; baseline must be current, have positive balance and not already be exited/defaulted.",
        "- Candidate positive: numeric delinquency >= 3, RA, or exit 02/03/09 during months 1-12 after observation.",
        "- Terminal events use their effective month, falling back to the reporting month when that date is missing.",
        "- Candidate negative: 12 distinct future reporting months, all statuses known, no observed default or early exit.",
        "- Prepayments/maturity and other early exits are censored, not automatically labelled non-default.",
        "- Duplicate loan-months within the outcome window, duplicate origination IDs and invalid reporting months leave the label null.",
        "- Known positives can be retained despite reporting gaps; gaps do not erase a documented event.",
        "- Internal-gap counts describe the interval between first/last observed months, not a full 30-year contractual history.",
        "- Unknown-code checks are conservative and limited to the configured register. Blank optional fields can be legitimate.",
        "- Development: observations before 2018; validation: 2018-2019; test: 2020-2023; later observations: reserve.",
        f"- Assumed label availability: later of horizon end/event reporting month plus {run['lag_months']} month(s). This is an assumption, not historical release-vintage proof.",
        "- Development/validation labels must be available before the next split begins; otherwise they are withheld.",
        "- Splits and outcomes are provisional. No final split files are frozen automatically.",
        "- Missing numeric feature values remain null; invalid numeric casts/ranges become null only in derived candidate features.",
        "- The borrower-count disclosure changed after 2018 Q1: earlier code 2 means more than one borrower, not exactly two. borrowers_code is retained as a code and requires consistent model encoding.",
        "- Monthly performance and quality/outcome columns are not predictive features. Approved origination input names:",
        ", ".join(alias for alias, _ in FEATURES.values()) + ".", "",
        "Checks are within each origination year. Cross-year ID uniqueness and duplicate/conflicting records require review before pooling years.", "",
        "Monthly histories begin at Freddie Mac acquisition, which can differ from origination. Do not describe the candidate model as an origination-time model without further checks.", "",
        f"Source definitions: {USER_GUIDE}", "",
    ]
    if run.get("error"):
        lines += ["## Error", "", display(run["error"]), "",
                  "Partial outputs may exist in this run folder. The run is not complete; original inputs were not modified.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
