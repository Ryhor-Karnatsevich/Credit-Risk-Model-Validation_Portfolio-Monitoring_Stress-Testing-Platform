"""One plain Markdown report with a short overview and focused detail."""

from uuid import uuid4


def display(value, percent=False, digits=3):
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        return value.replace("|", "/").replace("\n", " ")
    return f"{100*value:.{digits}f}%" if percent else f"{value:.{digits}f}"


def label(key):
    return key.replace("holdout_", "Holdout ").replace("oot_", "Later-period ")


def table(lines, headings, records):
    if not records:
        lines += ["No results available yet.", ""]
        return
    lines += ["| " + " | ".join(headings) + " |", "|" + "|".join("---" for _ in headings) + "|"]
    lines += ["| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |" for row in records]
    lines.append("")


def risk_rows(grouped):
    return [[row["segment"], f"{row['loans']:,}", f"{row['events']:,}", display(row["mean_pd"], True),
             display(row["observed_rate"], True), display(row["oe"]), display(row["excess_events"], digits=1)]
            for row in grouped]


def write_report(path, state):
    diagnostics = state.get("diagnostics", {})
    source = state.get("event_source", {})
    lines = ["# Logistic model validation and error diagnosis", "", f"Status: {state['status']}",
             f"Saved model: {state.get('model_experiment', '')}; preparation run: {state['prepared_run']}.", "",
             "The 90% logistic model and its preprocessing are frozen. Predictions are reproduced from saved weights; no model training, updating or recalibration is performed. Diagnostic calibration regressions below assess errors only.", "",
             "Evaluation uses the saved 10% holdouts from 2012-2017 and the already-inspected 2020 loans. No 2021-2023 starting observations are evaluated.", "",
             "## Main findings", ""]
    if diagnostics:
        worst_key, worst = max(diagnostics.items(), key=lambda item: item[1]["metrics"].get("observed_expected_ratio") or -1)
        m = worst["metrics"]
        lines += [f"- Largest aggregate observed/expected ratio among completed groups: {label(worst_key)}, "
                  f"{display(m['observed_expected_ratio'])}. Predicted {display(m['mean_predicted_pd'], True)}; "
                  f"observed {display(m['observed_event_rate'], True)}."]
        failures = sum(sum(value for name, value in item["date_checks"].items()
                           if name != "positives_with_any_history_gap") for item in diagnostics.values())
        lines += [f"- Saved-label/date contradiction flags: {failures:,}. Flags may overlap; this is not a count of distinct problem loans.",
                  "- Group and month differences locate the errors. They do not establish their economic or causal explanation."]
        incomplete = sum(not item["calibration"]["converged"] for item in diagnostics.values())
        if incomplete:
            lines += [f"- Calibration regressions undefined or not numerically converged: {incomplete}. Review their diagnostic estimates before interpreting them."]
    else:
        lines += ["Core diagnostics have not finished yet."]
    lines += [f"- Source event-history audit: {source.get('status', 'NOT STARTED')}. "
              f"Completed file checks: {len(source.get('files', {}))}/{source.get('planned_files', '?')}.", ""]
    if source.get("status") == "COMPLETE":
        source_flags = sum(value for name, value in source["checks"].items() if name != "positive_loans")
        lines += [f"Source-event contradiction/unmatched flags: {source_flags:,}; flags may overlap. Inspect the source-verification section before attributing these errors to the model.", ""]
    if state.get("error"):
        lines += ["Error: " + state["error"].replace("\n", " "), ""]
    lines += ["## Overall evaluation", "",
              "AUC measures ranking. Observed/expected (O/E) above 1 means underprediction. Intervals are approximate fixed-model, loan-level DeLong intervals; they do not capture shared economic shocks or uncertainty from model training.", ""]
    records = []
    for key, item in sorted(diagnostics.items()):
        m, ci = item["metrics"], item["auc_interval"]
        records.append([label(key), f"{m['loans']:,}", f"{m['bad_events']:,}", display(m["roc_auc"]),
                        f"{display(ci['low'])} to {display(ci['high'])}", display(m["mean_predicted_pd"], True),
                        display(m["observed_event_rate"], True), display(m["observed_expected_ratio"])])
    table(lines, ["Group", "Loans", "Bad events", "AUC", "Approx. 95% interval", "Predicted risk", "Observed risk", "O/E"], records)
    lines += ["## Calibration diagnostics", "",
              "Calibration-in-the-large (CITL) holds the model's risk spread fixed and estimates an overall log-odds adjustment; ideal is 0. Positive CITL indicates average underprediction. Joint calibration intercept/slope assess level and spread together; ideal is intercept 0, slope 1. These coefficients are not applied to predictions.", ""]
    table(lines, ["Group", "CITL", "Joint intercept", "Joint slope", "Numerical fit converged", "Log loss", "Brier score"],
          [[label(key), display(item["calibration"]["citl"]), display(item["calibration"]["intercept"]),
            display(item["calibration"]["slope"]), display(item["calibration"]["converged"]),
            display(item["metrics"]["log_loss"], digits=6), display(item["metrics"]["brier_score"], digits=6)]
           for key, item in sorted(diagnostics.items())])
    lines += ["## Sensitivity to starting-month composition", "",
              "The range below resamples entire starting-month cohorts 2,000 times with a fixed seed. It is a descriptive 95% month-resampling range for O/E, not a guarantee about future economic regimes. There are only about 12 month cohorts per year, their outcome windows overlap, and the models/holdouts have already been inspected.", ""]
    table(lines, ["Group", "Month cohorts", "O/E: 2.5th to 97.5th percentile"],
          [[label(key), item["month_resampling"]["months"],
            f"{display(item['month_resampling']['low'])} to {display(item['month_resampling']['high'])}"]
           for key, item in sorted(diagnostics.items())])

    headings = ["Group", "Loans", "Bad events", "Predicted risk", "Observed risk", "O/E", "Observed minus expected events"]
    for key in state["signature"]["focus"]:
        if key not in diagnostics:
            continue
        item = diagnostics[key]
        lines += [f"## {label(key)}: detailed checks", "",
                  "The following tables use only evaluation loans, never the 90% training rows. A monthly row means loans FIRST OBSERVED in that month and their following 12-month outcomes, not events occurring only during that calendar month.", "",
                  "### By starting month", ""]
        table(lines, headings, risk_rows([row for row in item["groups"] if row["dimension"] == "month"]))
        for dimension, title in (("fico", "Credit score"), ("ltv", "Loan-to-value"),
                                 ("dti", "Debt-to-income"), ("dti_missing", "Missing DTI")):
            lines += [f"### {title}", ""]
            table(lines, headings, risk_rows([row for row in item["groups"] if row["dimension"] == dimension]))
        lines += ["### States contributing most to the event shortfall", "",
                  "Sorted by observed minus expected events. Large states can contribute many excess events simply because they contain many loans. This is a contribution breakdown, not evidence that state caused the error.", ""]
        states = sorted([row for row in item["groups"] if row["dimension"] == "state"],
                        key=lambda row: row["excess_events"], reverse=True)[:10]
        table(lines, headings, risk_rows(states))
        lines += ["### Population differences from training", "",
                  "Total variation is a descriptive distance between binned feature distributions (0 means identical). PSI uses half an observation of smoothing per bin to handle zeros. No automatic approval thresholds are used, and drift does not itself prove the cause of miscalibration.", ""]
        table(lines, ["Feature", "PSI", "Total variation", "Missing: training", "Missing: evaluation"],
              [[row["dimension"], display(row["psi"]), display(row["total_variation"]),
                display(row["missing_training"], True), display(row["missing_evaluation"], True)] for row in item["drift"]])
        lines += ["### Saved-label and date checks", ""]
        table(lines, ["Check", "Flags"], [[name.replace("_", " "), f"{value:,}"] for name, value in item["date_checks"].items()])
        lines += ["The history-gap count concerns ANY internal gap in the retained lifetime history; it is not automatically a defect in this 12-month label. Other flags describe contradictions in saved fields, not an exhaustive source-data audit.", "",
                  "### When first bad events occurred", "",
                  "These are event MONTHS, separate from the starting months above. Counts come from the saved first-event dates. They are not monthly default rates, because no monthly at-risk denominator is being calculated.", ""]
        table(lines, ["Event month", "Loans with first bad event"],
              [[row["event_month"] or "Missing", f"{row['loans']:,}"] for row in item["event_calendar"]])
        if key in state.get("plots", {}):
            lines += ["### Calibration by predicted-risk group", "",
                      f"![Calibration: {label(key)}]({state['plots'][key]})", "",
                      "Points above the diagonal indicate underprediction. These are equal-sized risk-group point estimates, not a fitted recalibration function.", ""]

    lines += ["## Source event-history verification", "",
              f"Status: {source.get('status', 'NOT STARTED')}", "",
              "This reconstructs qualifying events from five cleaned performance columns for POSITIVE loans in the 2017 holdout and 2020 evaluation only. It verifies event evidence and timing, not every negative label or every source field. Zero-balance credit events use their effective month, falling back to reporting month exactly as in the saved target rules.", ""]
    if source.get("status") == "COMPLETE":
        table(lines, ["Check", "Loans"], [[name.replace("_", " "), f"{value:,}"] for name, value in source["checks"].items()])
        table(lines, ["Evaluation group", "Trigger(s) at saved first-event month", "Loans"],
              [[label(row["group_name"]), row["event_type"], f"{row['loans']:,}"] for row in source["types"]])
        lines += ["Trigger combinations are retained when several qualifying events occur in the same first-event month. The type rows are mutually exclusive and include unmatched cases. Source-check flags can overlap.", ""]
    else:
        lines += ["Source event types are not yet verified in this report. Complete the normal run without --skip-event-history; completed file checks will be reused.", ""]
    lines += ["## What remains a research decision", "",
              "- Validate any date/source contradictions before treating the probability gap as a model-only issue.",
              "- Inspect monthly patterns, group errors and input-distribution differences together. These are descriptive diagnostics, not causal estimates.",
              "- Small event counts make subgroup results noisy; no group is approved or rejected automatically.",
              "- Recalibration or new predictors must be fitted on separate development/calibration data and evaluated elsewhere. Do not fit an adjustment to 2020 outcomes and present the same-period match as independent validation.",
              "- 2021-2023 loan observations remain reserved. Source histories of older evaluated loans can contain later monthly records needed to check their outcomes.", "",
              "Supporting machine-readable statistics and source/cache fingerprints are retained in this run's manifest.json. Prediction and event caches are local generated data; original Parquets, split assignments and model weights are unchanged.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
