"""Validation diagnostics for the first logistic baseline; no threshold tuning."""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def evaluate(y, probabilities, training_rate):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if (y.size == 0 or p.shape != y.shape or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Invalid validation labels/probabilities")
    if not 0 < training_rate < 1:
        raise ValueError("The training benchmark needs both target classes")
    both_classes = np.unique(y).size == 2
    observed = int(y.sum())
    constant = np.full(y.size, training_rate, dtype=np.float64)
    brier = float(np.mean((p - y) ** 2))
    baseline_brier = float(np.mean((constant - y) ** 2))
    auc = float(roc_auc_score(y, p)) if both_classes else None
    metrics = {
        "loans": int(y.size), "bad_events": observed,
        "observed_event_rate": float(y.mean()), "mean_predicted_pd": float(p.mean()),
        "observed_expected_ratio": float(observed / p.sum()) if p.sum() > 0 else None,
        "roc_auc": auc, "gini": 2 * auc - 1 if auc is not None else None,
        "average_precision": float(average_precision_score(y, p)) if both_classes else None,
        "no_skill_average_precision": float(y.mean()) if both_classes else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "constant_benchmark_log_loss": float(log_loss(y, constant, labels=[0, 1])),
        "brier_score": brier, "constant_benchmark_brier": baseline_brier,
        "brier_skill_vs_training_rate": 1 - brier / baseline_brier if baseline_brier > 0 else None,
    }
    order = np.argsort(-p, kind="stable")
    bands = []
    for rank, indexes in enumerate(np.array_split(order, min(10, y.size)), start=1):
        predicted = float(p[indexes].sum())
        events = int(y[indexes].sum())
        bands.append({"risk_band": rank, "loans": int(indexes.size), "bad_events": events,
                      "mean_pd": float(p[indexes].mean()), "observed_rate": float(y[indexes].mean()),
                      "minimum_pd": float(p[indexes].min()), "maximum_pd": float(p[indexes].max()),
                      "observed_expected_ratio": events / predicted if predicted > 0 else None})
    for fraction, name in ((0.01, "top_1pct_event_capture"), (0.05, "top_5pct_event_capture"),
                           (0.10, "top_10pct_event_capture")):
        size = max(1, int(np.ceil(y.size * fraction)))
        metrics[name] = float(y[order[:size]].sum() / observed) if observed else None
    return metrics, bands


def number(value, percent=False):
    if value is None:
        return "Not defined (insufficient events/classes)"
    return f"{100 * value:.3f}%" if percent else f"{value:.6g}"


def write_summary(path, state, detail_report):
    """Stable, short user-facing entry point; technical run reports stay intact."""
    training = state["training"]
    validation = state["validation"]
    metrics = validation["metrics"]
    lines = ["# Current project results", "", "## First logistic-regression model", "",
             f"Status: {state['status']}", "",
             "Question: will a loan experience a qualifying bad event within the next 12 months, before full repayment?", "",
             "| Important result | Value |", "|---|---:|",
             f"| Training loans | {training['rows']:,} |",
             f"| Validation loans | {metrics['loans']:,} |",
             f"| Validation bad events | {metrics['bad_events']:,} |",
             f"| Model converged | {'Yes' if training['converged'] else 'No'} |",
             f"| Model fitting time | {training['fit_seconds']:.1f} seconds |",
             f"| ROC AUC (risk ranking; random = 0.5) | {number(metrics['roc_auc'])} |",
             f"| Average predicted bad-event probability | {number(metrics['mean_predicted_pd'], True)} |",
             f"| Actual bad-event rate | {number(metrics['observed_event_rate'], True)} |",
             f"| Bad events captured in the highest-risk 10% | {number(metrics['top_10pct_event_capture'], True)} |", "",
             "## What this means", ""]
    if not training["converged"]:
        lines += ["- The optimizer did not converge. Do not treat these results as a finalized baseline."]
    auc = metrics["roc_auc"]
    if auc is None:
        lines += ["- Ranking performance cannot be assessed because validation lacks both target classes."]
    elif auc > 0.5:
        lines += ["- The model ranked bad-event loans above other loans better than random in this validation sample. This alone does not establish that the probabilities are accurate."]
    else:
        lines += ["- Risk ranking did not exceed the random-ranking reference in this validation sample."]
    observed, predicted = metrics["observed_event_rate"], metrics["mean_predicted_pd"]
    if observed > predicted:
        lines += ["- Predicted probabilities were too low overall: more bad events occurred than the model expected. Probability calibration needs review."]
    elif observed < predicted:
        lines += ["- Predicted probabilities were too high overall: fewer bad events occurred than the model expected. Probability calibration needs review."]
    else:
        lines += ["- Average predicted and observed rates agree; agreement within risk groups still needs review."]
    if validation.get("bands"):
        highest = validation["bands"][0]
        lines += [f"- In the highest-risk group, predicted risk was {number(highest['mean_pd'], True)}, "
                  f"versus {number(highest['observed_rate'], True)} observed."]
    lines += ["- The final test remains unevaluated. These are validation results, not a model-approval or profitability decision.", "",
              f"Training observations: {training['first_observation']} to {training['last_observation']}.",
              f"Validation observations: {validation['first_observation']} to {validation['last_observation']}.", "",
              "## Next step", "",
              "Review probability calibration and the high-risk groups before changing the model or opening the final test.", "",
              "## Optional detail", "",
              f"[Full model report]({detail_report.relative_to(path.parent).as_posix()})", "",
              "This is the main report to open. Run-specific folders contain supporting diagnostics and checkpoints; you do not need to browse them for the main findings.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path, state):
    lines = ["# First logistic-regression model", "", f"Status: {state['status']}",
             f"Preparation run: {state['prepared_run']}", "",
             "Exploratory baseline, not a model-approval or regulatory validation decision.",
             "Target: a qualifying serious-delinquency/credit-loss event within 12 months. Verified full repayment without an earlier bad event is 0; a later payment does not erase an earlier event.", "",
             "## Design", "",
             "- Development rows only are used for fitting. All eligible development loans are used; no sampling or class balancing.",
             "- L2-regularized logistic regression, fitted with L-BFGS on a sparse matrix.",
             "- Only manifest feature_names enter the model. IDs, future outcomes and repayment markers are not predictors.",
             "- Imputation, encoding and scaling reuse the development-only preprocessing settings.",
             "- Validation is evaluated without refitting, tuning, threshold selection or recalibration.",
             "- The final test split is not scored or evaluated by this script.",
             "- These are raw model probabilities, not calibrated/approved production PDs.", ""]
    if "training" in state:
        training = state["training"]
        lines += ["## Training", "", "| Measure | Value |", "|---|---:|",
                  f"| Loans | {training['rows']:,} |",
                  f"| Bad events | {training['events']:,} |",
                  f"| Observed bad-event rate | {number(training['event_rate'], True)} |",
                  f"| First observation | {training['first_observation']} |",
                  f"| Last observation | {training['last_observation']} |",
                  f"| Encoded features | {training['features']} |",
                  f"| Training load seconds | {training['load_seconds']:.1f} |",
                  f"| Model fit seconds | {training['fit_seconds']:.1f} |",
                  f"| Optimizer iterations | {training['iterations']} |",
                  f"| Converged | {training['converged']} |", ""]
        if not training["converged"]:
            lines += ["WARNING: the optimizer did not converge. Treat coefficients and metrics as provisional; do not conclude that this is the final logistic baseline.", ""]
    if "validation" in state:
        m = state["validation"]["metrics"]
        lines += ["## Validation results", "",
                  f"Loans: {m['loans']:,}. Observed bad events: {m['bad_events']:,}.",
                  f"Observation dates: {state['validation']['first_observation']} to {state['validation']['last_observation']}.", "",
                  "| Measure | Model | Comparison / meaning |", "|---|---:|---|",
                  f"| ROC AUC | {number(m['roc_auc'])} | 0.5 is random ranking; higher is better |",
                  f"| Gini | {number(m['gini'])} | 2 x AUC - 1; higher is better |",
                  f"| Average precision (AP) | {number(m['average_precision'])} | No-skill reference: {number(m['no_skill_average_precision'])}; higher is better |",
                  f"| Log loss | {number(m['log_loss'])} | Constant training-rate benchmark: {number(m['constant_benchmark_log_loss'])}; lower is better |",
                  f"| Brier score | {number(m['brier_score'])} | Constant training-rate benchmark: {number(m['constant_benchmark_brier'])}; lower is better |",
                  f"| Brier skill | {number(m['brier_skill_vs_training_rate'])} | Positive means better than the constant benchmark |",
                  f"| Mean predicted probability | {number(m['mean_predicted_pd'], True)} | Compare with the observed event rate below |",
                  f"| Observed event rate | {number(m['observed_event_rate'], True)} | Actual validation outcomes |",
                  f"| Observed / expected events | {number(m['observed_expected_ratio'])} | 1 is aggregate agreement; above 1 means underprediction |",
                  f"| Events captured in riskiest 10% | {number(m['top_10pct_event_capture'], True)} | Random ranking captures approximately 10% |", "",
                  "The benchmark assigns every validation loan the development bad-event rate. It does not learn the validation event rate.", "",
                  "AP is average precision, not trapezoidal PR-AUC. A high ROC AUC does not by itself establish accurate probabilities or financial profitability.", "",
                  "## Risk bands: validation only", "",
                  "Band 1 contains the highest predicted risks. Bands are approximately equal-sized; tied scores can span bands.", "",
                  "| Band | Loans | Bad events | Mean predicted PD | Observed event rate | Observed / expected |",
                  "|---|---:|---:|---:|---:|---:|"]
        for band in state["validation"]["bands"]:
            lines.append(f"| {band['risk_band']} | {band['loans']:,} | {band['bad_events']:,} | "
                         f"{number(band['mean_pd'], True)} | {number(band['observed_rate'], True)} | "
                         f"{number(band['observed_expected_ratio'])} |")
        lines += ["", "## Validation by observation year", "",
                  "| Year | Loans | Bad events | ROC AUC | Mean PD | Observed rate |",
                  "|---|---:|---:|---:|---:|---:|"]
        for year, metrics in state["validation"]["by_year"].items():
            lines.append(f"| {year} | {metrics['loans']:,} | {metrics['bad_events']:,} | "
                         f"{number(metrics['roc_auc'])} | {number(metrics['mean_predicted_pd'], True)} | "
                         f"{number(metrics['observed_event_rate'], True)} |")
        lines += ["", "## How to evaluate this first result", "",
                  "1. Check convergence before trusting the fitted baseline.",
                  "2. Check ranking (AUC/AP) and improvement over the constant benchmark.",
                  "3. Compare predicted and observed event rates, overall and within the risk bands.",
                  "4. Check performance by observation year; do not equate origination year with prediction date.",
                  "5. Do not declare success from one metric. No universal acceptance threshold is imposed here.", ""]
    if state.get("warnings"):
        lines += ["## Warnings", ""] + ["- " + item.replace("\n", " ") for item in state["warnings"]] + [""]
    if state.get("error"):
        lines += ["## Error", "", state["error"].replace("\n", " "), ""]
    lines += ["## Output files", "",
              "- metrics.json: configuration, data fingerprints, training information and validation metrics.",
              "- validation_predictions.parquet: individual probabilities and observed validation outcomes.",
              "- risk_bands.csv and coefficients.csv: readable diagnostics.",
              "- Model weights and the fitted input-preparation settings are saved under data/models/logistic_regression/.", "",
              "Coefficients describe fitted associations, not causal effects. Numeric coefficients refer to standardized inputs; full one-hot categories are regularized jointly.",
              "Target, selection and temporal-design limitations still apply. No final-test performance, lending-profit estimate or approval decision is produced.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
