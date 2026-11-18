"""Numerical diagnostics and fixed, outcome-independent segment definitions."""

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit


SEGMENTS = {
    "month": "strftime(observation_month, '%Y-%m')",
    "fico": "CASE WHEN fico IS NULL THEN 'Missing' WHEN fico < 640 THEN '<640' WHEN fico < 680 THEN '640-679' WHEN fico < 720 THEN '680-719' WHEN fico < 760 THEN '720-759' ELSE '760+' END",
    "ltv": "CASE WHEN ltv IS NULL THEN 'Missing' WHEN ltv <= 60 THEN '<=60' WHEN ltv <= 80 THEN '(60,80]' WHEN ltv <= 90 THEN '(80,90]' ELSE '>90' END",
    "dti": "CASE WHEN dti IS NULL THEN 'Missing' WHEN dti <= 30 THEN '<=30' WHEN dti <= 40 THEN '(30,40]' WHEN dti <= 50 THEN '(40,50]' ELSE '>50' END",
    "dti_missing": "CASE WHEN dti IS NULL THEN 'Missing' ELSE 'Observed' END",
    "state": "coalesce(property_state, 'Missing')",
    "purpose": "coalesce(purpose, 'Missing')",
    "occupancy": "coalesce(occupancy, 'Missing')",
    "channel": "coalesce(channel, 'Missing')",
}


def summary_groups(connection, view, outcomes=True):
    """One Parquet scan; grouping sets aggregate a fixed list of segments."""
    dimensions = list(SEGMENTS) if outcomes else [name for name in SEGMENTS if name != "month"]
    definitions = [f"{SEGMENTS[name]} AS d_{name}" for name in dimensions]
    choose = "CASE " + " ".join(f"WHEN grouping(d_{name}) = 0 THEN '{name}'" for name in dimensions) + " END"
    segment = "CASE " + " ".join(f"WHEN grouping(d_{name}) = 0 THEN d_{name}" for name in dimensions) + " END"
    extra = ""
    if outcomes:
        eps = float(np.finfo(np.float64).eps)
        clipped = f"greatest({eps!r}, least({1 - eps!r}, predicted_pd))"
        definitions += ["candidate_default_12m AS y", "predicted_pd AS p",
                        f"-candidate_default_12m * ln({clipped}) - (1-candidate_default_12m) * ln(1-{clipped}) AS loss"]
        extra = ", sum(y) AS events, sum(p) AS expected_events, avg(loss) AS log_loss, avg((p-y)*(p-y)) AS brier"
    query = ("WITH b AS (SELECT " + ", ".join(definitions) + f' FROM "{view}") '
             + f"SELECT {choose} AS dimension, {segment} AS segment, count(*) AS loans{extra} "
             + "FROM b GROUP BY GROUPING SETS (" + ", ".join(f"(d_{name})" for name in dimensions) + ")")
    result = connection.execute(query)
    names = [column[0] for column in result.description]
    rows = [dict(zip(names, row)) for row in result.fetchall()]
    for row in rows:
        if outcomes:
            row["events"] = int(row["events"])
            row["mean_pd"] = row["expected_events"] / row["loans"]
            row["observed_rate"] = row["events"] / row["loans"]
            row["oe"] = row["events"] / row["expected_events"] if row["expected_events"] else None
            row["excess_events"] = row["events"] - row["expected_events"]
    return sorted(rows, key=lambda row: (row["dimension"], row["segment"]))


def drift(training, evaluation):
    """Coarse distribution diagnostics; no universal PSI acceptance cutoffs."""
    result = []
    for dimension in SEGMENTS:
        if dimension == "month":
            continue
        before = {row["segment"]: row["loans"] for row in training if row["dimension"] == dimension}
        after = {row["segment"]: row["loans"] for row in evaluation if row["dimension"] == dimension}
        bins = sorted(set(before) | set(after))
        n, m = sum(before.values()), sum(after.values())
        if not n or not m:
            continue
        a = np.asarray([before.get(key, 0) for key in bins], dtype=float)
        b = np.asarray([after.get(key, 0) for key in bins], dtype=float)
        # Half a pseudo-observation per bin avoids log(0); disclose smoothing.
        p, q = (a + 0.5) / (n + 0.5 * len(bins)), (b + 0.5) / (m + 0.5 * len(bins))
        result.append({"dimension": dimension, "psi": float(np.sum((q-p) * np.log(q/p))),
                       "total_variation": float(0.5 * np.abs(a/n-b/m).sum()),
                       "missing_training": before.get("Missing", 0) / n,
                       "missing_evaluation": after.get("Missing", 0) / m})
    return result


def auc_interval(y, p):
    """DeLong placement-value variance for a fixed model, including tied scores."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    positive, negative = np.sort(p[y == 1]), np.sort(p[y == 0])
    if len(positive) < 2 or len(negative) < 2:
        return {"low": None, "high": None, "reason": "Too few events/non-events"}
    v = (np.searchsorted(negative, positive, side="left")
         + np.searchsorted(negative, positive, side="right")) / (2 * len(negative))
    w = 1 - (np.searchsorted(positive, negative, side="left")
             + np.searchsorted(positive, negative, side="right")) / (2 * len(positive))
    auc = float(v.mean())
    se = float(np.sqrt(v.var(ddof=1)/len(positive) + w.var(ddof=1)/len(negative)))
    return {"low": max(0.0, auc-1.96*se), "high": min(1.0, auc+1.96*se), "standard_error": se,
            "reason": "Zero placement variance; interval degenerates" if se == 0 else None}


def calibration_diagnostics(y, p):
    """Diagnostic regressions only; never replace the model's predictions.

    CITL solves an intercept-only logistic regression with logit(p) as offset.
    Joint intercept/slope solve logit(P(Y=1)) = a + b*logit(p).
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Calibration diagnostics require binary labels and aligned probabilities")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid predicted probabilities")
    if len(y) == 0 or y.sum() in (0, len(y)):
        return {"citl": None, "intercept": None, "slope": None, "converged": False,
                "note": "Both outcome classes are required"}
    z = logit(np.clip(p, 1e-12, 1-1e-12))
    citl = float(brentq(lambda a: float(expit(z+a).mean()-y.mean()), -100, 100))
    center, scale = float(z.mean()), float(z.std())
    if scale < 1e-10:
        return {"citl": citl, "intercept": None, "slope": None, "converged": False,
                "note": "Constant predictions; slope is not identifiable"}
    x = (z-center)/scale

    def objective(parameters):
        linear = parameters[0] + parameters[1]*x
        residual = expit(linear)-y
        loss = np.mean(np.logaddexp(0, linear)-y*linear)
        return float(loss), np.asarray([residual.mean(), np.mean(residual*x)])

    fitted = minimize(objective, np.asarray([center, scale]), jac=True, method="BFGS",
                      options={"gtol": 1e-8, "maxiter": 100})
    if not np.isfinite(fitted.x).all():
        return {"citl": citl, "intercept": None, "slope": None, "converged": False,
                "note": "Non-finite calibration fit"}
    slope = float(fitted.x[1]/scale)
    return {"citl": citl, "intercept": float(fitted.x[0]-slope*center), "slope": slope,
            "converged": bool(fitted.success), "note": str(fitted.message)}


def month_resampling(months, repetitions=2000, seed=42):
    """Descriptive O/E range from resampling entire starting-month cohorts."""
    rows = [row for row in months if row["dimension"] == "month"]
    if len(rows) < 4:
        return {"low": None, "high": None, "months": len(rows), "repetitions": repetitions}
    events = np.asarray([row["events"] for row in rows], dtype=float)
    expected = np.asarray([row["expected_events"] for row in rows], dtype=float)
    indexes = np.random.default_rng(seed).integers(0, len(rows), size=(repetitions, len(rows)))
    denominator = expected[indexes].sum(axis=1)
    ratios = events[indexes].sum(axis=1)[denominator > 0] / denominator[denominator > 0]
    if not len(ratios):
        return {"low": None, "high": None, "months": len(rows), "repetitions": repetitions}
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "months": len(rows), "repetitions": repetitions}


def date_audit(connection):
    """Check saved labels/dates on evaluated loans, without rebuilding targets."""
    query = """
        SELECT
            count(*) FILTER (WHERE candidate_default_12m=1 AND
                (event_in_window IS NULL OR event_in_window<=observation_month OR event_in_window>horizon_end)) AS positive_event_date_errors,
            count(*) FILTER (WHERE candidate_default_12m=1 AND event_report_month IS NULL) AS positive_without_reporting_date,
            count(*) FILTER (WHERE candidate_default_12m=1 AND first_exit_month<event_in_window) AS positive_event_after_recorded_exit,
            count(*) FILTER (WHERE candidate_default_12m=0 AND event_in_window IS NOT NULL) AS zero_with_recorded_bad_event,
            count(*) FILTER (WHERE candidate_default_12m=0 AND NOT coalesce(reclassified_as_repaid,FALSE)
                AND (future_months!=12 OR future_unknown_months!=0)) AS zero_without_full_known_window,
            count(*) FILTER (WHERE reclassified_as_repaid IS TRUE AND
                (candidate_default_12m!=0 OR payoff_confirmation_month IS NULL OR first_exit_month IS NULL
                 OR first_exit_month<=observation_month OR first_exit_month>horizon_end
                 OR payoff_confirmation_month<first_exit_month)) AS repayment_evidence_date_errors,
            count(*) FILTER (WHERE assumed_label_available_month IS NULL
                OR assumed_label_available_month<horizon_end
                OR (event_report_month IS NOT NULL AND assumed_label_available_month<event_report_month)) AS label_availability_errors,
            count(*) FILTER (WHERE candidate_default_12m=1 AND internal_gap_months>0) AS positives_with_any_history_gap
        FROM evaluation
    """
    result = connection.execute(query)
    names = [column[0] for column in result.description]
    return dict(zip(names, result.fetchall()[0]))


def plot_calibration(path, bands, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([row["mean_pd"] for row in bands]) * 100
    y = np.asarray([row["observed_rate"] for row in bands]) * 100
    limit = max(float(x.max()), float(y.max()), 0.001)*1.08
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.plot([0, limit], [0, limit], color="gray", linestyle="--", label="Perfect agreement")
    order = np.argsort(x)
    axis.plot(x[order], y[order], "o-", label="Equal-sized risk groups")
    axis.set(xlabel="Mean predicted 12-month risk (%)", ylabel="Observed 12-month event rate (%)",
             title=title, xlim=(0, limit), ylim=(0, limit))
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
