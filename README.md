# Credit Risk Model Validation, Portfolio Monitoring & Stress Testing Platform

A mortgage credit-risk research project using Python, SQL and Freddie Mac loan-level data. It estimates the probability of serious delinquency or a credit-loss event within the next 12 months, then aims to extend those estimates into portfolio loss, monitoring and stress testing.

The data pipeline and first logistic-regression baseline are complete. Model validation is in progress.

## Key decisions

- **Data:** Freddie Mac Standard US fixed-rate residential mortgages. Data preparation covers 2012-2025; the first model uses eligible loans from the 2012-2023 vintages.
- **Prediction:** one risk probability per loan at its first available monthly performance observation. Predictors are starting mortgage characteristics, including credit score, DTI, LTV/CLTV, balance, rate, term and property/loan categories.
- **Outcome:** `1` for 90+ days delinquent, REO status or a qualifying terminal credit event within 12 months; `0` for an observed event-free window or verified full repayment without an earlier bad event. Later repayment does not erase an earlier bad event.
- **Incomplete histories:** unknown outcomes are excluded from model fitting. Missing predictor values are handled using settings learned from development loans only.
- **Time separation:** older observations train the model; later observations evaluate it. Training outcomes must be available before the evaluation period. The 2020-2023 final test remains unevaluated.
- **Baseline:** L2-regularized logistic regression using starting characteristics. Later payment behaviour is reserved for outcomes and potential future extensions.
- **Storage:** local Parquet files retain the source rows and columns. Basic conversion checks passed; storage fell from 148.16 GiB of TXT to 16.67 GiB of Parquet (88.75% less). These checks do not compare every cell. Raw data and generated outputs are excluded from Git.

## Latest logistic-regression evaluation

Saved run completed on **29 October 2026**. Training observations: **January 2012-November 2016**. Validation observations: **January-November 2018**.

| Measure | Result |
|---|---:|
| Training loans | 7,770,079 |
| Validation loans | 1,207,643 |
| Validation bad events | 2,546 |
| ROC AUC (random ranking = 0.5) | 0.800 |
| Average predicted risk | 0.171% |
| Observed bad-event rate | 0.211% |
| Bad events captured in the highest-risk 10% | 45.954% |
| Highest-risk 10%: predicted / observed risk | 0.605% / 0.969% |
| Optimizer converged | Yes |

The model distinguishes higher-risk loans, but underestimates risk overall and in the highest-risk group. These results cover one validation period; stability across years and probability calibration still need review.

These figures come from the saved code-generated evaluation. The full evaluation is preserved in the [research paper](research_paper.md#latest-saved-logistic-regression-evaluation). After a new run, the local [main results report](results/report.md) contains the updated summary; this README is a dated snapshot.

## Research documentation

[Research paper](research_paper.md): project rationale, data decisions, pipeline details, stage tables and the full saved baseline evaluation.

Next: check how many loans are included or excluded each year and why, then assess performance across time and risk groups before changing the model. Gradient boosting, portfolio expected loss, monitoring and stress testing follow later.
