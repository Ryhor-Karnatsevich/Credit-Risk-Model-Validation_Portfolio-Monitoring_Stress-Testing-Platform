# Credit Risk Model Validation, Portfolio Monitoring & Stress Testing Platform

Working research paper. The planning and progress notes below are preserved from the original README. Some describe earlier stages; see the [current research checkpoint](#current-research-checkpoint) and [latest saved evaluation](#latest-saved-logistic-regression-evaluation) at the end for verified progress.

An end-to-end credit-risk project covering PD model development, independent validation, portfolio monitoring, expected-loss analysis, and stress testing.

## Agreed mortgage scope and data acquisition

Decision recorded: 7 October 2026. The project will study US residential fixed-rate mortgages using Freddie Mac's Standard Single-Family Loan-Level Dataset.

Main question: What is the probability that an eligible mortgage becomes seriously delinquent or reaches a credit-loss event within the next 12 months?

The observation date is provisionally the first monthly performance record. We must check loan age and initial delinquency and define eligibility before freezing the population. Performance starts at Freddie Mac acquisition; it does not necessarily start at origination. The proposed 90+ day delinquency and terminal-event definition still needs code-level confirmation against the official dictionary.

### What already exists

- Full annual archives and extracted origination/performance files for 2012-2026 are stored locally, alongside four FRED series.
- Part 1 is implemented in `src/data_extracting/`. Its pipeline prepares all 112 required files for 2012-2025 using Parquet -> TXT -> local ZIP fallback, then runs basic checks. Downloads remain manual. The latest pipeline report passed all 112 required files; existing Parquets did not require TXT comparisons to be repeated.
- The retained TXT-Parquet comparison report is `results/data_extracting/parquet_comparison.md`. It records 114 basic passes across the downloaded 2012-2026 files. The 2026 Q1 files are retained but excluded from the current pipeline scope.
- Obsolete format examples, layout downloads and the old raw-data `views` folder have been removed. Generated outputs belong in `results/`.
- Data preparation and basic conversion checks are complete. Loan-level data-quality checks and modelling feasibility are not yet complete.
- There is no historical loan downloader, joined modelling table, target construction, temporal split, trained model, loss calculation, monitoring system, or stress engine.

### Required data and proposed period

| Data | Proposed coverage | Required information |
|---|---|---|
| Standard mortgage origination | 2012-2023 cohorts | Loan ID, first payment date, maturity, credit score, DTI, LTV/CLTV, original balance, rate, term, purpose, occupancy, property type/state |
| Corresponding monthly performance | All available months for those cohorts in one frozen release | Reporting month, balance, delinquency, loan age, modifications, termination reason/date, reporting gaps |
| Loss and recovery fields | Retain the full available histories, including later resolutions | Actual loss, removal balance, recoveries, sales proceeds, expenses, accrued interest |
| FRED UNRATE, USSTHPI, MORTGAGE30US, FEDFUNDS | At least 2011 through the loan-performance cutoff | Dates, values, frequency, source and release/vintage information |
| Official documentation | Same release as the loan files | Layout, dictionary, release notes, usage terms |

Provisional cohort split: development 2012-2017, validation/calibration 2018-2019, final out-of-time test 2020-2023. These dates are a proposal, not a frozen research design. Ensure training and calibration outcomes would be available before the next evaluation begins; apply boundary gaps or move dates as needed for the 12-month horizon and reporting lag.

Require a complete observable outcome window or a documented terminal outcome. Missing reports and unexplained disappearance do not mean non-default. Recoveries can take longer than 12 months, so do not truncate all histories at the PD horizon. Mortgage prepayments need an explicit target treatment.

Historical macro values may be revised. Use release dates/vintages or defensible lags when treating macro variables as prediction inputs. Stress scenarios can use explicitly stated assumptions.

### Access and completeness

Freddie Mac provides free access for non-commercial research subject to registration and applicable terms. The available archive covers the proposed cohort years, but it represents Freddie Mac's disclosed population, not all US mortgages. Raw data must remain local and excluded from Git.

Neither error-free source data nor uninterrupted downloads can be guaranteed. Verify archive integrity, schema, loan-ID joins, monthly coverage, duplicates, missing/sentinel values, and default counts. Freeze the release and record source URLs, download dates and file hashes.

Official sources:
- https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
- https://www.freddiemac.com/fmac-resources/research/pdf/faq.pdf
- https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf

### Measured dataset size

The downloaded full Standard files for 2012-2026 contain 114 origination/performance files. Their extracted TXT size is **148.16 GiB**; the Parquet copies occupy **16.67 GiB**, an **88.75% reduction** (approximately **8.89 times smaller**).

The conversion uses lossless Zstandard compression and preserves all source columns as strings, including identifiers, leading zeros, empty fields and missing-value codes. No rows or columns are intentionally filtered. All 114 file pairs passed row-count, column/schema and first-100-row value comparisons. These are basic conversion checks, not an exhaustive comparison of every value.

ZIP and TXT originals are still retained. The storage figures compare representations; they do not mean that the original copies have been deleted or that analysis runs 8.89 times faster. Analysis speed has not been benchmarked.

Next step: check loan-level data quality using Parquet and review the findings before modelling.


## High-level project stages

| Index | Name | Current situation |
|---:|---|---|
| 0 | Scope and data feasibility gate | In progress |
| 1 | Cohort construction and data-quality system | |
| 2 | Exploratory credit-risk analysis | |
| 3 | Interpretable logistic-regression PD baseline | |
| 4 | Machine-learning challenger development | |
| 5 | Independent model validation | |
| 6 | Portfolio expected-loss layer | |
| 7 | Portfolio and model monitoring framework | |
| 8 | Scenario and stress testing | |
| 9 | Automated validation and management reporting | |
| 10 | Engineering, testing, and audit trail | |
| 11 | Optional correlated-default and economic-capital extension | |

## IDEA

The project follows this simple pipeline:

1. Find a public loan dataset with clear dates and loan outcomes.
2. Define exactly what counts as default and which loans can be studied fairly.
3. Clean the data and remove information that would reveal the future.
4. Split the loans by time: older loans for model development and later loans for the final test.
5. Build a simple logistic-regression PD model as the baseline.
6. Build a machine-learning model and compare it with the baseline.
7. Check whether both models rank risk correctly, predict realistic probabilities, and remain stable over time.
8. Approve, recalibrate, or reject the models based on the validation results.
9. Convert the selected PD estimates into expected portfolio loss using LGD and EAD.
10. Monitor changes in the model and portfolio, then test adverse scenarios.
11. Produce a final report explaining the results, risks, limitations, and required actions.

## What the project will model

The unit of analysis is one Freddie Mac fixed-rate mortgage observed when it first enters the monthly performance data.

The main model will estimate:

> The probability that the mortgage becomes seriously delinquent or reaches a credit-loss event within the next 12 months.

The initial default definition will be triggered by either:

- delinquency of 90 days or more; or
- a terminal credit event such as a third-party sale, short sale, charge-off, or REO disposition.

Only information available at origination or at the first observation date will be used as model input. Later payment behaviour, modifications, recoveries, expenses, and loan outcomes will be reserved for target construction, validation, and loss analysis. This separation prevents future information from leaking into the model.

The logistic-regression model will be the interpretable baseline. A gradient-boosting model will be the challenger. Both will be tested on later loan cohorts that were not used for model development or calibration.

## Data needed

| Data group | Main fields | Use in the project |
|---|---|---|
| Mortgage characteristics | Credit score, DTI, LTV/CLTV, original balance, interest rate, term, loan purpose, occupancy, property type and state | PD model inputs and segment analysis |
| Monthly loan performance | Reporting month, current balance, delinquency status, modification flag, termination code and termination date | Target construction, monitoring and cohort outcomes |
| Loss and recovery data | Sales proceeds, mortgage-insurance recoveries, other recoveries, expenses, deferred balance and actual loss | LGD, EAD and expected-loss analysis |
| Macroeconomic data | Unemployment, house-price index, mortgage rate and federal funds rate | Economic context, monitoring and stress scenarios |

## Selected data sources

| Source | Purpose | Current situation |
|---|---|---|
| Freddie Mac Single-Family Loan-Level Dataset | Primary origination, performance and actual-loss data | Full annual archives for 2012-2026 extracted locally; data-quality review pending |
| FRED `UNRATE` | US unemployment rate | Extracted |
| FRED `USSTHPI` | US house-price index | Extracted |
| FRED `MORTGAGE30US` | 30-year fixed mortgage rate | Extracted |
| FRED `FEDFUNDS` | Effective federal funds rate | Extracted |

The modelling population is restricted to mortgages purchased or guaranteed by Freddie Mac. Results will not be presented as representative of rejected applicants, unsecured consumer loans, or the full US mortgage market.

## Research stages

| Index | Name | Current situation |
|---:|---|---|
| 0 | Define research objective and model use | Complete |
| 1 | Identify candidate public datasets | Complete |
| 2 | Compare datasets and complete the feasibility gate | In progress |
| 3 | Define the model population | Complete |
| 4 | Define default and the performance horizon | In progress |
| 5 | Review field availability and create the leakage register | |
| 6 | Construct mature loan cohorts | |
| 7 | Run data-quality checks | |
| 8 | Create the chronological development, validation, and out-of-time samples | |
| 9 | Analyse portfolio composition and default behaviour | |
| 10 | Analyse missingness, distributions, and temporal drift | |
| 11 | Freeze the modelling sample and eligible features | |
| 12 | Build the preprocessing pipeline | |
| 13 | Develop the logistic-regression PD baseline | |
| 14 | Validate and interpret the logistic baseline | |
| 15 | Develop the machine-learning challenger | |
| 16 | Analyse challenger feature contributions and model risk | |
| 17 | Calibrate baseline and challenger PD estimates | |
| 18 | Complete the final out-of-time model comparison | |
| 19 | Perform independent discrimination, calibration, stability, and robustness validation | |
| 20 | Issue the approve, condition, recalibrate, or reject decision | |
| 21 | Define LGD and EAD estimates or assumptions | |
| 22 | Calculate and decompose portfolio expected loss | |
| 23 | Build the model and portfolio monitoring framework | |
| 24 | Define and execute stress scenarios | |
| 25 | Automate the validation and management report | |
| 26 | Add tests, run metadata, and the audit trail | |
| 27 | Document final findings, limitations, and conclusions | |
| 28 | Optional correlated-default and economic-capital extension | |

## [1] Data extracting

Status recorded: 27 October 2026. Initial data-preparation pipeline complete.

- Manual download -> local ZIP/TXT fallback -> Parquet conversion -> basic checks.
- Current pipeline scope: 2012-2025, all four quarters, origination and performance; **112/112 passed**.
- Full downloaded conversion: 2012-2026 Q1; **114/114 file pairs passed basic TXT-Parquet comparisons**.
- **148.16 GiB TXT -> 16.67 GiB Parquet: 88.75% less storage using lossless compression.**
- Retained results: [Data comparison](results/data_extracting/parquet_comparison.md).

Part 1 code is in `src/data_extracting/`; its reports are saved in `results/data_extracting/`.

## [2] Data preparation

Run `src/data_preparation/pipeline.py` normally to prepare all full years from 2012 through 2025. Completed files are automatically checked and reused. A smaller selection can be specified with `--years`.

- Reads Stage 1 Parquets without modifying them; writes separate cleaned Parquets under `data/prepared/<run_id>/<year>/`.
- Trims surrounding spaces and converts blanks/configured missing codes to null. Source row counts and columns are preserved; unusual values and duplicates are flagged rather than silently removed.
- Checks loan-ID joins, duplicate loan-months, internal monthly gaps, missing fields and configured numeric/date ranges.
- Creates `cohort_candidates.parquet` with provisional 12-month outcomes, exclusion reasons, origination features and chronological split flags. This is a review table, not a frozen training dataset.
- Writes a plain Markdown change report and source/configuration manifest to `results/data_preparation/<run_id>/`.
- No imputation, scaling, final feature selection or model fitting. The report explains the provisional baseline, censoring, default-event and time-boundary rules that still require review.

Requires `duckdb==2.0.0` from `requirements.txt`. Defaults to two engine threads, a 4 GB DuckDB memory budget and disk spilling; this is not a hard cap on total Python/OS memory. Stage 1 data is never overwritten.

Cleaning now counts changes and writes cleaned output in one source pass using bounded Arrow batches. History flags share a single loan-month grouping instead of separately parsing the performance data twice. Exact speed gains have not been benchmarked.

Normal runs automatically reuse saved progress for the same selected years, or extend an earlier run when all its years remain selected. Completed cleaned files are checked and skipped; an interrupted file restarts. The manifest is replaced atomically after each completed file. Reuse checks source sizes, modification times and row counts, plus output metadata and cached report counts; it does not revalidate every output value. Changed inputs or incompatible saved work cause an error instead of silently repeating cleaning. Incomplete uncached outputs are not overwritten.

`--resume-run <run_id>` selects a specific saved run. Only `--new-run` intentionally repeats the work in a new folder. Stop the previous process before restarting; do not run two copies against the same saved run.

## [3] Logistic-regression input preparation

Run `src/logistic_regression/preprocessing.py` normally for the 2012-2023 modelling scope. It selects a completed data-preparation run automatically. The corrected repayment-policy run has completed: 19,609,717 eligible rows, comprising 7,770,079 development, 1,207,643 validation and 10,631,995 final-test observations. It retained 1,878,495 verified repayments as 0 and preserved existing bad-event labels. These are input-preparation counts, not model-performance results.

- Checks loan IDs across the selected years and checks the existing temporal eligibility rules.
- First applies the agreed repayment rule through `src/data_preparation/repayment_targets.py`: verified voluntary full repayment (code 01), adequate history through payoff and no prior qualifying bad event -> 0; an existing bad-event label remains 1, even if subsequently repaid. Unresolved exits/sales remain unknown. This is an observed outcome, not imputation.
- Reuses the cleaned Parquets without repeating cleaning. For this target update only five monthly columns are read, and only previously unlabelled early-exit IDs are grouped. Compatible corrected targets are reused on subsequent runs.
- Preserves every base candidate row and records old labels/reasons plus repayment markers in `data/prepared/<preparation_run>/target_revisions/verified_repayment_v2/<year>/cohort_candidates.parquet`. The original preparation reports/candidates remain historical records under their original policy.
- Excludes remaining unknown outcomes and split-ineligible loans; outcomes are never filled with guesses. Time boundaries are unchanged, and payoff confirmation cannot make outcome availability earlier than the original horizon-plus-lag assumption.
- Learns numeric medians, scaling and category lists from eligible development rows only, then applies the same settings to validation/test rows.
- Adds numeric missing-value indicators, encodes missing/unseen categories as unknown, and one-hot encodes categories. Borrower count is grouped as single/multiple to reconcile the historical disclosure change.
- Writes revised model-input Parquets to `data/model_inputs/logistic_regression/<preparation_run>/verified_repayment_v2/` and a plain Markdown report plus fitted preprocessing settings to `results/logistic_regression/<preparation_run>/verified_repayment_v2/`. The report compares labels before/after and shows the newly included repayments. Target checkpoints are under the matching `results/data_preparation/<preparation_run>/target_revisions/verified_repayment_v2/` folder.
- Reuses compatible revised outputs after metadata checks; older-policy inputs are preserved but never reused for the new policy. Training medians/scales/categories are refitted because the eligible training population changes. No model is trained; other research rules remain provisional.

Only the saved manifest's `feature_names` are predictors. Loan IDs, dates, split labels and the future outcome are metadata/targets, never predictors.

## [4] First logistic-regression baseline

Code written; training and synthetic tests have not been run by the assistant. After corrected preprocessing finishes, run `src/logistic_regression/train.py` normally in PyCharm; no parameters are needed.

- Reads only completed `verified_repayment_v2` inputs and the explicit feature whitelist. Uses all eligible development loans, without sampling, class weights or oversampling.
- Fits L2-regularized logistic regression with L-BFGS. Arrow batches and a preallocated sparse matrix keep the 90 encoded features from becoming a large dense table. The default allocation guard is 6 GiB, not a hard process/OS memory cap; actual runtime and memory use are not yet benchmarked.
- Scores validation loans only. Reports ROC AUC/Gini, average precision, log loss, Brier score, calibration by risk band, observed-versus-predicted event rates, high-risk event capture and observation-year results.
- Compares against a constant probability equal to the development bad-event rate. Overall accuracy is not used as the main success measure.
- Records optimizer convergence, fitting time, coefficients and individual validation predictions. No probability recalibration or lending cutoff is selected automatically; no final-test scores are produced.
- Saves `report.md`, `metrics.json`, `risk_bands.csv`, `coefficients.csv` and `validation_predictions.parquet` under `results/logistic_regression/<preparation_run>/verified_repayment_v2/baseline/`.
- Saves numeric model weights (NPZ, no pickle) and fitted preprocessing settings under `data/models/logistic_regression/<preparation_run>/verified_repayment_v2/baseline/`. A matching rerun reuses weights and refreshes validation diagnostics; use `--name another_name` for an intentional new experiment with different settings.

This is an exploratory baseline, not a model-approval decision. Nonconvergence is a visible failure to finalize the baseline; ranking, probability calibration and stability must be reviewed together before proceeding to ML or the final test.

Current results for reading: [Main report](results/report.md). This short report is updated automatically when a model run finishes; run-specific folders contain supporting diagnostics and checkpoints.

## Current research checkpoint

Verified against saved reports on 30 October 2026.

- Data extraction/conversion and data preparation are complete for 2012-2025. The first model uses eligible loans from the 2012-2023 vintages.
- Extracted TXT copies have been deleted locally; ZIP archives and Parquet data remain. The earlier storage paragraph records the situation before that deletion.
- Corrected input preparation is complete under `verified_repayment_v2`: 19,609,717 eligible loans, including 7,770,079 development, 1,207,643 validation and 10,631,995 reserved final-test observations.
- The logistic-regression baseline has been fitted and evaluated on validation data. The earlier stage tables and implementation notes retain their historical status statements.
- Actual training observation dates are January 2012-November 2016; validation observation dates are January-November 2018. Outcome-availability rules remove observations near split boundaries so future information cannot enter training. These are observation dates, not origination-year ranges.
- The 2020-2023 final test remains unevaluated. Year-by-year inclusion/exclusion checks, temporal robustness and calibration review remain to be done before further model changes or final testing.
- Gradient boosting, portfolio expected loss, monitoring and stress testing remain planned work.

## Latest saved logistic-regression evaluation

The following evaluation is copied from the code-generated report for the run completed on 29 October 2026. It is a saved snapshot; rerunning the model updates the local [main results report](results/report.md), not this paper automatically.

Source: [Generated model report](results/logistic_regression/20261027_122450_c633a3/verified_repayment_v2/baseline/report.md) and [metrics.json](results/logistic_regression/20261027_122450_c633a3/verified_repayment_v2/baseline/metrics.json). Generated data and reports remain local and excluded from Git.

Status: BASELINE FITTED - VALIDATION ONLY; NOT APPROVED
Preparation run: 20261027_122450_c633a3

Exploratory baseline, not a model-approval or regulatory validation decision.
Target: a qualifying serious-delinquency/credit-loss event within 12 months. Verified full repayment without an earlier bad event is 0; a later payment does not erase an earlier event.

### Design

- Development rows only are used for fitting. All eligible development loans are used; no sampling or class balancing.
- L2-regularized logistic regression, fitted with L-BFGS on a sparse matrix.
- Only manifest feature_names enter the model. IDs, future outcomes and repayment markers are not predictors.
- Imputation, encoding and scaling reuse the development-only preprocessing settings.
- Validation is evaluated without refitting, tuning, threshold selection or recalibration.
- The final test split is not scored or evaluated by this script.
- These are raw model probabilities, not calibrated/approved production PDs.

### Training

| Measure | Value |
|---|---:|
| Loans | 7,770,079 |
| Bad events | 12,717 |
| Observed bad-event rate | 0.164% |
| First observation | 2012-01-01 |
| Last observation | 2016-11-01 |
| Encoded features | 90 |
| Training load seconds | 18.5 |
| Model fit seconds | 14.0 |
| Optimizer iterations | 31 |
| Converged | True |

### Validation results

Loans: 1,207,643. Observed bad events: 2,546.
Observation dates: 2018-01-01 to 2018-11-01.

| Measure | Model | Comparison / meaning |
|---|---:|---|
| ROC AUC | 0.799921 | 0.5 is random ranking; higher is better |
| Gini | 0.599842 | 2 x AUC - 1; higher is better |
| Average precision (AP) | 0.0104154 | No-skill reference: 0.00210824; higher is better |
| Log loss | 0.013837 | Constant training-rate benchmark: 0.0151591; lower is better |
| Brier score | 0.0020962 | Constant training-rate benchmark: 0.00210402; lower is better |
| Brier skill | 0.00371336 | Positive means better than the constant benchmark |
| Mean predicted probability | 0.171% | Compare with the observed event rate below |
| Observed event rate | 0.211% | Actual validation outcomes |
| Observed / expected events | 1.23287 | 1 is aggregate agreement; above 1 means underprediction |
| Events captured in riskiest 10% | 45.954% | Random ranking captures approximately 10% |

The benchmark assigns every validation loan the development bad-event rate. It does not learn the validation event rate.

AP is average precision, not trapezoidal PR-AUC. A high ROC AUC does not by itself establish accurate probabilities or financial profitability.

### Risk bands: validation only

Band 1 contains the highest predicted risks. Bands are approximately equal-sized; tied scores can span bands.

| Band | Loans | Bad events | Mean predicted PD | Observed event rate | Observed / expected |
|---|---:|---:|---:|---:|---:|
| 1 | 120,765 | 1,170 | 0.605% | 0.969% | 1.60098 |
| 2 | 120,765 | 482 | 0.301% | 0.399% | 1.32736 |
| 3 | 120,765 | 271 | 0.214% | 0.224% | 1.04637 |
| 4 | 120,764 | 213 | 0.164% | 0.176% | 1.07781 |
| 5 | 120,764 | 117 | 0.128% | 0.097% | 0.757067 |
| 6 | 120,764 | 103 | 0.101% | 0.085% | 0.848452 |
| 7 | 120,764 | 75 | 0.078% | 0.062% | 0.79821 |
| 8 | 120,764 | 51 | 0.058% | 0.042% | 0.723242 |
| 9 | 120,764 | 42 | 0.041% | 0.035% | 0.857277 |
| 10 | 120,764 | 22 | 0.021% | 0.018% | 0.875001 |

### Validation by observation year

| Year | Loans | Bad events | ROC AUC | Mean PD | Observed rate |
|---|---:|---:|---:|---:|---:|
| 2018 | 1,207,643 | 2,546 | 0.799921 | 0.171% | 0.211% |

### How to evaluate this first result

1. Check convergence before trusting the fitted baseline.
2. Check ranking (AUC/AP) and improvement over the constant benchmark.
3. Compare predicted and observed event rates, overall and within the risk bands.
4. Check performance by observation year; do not equate origination year with prediction date.
5. Do not declare success from one metric. No universal acceptance threshold is imposed here.

### Output files

- metrics.json: configuration, data fingerprints, training information and validation metrics.
- validation_predictions.parquet: individual probabilities and observed validation outcomes.
- risk_bands.csv and coefficients.csv: readable diagnostics.
- Model weights and the fitted input-preparation settings are saved under data/models/logistic_regression/.

Coefficients describe fitted associations, not causal effects. Numeric coefficients refer to standardized inputs; full one-hot categories are regularized jointly.
Target, selection and temporal-design limitations still apply. No final-test performance, lending-profit estimate or approval decision is produced.

