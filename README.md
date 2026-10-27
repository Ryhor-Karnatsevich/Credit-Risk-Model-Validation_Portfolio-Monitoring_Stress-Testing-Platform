# Credit Risk Model Validation, Portfolio Monitoring & Stress Testing Platform

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

