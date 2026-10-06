# Credit Risk Model Validation, Portfolio Monitoring & Stress Testing Platform

An end-to-end credit-risk project covering PD model development, independent validation, portfolio monitoring, expected-loss analysis, and stress testing.

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
| Freddie Mac Single-Family Loan-Level Dataset | Primary origination, performance and actual-loss data | Public sample and metadata extracted; full historical files require free registration and acceptance of the data terms |
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
