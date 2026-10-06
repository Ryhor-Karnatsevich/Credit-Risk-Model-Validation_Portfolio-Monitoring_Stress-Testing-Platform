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

## Research stages

| Index | Name | Current situation |
|---:|---|---|
| 0 | Define research objective and model use | In progress |
| 1 | Identify candidate public datasets | |
| 2 | Compare datasets and complete the feasibility gate | |
| 3 | Define the model population | |
| 4 | Define default and the performance horizon | |
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
