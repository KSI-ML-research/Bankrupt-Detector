# Bankrupt-Detector

Research project predicting the risk of late B2B invoice payments as an early signal of SME cash-flow / liquidity risk, based on invoice data and customer payment history.

**Project in progress.** Done: EDA, concept drift analysis, leakage-safe historical customer features, baseline (logistic regression), XGBoost + SHAP + Optuna tuning, survival analysis (Kaplan-Meier, Cox PH). Next up: Random Survival Forest and cash-flow forecasting.

## Project vision

The end goal is a system that answers: *will the company face a cash-flow gap in the next 30/60/90 days?* It consists of three layers: classification (will a given invoice be paid late), survival analysis (how many days the payment will be delayed and with what probability), and cash-flow forecasting (aggregating per-invoice risk into a company-level balance forecast).

## Approach

- **Data:** ~50k B2B invoices, invoice features + customer payment history, chronological (not random) train/test split.
- **Models:** logistic regression (baseline) → XGBoost + SHAP → survival analysis (Kaplan-Meier, Cox PH, Random Survival Forest planned).

## Selected results

| Model | Metric | Result |
|---|---|---|
| Logistic regression | ROC-AUC | 0.72 |
| XGBoost + Optuna | ROC-AUC | 0.80 |
| Kaplan-Meier (BE vs SME) | log-rank p-value | 2.97e-29 |
| Cox PH | C-index (test) | 0.45 – suggests the model's assumptions are violated, motivating the move to RSF |

## Repository structure

- `eda/` — exploratory data analysis
- `baseline_logistic_regression/`, `baseline_other/` — baseline models
- `SHAP_model/` — XGBoost, tuning, SHAP explanations
- `phase1_survival_analisys/`, `cox_proportional_hazards/`, `random_survival_forest/` — survival analysis
- `utils/` — data preparation and cleaning
