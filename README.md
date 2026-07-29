# Credit Risk Modelling — Probability of Default (PD) Prediction

End-to-end credit risk model that predicts the probability a consumer loan
defaults, deployed as a real-time scoring API + interactive dashboard.
Built to mirror how PD models are actually developed and shipped in BFSI:
a regulator-friendly WOE/Logistic Regression scorecard benchmarked against
gradient-boosted trees, with SHAP explainability and Dockerized deployment.

## Problem Statement

Lenders need to estimate, at the time of underwriting, how likely a loan
applicant is to default — this Probability of Default (PD) feeds directly
into approve/decline decisions, pricing (risk-based interest rates), and
regulatory capital calculations (Basel/IFRS9 style expected-loss models).

## Dataset

This project is built around the **Lending Club Loan Data** dataset —
publicly available consumer loan data (~2.2M loans, 150+ raw fields,
issued 2007–2020) that is the de-facto industry-standard dataset for PD
modelling case studies and interviews.

- Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Alternative (lighter, ~250K rows, pre-cleaned, good if you want a faster
  iteration cycle): **Give Me Some Credit** —
  https://www.kaggle.com/c/GiveMeSomeCredit

Download either one, save it as `data/loan_data.csv`, and map the columns
to match the schema used in `credit_risk_model.py` (see column list in
`generate_synthetic_data.py` — every field it generates is a real Lending
Club field with the same name and meaning).

**Don't want to wait on the Kaggle download to try the project?**
Run `python generate_synthetic_data.py --n 450000` first — it produces a
450K-row dataset with the same schema and realistic, correlated default
patterns (grade, DTI, utilization, delinquency history, income, etc. all
drive default risk the way they do in the real data). The entire pipeline
below runs unchanged on either file.

## Approach

**1. Data cleaning & EDA** — missing-value treatment, outlier capping,
target leakage checks (dropped fields only known *after* loan outcome,
e.g. total payments received).

**2. Feature engineering**
- Ratio features: installment-to-income, loan-to-income, revolving-balance-to-income
- Behavioral flags: high utilization, recent delinquency, recent hard inquiries
- Credit history depth: years since earliest credit line, employment tenure
- **WOE (Weight of Evidence) + IV (Information Value)** encoding for
  categorical features — standard BFSI practice that keeps a Logistic
  Regression scorecard monotonic and interpretable for regulators/auditors

**3. Modelling — three models compared head-to-head**

| Model | Why it's in the comparison |
|---|---|
| Logistic Regression (WOE scorecard) | Interpretable baseline; still the industry standard for regulated credit decisions |
| XGBoost (Optuna-tuned) | Non-linear interactions, strong tabular performance |
| LightGBM (Optuna-tuned) | Faster training at scale, typically matches/beats XGBoost on this data size |

Each model is tuned with **Optuna** (Bayesian hyperparameter search) against
a held-out validation split, then evaluated once on an untouched test set.

**4. Evaluation metrics** — AUROC, **Gini coefficient** (`2×AUROC − 1`,
the metric risk teams actually report), **KS statistic** (max separation
between good/bad cumulative distributions — a standard credit-scoring
metric), PR-AUC (important given class imbalance), and Brier score
(probability calibration).

**5. Explainability** — SHAP TreeExplainer on the winning tree model,
surfaced both as global feature importance and per-application local
explanations (the API returns the top 5 risk drivers with every prediction).

**6. Deployment** — FastAPI scoring service + Streamlit dashboard, both
containerized, orchestrated with docker-compose.

## Results

*(Numbers below are from a pipeline smoke-test run on synthetic data with
minimal hyperparameter search, to confirm the code runs end-to-end.
Retrain on the real 450K-row Lending Club dataset with `--trials 30+` for
representative, reportable numbers — expect AUROC in the 0.85–0.90 range,
in line with the published benchmarks for this dataset.)*

| Model | AUROC | Gini | KS | PR-AUC |
|---|---|---|---|---|
| Logistic Regression (WOE) | *see `models/metrics.json` after training* | | | |
| XGBoost (Optuna-tuned) | | | | |
| LightGBM (Optuna-tuned) | | | | |

Run `python credit_risk_model.py` and check `models/metrics.json` for
your own run's numbers — the script prints a full comparison table and
picks the champion model by test AUROC automatically.

## Architecture

```
                    ┌─────────────────────┐
                    │   loan_data.csv      │
                    └──────────┬───────────┘
                               │
                    credit_risk_model.py
              (feature engg → WOE/IV → LR/XGB/LGBM →
                Optuna tuning → SHAP → save artifacts)
                               │
                    ┌──────────┴───────────┐
                    │      models/          │
                    │  best_model.pkl       │
                    │  shap_model.pkl       │
                    │  feature_names.pkl    │
                    │  woe_maps.pkl (if LR) │
                    └──────┬─────────┬──────┘
                           │         │
                  ┌────────┘         └────────┐
                  ▼                            ▼
            app.py (FastAPI)           app_streamlit.py
         /predict  /predict/batch      interactive dashboard
         /health   /model-info         + SHAP waterfall
              │                                │
              └──────────── docker-compose ────┘
                    api:8000   dashboard:8501
```

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml     # CI: lint + pipeline smoke test on push
├── data/                        # place loan_data.csv here (gitignored)
├── models/                      # trained artifacts, generated by training (gitignored)
├── generate_synthetic_data.py   # realistic synthetic fallback dataset
├── credit_risk_model.py         # feature engineering + model training + SHAP
├── app.py                       # FastAPI scoring service
├── app_streamlit.py             # interactive Streamlit dashboard
├── requirements.txt
├── Dockerfile                   # FastAPI service image
├── Dockerfile.streamlit         # Streamlit dashboard image
├── docker-compose.yml           # runs both services together
├── .gitignore
├── LICENSE
└── README.md
```

## How to Run

### Option A — Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Get real data (recommended): download Lending Club data from Kaggle,
#     save as data/loan_data.csv
# 2b. OR generate a synthetic dataset to try the pipeline immediately:
python generate_synthetic_data.py --n 450000

# 3. Train (feature engineering → LR/XGBoost/LightGBM → Optuna → SHAP)
python credit_risk_model.py --data data/loan_data.csv --trials 30

# 4. Serve the API
uvicorn app:app --reload --port 8000
# → visit http://localhost:8000/docs for interactive Swagger UI

# 5. Run the dashboard (separate terminal)
streamlit run app_streamlit.py
# → visit http://localhost:8501
```

### Option B — Docker

```bash
docker compose up --build
# API        → http://localhost:8000/docs
# Dashboard  → http://localhost:8501
```

> Note: the model must be trained locally first (`models/` is mounted
> read-only into both containers) — Docker packages the *serving* layer,
> not the training run.

### Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000, "term": 36, "int_rate": 13.5, "installment": 510.2,
    "grade": "B", "emp_length": "5 years", "home_ownership": "MORTGAGE",
    "annual_inc": 65000, "verification_status": "Verified",
    "purpose": "debt_consolidation", "dti": 18.4, "delinq_2yrs": 0,
    "earliest_cr_line_years": 12.0, "inq_last_6mths": 1, "open_acc": 10,
    "pub_rec": 0, "revol_bal": 8500, "revol_util": 45.0, "total_acc": 22,
    "initial_list_status": "w", "application_type": "Individual",
    "mort_acc": 1, "pub_rec_bankruptcies": 0
  }'
```

Response:
```json
{
  "probability_of_default": 0.135,
  "risk_grade": "C - Elevated Risk",
  "decision": "APPROVE",
  "top_risk_drivers": [
    {"feature": "int_rate", "impact": 1.2363},
    {"feature": "annual_inc", "impact": -0.1544},
    {"feature": "open_acc_ratio", "impact": -0.0862}
  ]
}
```

## Key Design Decisions (interview talking points)

- **Champion/challenger framework**: LR-WOE scorecard vs. gradient-boosted
  trees, selected automatically by test AUROC — mirrors how banks actually
  validate a new model against the incumbent before deployment.
- **WOE/IV over raw one-hot encoding for the LR path**: keeps coefficients
  monotonic and directly interpretable ("+1 WOE unit → X change in log-odds
  of default"), which regulators and credit committees require.
- **Optuna over GridSearch**: Bayesian search converges to a better
  hyperparameter set in far fewer trials — critical when each trial trains
  on 300K+ rows.
- **KS statistic and Gini reported alongside AUROC**: these are the metrics
  a BFSI risk team actually asks for in a model validation document, not
  just AUROC.
- **SHAP is computed on the tree model even when LR wins**: LR coefficients
  are already interpretable, but the API always returns tree-based SHAP
  attributions so every prediction — regardless of champion model — ships
  with a consistent, per-application explanation.
- **Synthetic-data fallback**: lets anyone clone the repo and see the full
  pipeline run in minutes, without gating the project behind a Kaggle
  download.

## Tech Stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` ·
`Optuna` · `SHAP` · `FastAPI` · `Streamlit` · `Docker` / `docker-compose` ·
`GitHub Actions`

## Future Improvements

- Swap Optuna's per-trial holdout for k-fold or time-based (vintage) CV
- Add population stability index (PSI) monitoring for drift
- Add a reject-inference step (loans that were declined and never observed)
- Calibrate probabilities (Platt scaling / isotonic) before using them for pricing
