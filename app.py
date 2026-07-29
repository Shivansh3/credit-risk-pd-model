"""
app.py
---------------------------------
FastAPI microservice for real-time Probability of Default (PD) scoring.

Endpoints
    GET  /health           - liveness check
    GET  /model-info        - metadata about the deployed model
    POST /predict           - score a single loan application
    POST /predict/batch     - score a list of loan applications

Run locally:
    uvicorn app:app --reload --port 8000

Run via Docker:
    docker compose up
"""
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk_model import engineer_features, TARGET

ARTIFACT_DIR = Path("models")

app = FastAPI(
    title="Credit Risk PD Scoring API",
    description="Real-time probability-of-default scoring for consumer loan applications.",
    version="1.0.0",
)

# ---------------------------------------------------------------------- #
# Load artifacts once at startup
# ---------------------------------------------------------------------- #
model = None
feature_names = None
model_meta = None
shap_model = None
explainer = None
woe_maps = None
lr_scaler = None


@app.on_event("startup")
def load_artifacts():
    global model, feature_names, model_meta, shap_model, explainer, woe_maps, lr_scaler
    try:
        model = joblib.load(ARTIFACT_DIR / "best_model.pkl")
        feature_names = joblib.load(ARTIFACT_DIR / "feature_names.pkl")
        model_meta = joblib.load(ARTIFACT_DIR / "model_meta.pkl")
        shap_model = joblib.load(ARTIFACT_DIR / "shap_model.pkl")
        explainer = shap.TreeExplainer(shap_model)
        if model_meta["model_type"] == "logistic_regression":
            woe_maps = joblib.load(ARTIFACT_DIR / "woe_maps.pkl")
            lr_scaler = joblib.load(ARTIFACT_DIR / "lr_scaler.pkl")
        print(f"Loaded model: {model_meta['model_type']}, {len(feature_names)} features")
    except FileNotFoundError:
        print("WARNING: model artifacts not found. Run `python credit_risk_model.py` first.")


# ---------------------------------------------------------------------- #
# Schemas
# ---------------------------------------------------------------------- #
class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., example=15000, description="Requested loan amount")
    term: int = Field(..., example=36, description="Loan term in months (36 or 60)")
    int_rate: float = Field(..., example=13.5, description="Interest rate (%)")
    installment: float = Field(..., example=510.2, description="Monthly installment")
    grade: str = Field(..., example="B", description="Internal risk grade A-G")
    emp_length: str = Field(..., example="5 years")
    home_ownership: str = Field(..., example="MORTGAGE")
    annual_inc: float = Field(..., example=65000)
    verification_status: str = Field(..., example="Verified")
    purpose: str = Field(..., example="debt_consolidation")
    dti: float = Field(..., example=18.4, description="Debt-to-income ratio")
    delinq_2yrs: int = Field(0, example=0)
    earliest_cr_line_years: float = Field(..., example=12.0, description="Years since first credit line")
    inq_last_6mths: int = Field(0, example=1)
    open_acc: int = Field(..., example=10)
    pub_rec: int = Field(0, example=0)
    revol_bal: float = Field(..., example=8500)
    revol_util: float = Field(..., example=45.0)
    total_acc: int = Field(..., example=22)
    initial_list_status: str = Field(..., example="w")
    application_type: str = Field("Individual", example="Individual")
    mort_acc: int = Field(0, example=1)
    pub_rec_bankruptcies: int = Field(0, example=0)


class PredictionResponse(BaseModel):
    probability_of_default: float
    risk_grade: str
    decision: str
    top_risk_drivers: List[dict]


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def grade_from_pd(pd_value: float) -> str:
    if pd_value < 0.03:
        return "A - Low Risk"
    elif pd_value < 0.08:
        return "B - Moderate Risk"
    elif pd_value < 0.15:
        return "C - Elevated Risk"
    elif pd_value < 0.25:
        return "D - High Risk"
    return "E - Very High Risk"


def decision_from_pd(pd_value: float) -> str:
    return "APPROVE" if pd_value < 0.15 else ("REVIEW" if pd_value < 0.25 else "DECLINE")


def prepare_tree_features(df_fe: pd.DataFrame) -> pd.DataFrame:
    """One-hot feature matrix used by XGBoost / LightGBM (and for SHAP, always)."""
    cat_cols = df_fe.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df_fe.select_dtypes(include=[np.number]).columns]
    X = pd.get_dummies(df_fe[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt_") for c in X.columns]
    return X.reindex(columns=feature_names, fill_value=0)


def prepare_lr_features(df_fe: pd.DataFrame) -> np.ndarray:
    """WOE-encoded + scaled feature matrix used by the Logistic Regression scorecard."""
    cat_cols = model_meta["cat_cols"]
    woe_cols = model_meta["woe_cols"]
    df_woe = df_fe.copy()
    for col in cat_cols:
        df_woe[col + "_woe"] = df_fe[col].map(woe_maps[col]).fillna(0)
    X_woe = df_woe[woe_cols]
    return lr_scaler.transform(X_woe)


def prepare_features(payload: dict):
    df = pd.DataFrame([payload])
    df_fe = engineer_features(df.assign(**{TARGET: 0})).drop(columns=[TARGET])
    X_tree = prepare_tree_features(df_fe)  # always built — used for SHAP either way
    if model_meta["model_type"] == "logistic_regression":
        return prepare_lr_features(df_fe), X_tree
    return X_tree, X_tree


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    if model_meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": model_meta["model_type"],
        "n_features": len(feature_names),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    X_model, X_tree = prepare_features(application.dict())
    prob = float(model.predict_proba(X_model)[:, 1][0])

    # SHAP explanation always computed on the tree feature space (SHAP TreeExplainer
    # runs against shap_model, which is always a gradient-boosted model — see training script)
    shap_vals = explainer.shap_values(X_tree)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    contributions = pd.Series(shap_vals[0], index=X_tree.columns).sort_values(key=np.abs, ascending=False).head(5)
    top_drivers = [{"feature": f, "impact": round(float(v), 4)} for f, v in contributions.items()]

    return PredictionResponse(
        probability_of_default=round(prob, 4),
        risk_grade=grade_from_pd(prob),
        decision=decision_from_pd(prob),
        top_risk_drivers=top_drivers,
    )


@app.post("/predict/batch")
def predict_batch(applications: List[LoanApplication]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    results = []
    for app_in in applications:
        X_model, _ = prepare_features(app_in.dict())
        prob = float(model.predict_proba(X_model)[:, 1][0])
        results.append({
            "probability_of_default": round(prob, 4),
            "risk_grade": grade_from_pd(prob),
            "decision": decision_from_pd(prob),
        })
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
