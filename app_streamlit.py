"""
app_streamlit.py
---------------------------------
Interactive Streamlit dashboard for the Credit Risk PD model.
Loads the trained model artifacts directly (no API dependency), so it can
run standalone with `streamlit run app_streamlit.py`, or alongside the
FastAPI service via docker-compose.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.graph_objects as go

from credit_risk_model import engineer_features, TARGET, grade_from_pd, decision_from_pd

ARTIFACT_DIR = Path("models")

st.set_page_config(page_title="Credit Risk PD Dashboard", page_icon="💳", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "best_model.pkl")
    feature_names = joblib.load(ARTIFACT_DIR / "feature_names.pkl")
    model_meta = joblib.load(ARTIFACT_DIR / "model_meta.pkl")
    shap_model = joblib.load(ARTIFACT_DIR / "shap_model.pkl")
    explainer = shap.TreeExplainer(shap_model)
    woe_maps, lr_scaler = None, None
    if model_meta["model_type"] == "logistic_regression":
        woe_maps = joblib.load(ARTIFACT_DIR / "woe_maps.pkl")
        lr_scaler = joblib.load(ARTIFACT_DIR / "lr_scaler.pkl")
    return model, feature_names, model_meta, shap_model, explainer, woe_maps, lr_scaler


def prepare_tree_features(df_fe, feature_names):
    cat_cols = df_fe.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df_fe.select_dtypes(include=[np.number]).columns]
    X = pd.get_dummies(df_fe[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt_") for c in X.columns]
    return X.reindex(columns=feature_names, fill_value=0)

    
def prepare_lr_features(df_fe, model_meta, woe_maps, lr_scaler):
    cat_cols = model_meta["cat_cols"]
    woe_cols = model_meta["woe_cols"]
    df_woe = df_fe.copy()
    for col in cat_cols:
        df_woe[col + "_woe"] = df_fe[col].map(woe_maps[col]).fillna(0)
    return lr_scaler.transform(df_woe[woe_cols])


GRADE_COLORS = {
    "A - Low Risk": "#2ecc71",
    "B - Moderate Risk": "#a3d977",
    "C - Elevated Risk": "#f1c40f",
    "D - High Risk": "#e67e22",
    "E - Very High Risk": "#e74c3c",
}


def grade_color(label: str) -> str:
    return GRADE_COLORS.get(label, "#95a5a6")


def validate_inputs(payload: dict, feature_ranges: dict) -> list:
    """Flags any numeric input that falls outside the min/max seen during
    training. Doesn't block scoring -- the model can still produce a number --
    but the number is an extrapolation, not an interpolation, and the user
    should know that before trusting it."""
    warnings = []
    for field, bounds in (feature_ranges or {}).items():
        if field not in payload:
            continue
        value = payload[field]
        lo, hi = bounds.get("min"), bounds.get("max")
        if lo is not None and value < lo:
            warnings.append(f"**{field}** = {value:,} is below the training range minimum ({lo:,.0f}).")
        elif hi is not None and value > hi:
            warnings.append(f"**{field}** = {value:,} is above the training range maximum ({hi:,.0f}).")
    return warnings


st.title("💳 Credit Risk — Probability of Default Dashboard")
st.caption("LightGBM / XGBoost / Logistic-Regression scorecard, with SHAP explainability")

try:
    model, feature_names, model_meta, shap_model, explainer, woe_maps, lr_scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python credit_risk_model.py` first to train the model.")
    st.stop()

st.sidebar.header("Loan Application")
loan_amnt = st.sidebar.slider("Loan amount ($)", 1000, 40000, 15000, step=500)
term = st.sidebar.selectbox("Term (months)", [36, 60])
int_rate = st.sidebar.slider("Interest rate (%)", 5.0, 31.0, 13.5, step=0.1)
grade = st.sidebar.selectbox("Internal grade", list("ABCDEFG"), index=1)
annual_inc = st.sidebar.number_input("Annual income ($)", 15000, 800000, 65000, step=1000)
emp_length = st.sidebar.selectbox(
    "Employment length",
    ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
     "6 years", "7 years", "8 years", "9 years", "10+ years"], index=5)
home_ownership = st.sidebar.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=1)
verification_status = st.sidebar.selectbox("Verification status", ["Verified", "Source Verified", "Not Verified"])
purpose = st.sidebar.selectbox(
    "Loan purpose",
    ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
     "small_business", "medical", "car", "other"])
dti = st.sidebar.slider("Debt-to-income ratio (%)", 0.0, 60.0, 18.0, step=0.5)
revol_util = st.sidebar.slider("Revolving credit utilization (%)", 0.0, 150.0, 45.0, step=1.0)
revol_bal = st.sidebar.number_input("Revolving balance ($)", 0, 200000, 8500, step=500)
delinq_2yrs = st.sidebar.number_input("Delinquencies (last 2 yrs)", 0, 10, 0)
inq_last_6mths = st.sidebar.number_input("Inquiries (last 6 months)", 0, 10, 1)
pub_rec = st.sidebar.number_input("Public records", 0, 5, 0)
pub_rec_bankruptcies = st.sidebar.number_input("Bankruptcies", 0, 3, 0)
mort_acc = st.sidebar.number_input("Mortgage accounts", 0, 10, 1)
open_acc = st.sidebar.number_input("Open credit lines", 1, 40, 10)
total_acc = st.sidebar.number_input("Total credit lines", 1, 60, 22)
earliest_cr_line_years = st.sidebar.slider("Years of credit history", 1, 45, 12)
application_type = st.sidebar.selectbox("Application type", ["Individual", "Joint App"])

installment = round((loan_amnt * (int_rate / 1200)) / (1 - (1 + int_rate / 1200) ** (-term)), 2)

payload = {
    "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "installment": installment,
    "grade": grade, "emp_length": emp_length, "home_ownership": home_ownership,
    "annual_inc": annual_inc, "verification_status": verification_status, "purpose": purpose,
    "dti": dti, "delinq_2yrs": delinq_2yrs, "earliest_cr_line_years": earliest_cr_line_years,
    "inq_last_6mths": inq_last_6mths, "open_acc": open_acc, "pub_rec": pub_rec,
    "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
    "initial_list_status": "w", "application_type": application_type,
    "mort_acc": mort_acc, "pub_rec_bankruptcies": pub_rec_bankruptcies,
}

out_of_range_warnings = validate_inputs(payload, model_meta.get("feature_ranges"))
if out_of_range_warnings:
    st.warning(
        "⚠️ **Some inputs fall outside the range seen during training** — the model "
        "is extrapolating for these fields, so treat this prediction with extra caution:\n\n"
        + "\n".join(f"- {w}" for w in out_of_range_warnings)
    )

df = pd.DataFrame([payload])
df_fe = engineer_features(df.assign(**{TARGET: 0})).drop(columns=[TARGET])
X_tree = prepare_tree_features(df_fe, feature_names)

if model_meta["model_type"] == "logistic_regression":
    X_model = prepare_lr_features(df_fe, model_meta, woe_maps, lr_scaler)
else:
    X_model = X_tree

prob = float(model.predict_proba(X_model)[:, 1][0])
risk_label = grade_from_pd(prob)
color = grade_color(risk_label)
decision = decision_from_pd(prob)

approval_threshold = model_meta.get("approval_threshold", 0.15)
decline_threshold = model_meta.get("decline_threshold", 0.25)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Probability of Default"},
        gauge={
            "axis": {"range": [0, 50]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 3], "color": "#eafaf1"},
                {"range": [3, 8], "color": "#e8f8d4"},
                {"range": [8, 15], "color": "#fdf3d0"},
                {"range": [15, 25], "color": "#fbe4cc"},
                {"range": [25, 50], "color": "#fadbd8"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Risk Grade", risk_label)
    st.metric("Model Decision", decision)
    st.metric("Model", model_meta["model_type"].replace("_", " ").title())
    st.markdown(
    f"""
    **Decision Policy**

    - **Approve:** PD < {approval_threshold:.0%}
    - **Review:** {approval_threshold:.0%} ≤ PD < {decline_threshold:.0%}
    - **Decline:** PD ≥ {decline_threshold:.0%}
    """
    )

with col3:
    st.metric("Monthly Installment", f"${installment:,.2f}")
    st.metric("Loan-to-Income", f"{(loan_amnt/annual_inc):.2f}")
    st.metric("Installment-to-Income", f"{(installment*12/annual_inc):.2%}")

# ---- historical risk-band summary ----
# Ties the live prediction back to what actually happened, historically, to
# other borrowers who landed in the same risk band -- makes the score
# interpretable as more than just "a number the model produced."
band_stats = (model_meta.get("risk_band_summary") or {}).get(risk_label)
if band_stats and band_stats.get("observed_default_rate") is not None:
    st.info(
        f"📊 **Historical context:** customers in the **{risk_label}** band historically had an "
        f"observed default rate of **~{band_stats['observed_default_rate']:.1%}** "
        f"on the held-out test set (n = {band_stats['n']:,})."
    )

st.divider()
st.subheader("Why this score? — SHAP feature contributions")

shap_vals = explainer.shap_values(X_tree)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
contributions = pd.Series(shap_vals[0], index=X_tree.columns).sort_values(key=np.abs, ascending=True).tail(12)

fig2 = go.Figure(go.Bar(
    x=contributions.values,
    y=contributions.index,
    orientation="h",
    marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in contributions.values],
))
fig2.update_layout(
    height=420,
    xaxis_title="SHAP value (impact on predicted default probability)",
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Red bars push the prediction toward higher default risk; green bars push it toward lower risk.")



