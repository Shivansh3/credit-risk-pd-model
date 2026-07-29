"""
credit_risk_model.py
---------------------------------
End-to-end Probability of Default (PD) modelling pipeline.

Stages
    1. Load & clean raw consumer-loan data
    2. Feature engineering (ratios, WOE/IV for categoricals, credit-history features)
    3. Train/test split (stratified, with a held-out OOT-style split option)
    4. Train 3 models: Logistic Regression (WOE-based scorecard baseline),
       XGBoost, LightGBM (Optuna-tuned)
    5. Evaluate: AUROC, Gini, KS statistic, PR-AUC, Brier score
    6. SHAP explainability on the best model
    7. Persist the winning model + preprocessing artifacts for app.py / app_streamlit.py

Run:
    python credit_risk_model.py --data data/loan_data.csv --trials 30
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET = "loan_status_default"
ARTIFACT_DIR = Path("models")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Single source of truth for PD -> risk grade bucketing, so training-time
# "historical observed default rate per band" lines up exactly with what
# app.py / app_streamlit.py show for a live prediction.
RISK_BANDS = [
    (0.03, "A - Low Risk"),
    (0.08, "B - Moderate Risk"),
    (0.15, "C - Elevated Risk"),
    (0.25, "D - High Risk"),
    (float("inf"), "E - Very High Risk"),
]
APPROVAL_THRESHOLD = 0.15   # PD below this -> APPROVE
DECLINE_THRESHOLD = 0.25    # PD at/above this -> DECLINE, otherwise REVIEW


def grade_from_pd(p: float) -> str:
    for cutoff, label in RISK_BANDS:
        if p < cutoff:
            return label
    return RISK_BANDS[-1][1]


def decision_from_pd(p: float) -> str:
    if p < APPROVAL_THRESHOLD:
        return "APPROVE"
    elif p < DECLINE_THRESHOLD:
        return "REVIEW"
    return "DECLINE"


# --------------------------------------------------------------------------- #
# 1. LOAD
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Default rate: {df[TARGET].mean():.2%}")
    return df


# --------------------------------------------------------------------------- #
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- missing value handling ----
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET, errors="ignore")
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    cat_cols = df.select_dtypes(include="object").columns
    for c in cat_cols:
        df[c] = df[c].fillna("Missing")

    # ---- domain ratio features (this is where most PD lift comes from) ----
    df["installment_to_income"] = (df["installment"] * 12) / (df["annual_inc"] + 1)
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["revol_bal_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)
    df["credit_util_flag_high"] = (df["revol_util"] > 80).astype(int)
    df["open_acc_ratio"] = df["open_acc"] / (df["total_acc"] + 1)
    df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)
    df["has_public_record"] = (df["pub_rec"] > 0).astype(int)
    df["recent_inquiries_flag"] = (df["inq_last_6mths"] >= 3).astype(int)

    emp_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10, "Missing": -1,
    }
    if "emp_length" in df.columns:
        df["emp_length_years"] = df["emp_length"].map(emp_map).fillna(-1)

    df["income_bucket"] = pd.cut(
        df["annual_inc"], bins=[0, 30000, 60000, 100000, 150000, np.inf],
        labels=["<30k", "30-60k", "60-100k", "100-150k", "150k+"]
    ).astype(str)

    return df


def woe_encode(df: pd.DataFrame, cat_cols: list, target: str, min_bin=50):
    """Weight-of-Evidence encoding used for the Logistic Regression scorecard
    baseline — standard practice in BFSI credit risk (keeps the LR model
    monotonic, interpretable, and regulator-friendly)."""
    woe_maps, iv_summary = {}, {}
    encoded = df.copy()
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()

    for col in cat_cols:
        grp = df.groupby(col)[target].agg(["count", "sum"])
        grp.columns = ["total", "bad"]
        grp["good"] = grp["total"] - grp["bad"]
        grp = grp[grp["total"] >= min_bin]
        grp["good_dist"] = grp["good"].clip(lower=0.5) / total_good
        grp["bad_dist"] = grp["bad"].clip(lower=0.5) / total_bad
        grp["woe"] = np.log(grp["good_dist"] / grp["bad_dist"])
        grp["iv"] = (grp["good_dist"] - grp["bad_dist"]) * grp["woe"]

        woe_maps[col] = grp["woe"].to_dict()
        iv_summary[col] = grp["iv"].sum()
        encoded[col + "_woe"] = df[col].map(woe_maps[col]).fillna(0)

    return encoded, woe_maps, iv_summary


# --------------------------------------------------------------------------- #
# 3. EVALUATION HELPERS
# --------------------------------------------------------------------------- #
def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate(name, y_true, y_prob) -> dict:
    auroc = roc_auc_score(y_true, y_prob)
    metrics = {
        "model": name,
        "auroc": round(auroc, 4),
        "gini": round(2 * auroc - 1, 4),
        "ks": round(ks_statistic(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
    }
    print(f"  [{name}] AUROC={metrics['auroc']}  Gini={metrics['gini']}  "
          f"KS={metrics['ks']}  PR-AUC={metrics['pr_auc']}  Brier={metrics['brier_score']}")
    return metrics


# --------------------------------------------------------------------------- #
# 4. TRAINING
# --------------------------------------------------------------------------- #
def train_logistic_baseline(X_train_woe, y_train, X_test_woe, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_woe)
    X_test_s = scaler.transform(X_test_woe)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
    model.fit(X_train_s, y_train)
    prob = model.predict_proba(X_test_s)[:, 1]
    metrics = evaluate("LogisticRegression (WOE)", y_test, prob)
    return model, scaler, metrics


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=30):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "random_state": 42,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best LightGBM AUROC (val): {study.best_value:.4f}")
    return study.best_params


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=30):
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "random_state": 42,
            "tree_method": "hist",
        }
        model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best XGBoost AUROC (val): {study.best_value:.4f}")
    return study.best_params


# --------------------------------------------------------------------------- #
# 5. MAIN
# --------------------------------------------------------------------------- #
def main(data_path: str, n_trials: int, fast: bool = False):
    df = load_data(data_path)
    df = engineer_features(df)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]

    y = df[TARGET]
    X_tree = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    # sanitize column names: XGBoost rejects [, ], < in feature names
    X_tree.columns = [
        str(c).replace("[", "(").replace("]", ")").replace("<", "lt_")
        for c in X_tree.columns
    ]
    feature_names = X_tree.columns.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_tree, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ---- WOE features for the Logistic Regression scorecard baseline ----
    train_idx = X_train.index
    df_train_raw = df.loc[train_idx]
    df_all_woe, woe_maps, iv_summary = woe_encode(df, cat_cols, TARGET)
    print("\nInformation Value (categorical features, higher = more predictive):")
    for k, v in sorted(iv_summary.items(), key=lambda x: -x[1]):
        print(f"    {k}: IV={v:.3f}")

    woe_cols = [c + "_woe" for c in cat_cols] + num_cols
    X_woe_all = df_all_woe[woe_cols]
    X_train_woe, X_test_woe = X_woe_all.loc[X_train.index], X_woe_all.loc[X_test.index]

    all_metrics = []
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    # ---- 1) Logistic Regression baseline ----
    print("\n[1/3] Training Logistic Regression (WOE scorecard baseline)...")
    lr_model, lr_scaler, lr_metrics = train_logistic_baseline(X_train_woe, y_train, X_test_woe, y_test)
    all_metrics.append(lr_metrics)

    if fast:
        # ---- 2) LightGBM, sensible fixed defaults, no tuning ----
        print("\n[2/3] Training LightGBM (fast mode — fixed defaults, no tuning)...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=31, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=42, verbosity=-1,
        )
        lgb_model.fit(X_train, y_train)
        lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
        lgb_metrics = evaluate("LightGBM (default params)", y_test, lgb_prob)
        all_metrics.append(lgb_metrics)

        # ---- 3) XGBoost, sensible fixed defaults, no tuning ----
        print("\n[3/3] Training XGBoost (fast mode — fixed defaults, no tuning)...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=42, tree_method="hist",
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate("XGBoost (default params)", y_test, xgb_prob)
        all_metrics.append(xgb_metrics)
    else:
        # ---- 2) LightGBM (Optuna tuned) ----
        print(f"\n[2/3] Tuning LightGBM with Optuna ({n_trials} trials)...")
        lgb_params = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials)
        lgb_params.update({"objective": "binary", "random_state": 42, "scale_pos_weight": spw})
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)
        lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
        lgb_metrics = evaluate("LightGBM (Optuna-tuned)", y_test, lgb_prob)
        all_metrics.append(lgb_metrics)

        # ---- 3) XGBoost (Optuna tuned) ----
        print(f"\n[3/3] Tuning XGBoost with Optuna ({n_trials} trials)...")
        xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials)
        xgb_params.update({"objective": "binary:logistic", "random_state": 42,
                            "tree_method": "hist", "scale_pos_weight": spw})
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate("XGBoost (Optuna-tuned)", y_test, xgb_prob)
        all_metrics.append(xgb_metrics)

    # ---- pick winner by AUROC ----
    best = max(all_metrics, key=lambda m: m["auroc"])
    print(f"\nBest model: {best['model']} (AUROC={best['auroc']})")

    if best["model"].startswith("LightGBM"):
        best_model, model_type = lgb_model, "lightgbm"
    elif best["model"].startswith("XGBoost"):
        best_model, model_type = xgb_model, "xgboost"
    else:
        best_model, model_type = lr_model, "logistic_regression"

    # ---- SHAP explainability on the winning tree model (fallback to LGBM if LR wins,
    # since SHAP TreeExplainer needs a tree model for fast, exact explanations) ----
    shap_model = best_model if model_type != "logistic_regression" else lgb_model
    print("\nComputing SHAP values on a 2,000-row sample...")
    sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.Series(mean_abs_shap, index=sample.columns).sort_values(ascending=False).head(15)
    print("\nTop 15 features by mean |SHAP value|:")
    for feat, val in top_features.items():
        print(f"    {feat}: {val:.4f}")

    # ---- probability calibration ----
    # Gradient-boosted trees are good at *ranking* risk (AUROC/Gini/KS) but their
    # raw predict_proba output is often poorly calibrated -- a predicted 20% PD
    # doesn't necessarily behave like a true 20% observed default rate. We
    # calibrate on the VALIDATION set only (never the test set), so the Brier
    # score we report afterward stays an honest, untouched measure of quality.
    print("\nCalibrating predicted probabilities...")
    if model_type == "logistic_regression":
        calibration_method = "none"
        prob_before = lr_model.predict_proba(lr_scaler.transform(X_test_woe))[:, 1]
        brier_before = float(brier_score_loss(y_test, prob_before))
        brier_after = brier_before
        scoring_model = lr_model
        final_test_probs = prob_before
        print("  Logistic Regression already outputs a sigmoid-based probability, "
              "which is naturally well-calibrated -- skipping explicit calibration.")
    else:
        calibration_method = "isotonic"
        uncalibrated_prob = best_model.predict_proba(X_test)[:, 1]
        brier_before = float(brier_score_loss(y_test, uncalibrated_prob))
        calibrated = CalibratedClassifierCV(best_model, method=calibration_method, cv="prefit")
        calibrated.fit(X_val, y_val)  # calibrator is fit on validation data only
        calibrated_prob = calibrated.predict_proba(X_test)[:, 1]
        brier_after = float(brier_score_loss(y_test, calibrated_prob))
        scoring_model = calibrated
        final_test_probs = calibrated_prob
        print(f"  Method: {calibration_method}")
        print(f"  Brier score before calibration: {brier_before:.4f}")
        print(f"  Brier score after  calibration: {brier_after:.4f}")

    # ---- risk-band historical summary ----
    # Buckets the (calibrated) test-set probabilities into the same 5 risk grades
    # shown in the dashboard, then reports the ACTUAL observed default rate inside
    # each bucket -- this is what lets the dashboard say "customers in this band
    # historically defaulted ~9% of the time" instead of just showing a raw score.
    print("\nComputing historical risk-band summary (observed default rate per band)...")
    y_test_arr = y_test.to_numpy()
    band_labels = np.array([grade_from_pd(p) for p in final_test_probs])
    risk_band_summary = {}
    for _, label in RISK_BANDS:
        mask = band_labels == label
        n = int(mask.sum())
        risk_band_summary[label] = {
            "observed_default_rate": round(float(y_test_arr[mask].mean()), 4) if n > 0 else None,
            "n": n,
        }
        print(f"    {label}: observed default rate = {risk_band_summary[label]['observed_default_rate']}, n = {n}")

    # ---- feature validation ranges ----
    # Min/max of every raw input field as seen in TRAINING data only, so the
    # dashboard/API can flag when a live input falls outside the range the
    # model actually learned from (extrapolation risk).
    print("\nComputing feature validation ranges (min/max seen during training)...")
    raw_input_cols = [
        "loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti",
        "delinq_2yrs", "earliest_cr_line_years", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
        "pub_rec_bankruptcies",
    ]
    feature_ranges = {
        c: {"min": float(df_train_raw[c].min()), "max": float(df_train_raw[c].max())}
        for c in raw_input_cols if c in df_train_raw.columns
    }

    # ---- persist artifacts ----
    joblib.dump(scoring_model, ARTIFACT_DIR / "best_model.pkl")
    joblib.dump(feature_names, ARTIFACT_DIR / "feature_names.pkl")
    joblib.dump({
        "model_type": model_type, "cat_cols": cat_cols, "num_cols": num_cols,
        "woe_cols": woe_cols,
        "calibration_method": calibration_method,
        "brier_before_calibration": round(brier_before, 4),
        "brier_after_calibration": round(brier_after, 4),
        "feature_ranges": feature_ranges,
        "risk_band_summary": risk_band_summary,
        "approval_threshold": APPROVAL_THRESHOLD,
        "decline_threshold": DECLINE_THRESHOLD,
    }, ARTIFACT_DIR / "model_meta.pkl")
    joblib.dump(shap_model, ARTIFACT_DIR / "shap_model.pkl")
    # needed only if the LR/WOE scorecard turns out to be the champion model
    joblib.dump(woe_maps, ARTIFACT_DIR / "woe_maps.pkl")
    joblib.dump(lr_scaler, ARTIFACT_DIR / "lr_scaler.pkl")

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump({"all_models": all_metrics, "best_model": best, "iv_summary": iv_summary,
                   "risk_band_summary": risk_band_summary,
                   "calibration": {"method": calibration_method, "brier_before": brier_before,
                                    "brier_after": brier_after}}, f, indent=2)

    top_features.to_csv(ARTIFACT_DIR / "top_shap_features.csv")

    print(f"\nArtifacts saved to {ARTIFACT_DIR}/")
    print("  - best_model.pkl, feature_names.pkl, model_meta.pkl, shap_model.pkl")
    print("  - metrics.json, top_shap_features.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/loan_data.csv")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per model")
    parser.add_argument("--fast", action="store_true",
                         help="skip Optuna tuning, train LightGBM/XGBoost with fixed default params (runs in seconds/minutes)")
    args = parser.parse_args()
    main(args.data, args.trials, args.fast)
