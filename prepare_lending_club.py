"""
prepare_lending_club.py
---------------------------------
Converts the RAW Kaggle Lending Club file (2.2M rows, 151 columns) into the
clean schema credit_risk_model.py expects (23 columns, binary target).
"""
import argparse
import pandas as pd
import numpy as np

BAD_STATUSES = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}

KEEP_RAW_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "dti",
    "delinq_2yrs", "earliest_cr_line", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "application_type", "mort_acc", "pub_rec_bankruptcies", "loan_status",
    "issue_d",
]


def main(raw_path: str, out_path: str, sample: int = None):
    print(f"Reading {raw_path} ...")
    usecols = [c for c in KEEP_RAW_COLS]
    df = pd.read_csv(raw_path, usecols=lambda c: c in usecols, low_memory=False)
    print(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")

    df = df[df["loan_status"].isin(BAD_STATUSES | GOOD_STATUSES)].copy()
    df["loan_status_default"] = df["loan_status"].isin(BAD_STATUSES).astype(int)
    df = df.drop(columns=["loan_status"])
    print(f"After dropping in-progress loans: {len(df):,} rows, "
          f"default rate = {df['loan_status_default'].mean():.2%}")

    df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    df["issue_d_parsed"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line_parsed"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line_years"] = (
        (df["issue_d_parsed"] - df["earliest_cr_line_parsed"]).dt.days / 365.25
    ).clip(lower=0)
    df = df.drop(columns=["earliest_cr_line", "issue_d", "issue_d_parsed", "earliest_cr_line_parsed"])

    df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "", regex=False)
    df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
    df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "", regex=False)
    df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    df = df.dropna(subset=["annual_inc", "dti", "term"])

    if sample:
        df = df.sample(min(sample, len(df)), random_state=42)
        print(f"Sampled down to {len(df):,} rows")

    df.to_csv(out_path, index=False)
    print(f"Saved cleaned file to {out_path} — {len(df):,} rows, {df.shape[1]} columns")
    print(f"Final default rate: {df['loan_status_default'].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=str,
        default=r"C:\Users\tripa\Downloads\accepted_2007_to_2018Q4.csv",
        help="path to the raw Kaggle CSV",
    )
    parser.add_argument("--out", type=str, default="data/loan_data.csv")
    parser.add_argument("--sample", type=int, default=400000,
                         help="randomly sample down to N rows")
    args = parser.parse_args()
    main(args.raw, args.out, args.sample)