"""Data loading, cleaning (FR-02), feature engineering (FR-04) and the sklearn preprocessing pipeline.

The saved model is a single sklearn Pipeline that accepts RAW customer columns:
    engineer_features -> impute/scale/one-hot -> classifier
so the Streamlit app and `predict.py` never need to repeat any preprocessing by hand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import DATA_PATH, ID_COL, RANDOM_STATE, TARGET, TEST_SIZE

# --------------------------------------------------------------------------------------
# Column definitions
# --------------------------------------------------------------------------------------
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
RAW_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    *SERVICE_COLS, "Contract", "PaperlessBilling", "PaymentMethod",
]
RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_FEATURES = RAW_CATEGORICAL + RAW_NUMERIC

ENGINEERED_NUMERIC = [
    "n_addon_services", "has_internet", "avg_monthly_spend",
    "charge_delta", "auto_pay", "has_family",
]
ENGINEERED_CATEGORICAL = ["tenure_group"]

NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL

ENGINEERED_DESCRIPTIONS = {
    "tenure_group": "Tenure bucket (0-12, 13-24, 25-48, 49+ months). New customers churn far more.",
    "n_addon_services": "How many of the 6 add-on services (security, backup, protection, support, TV, movies) the customer has.",
    "has_internet": "1 if the customer has any internet service.",
    "avg_monthly_spend": "TotalCharges / tenure - the customer's historical average monthly bill.",
    "charge_delta": "MonthlyCharges - avg_monthly_spend. Positive = bill is higher than it used to be.",
    "auto_pay": "1 if the payment method is automatic (bank transfer / credit card).",
    "has_family": "1 if the customer has a partner or dependents.",
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV file (FR-01)."""
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the IBM Telco CSV at data/customer_data.csv "
            "or set the CHURN_DATA_PATH environment variable."
        )
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------
# Cleaning (FR-02)
# --------------------------------------------------------------------------------------
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count real NaNs *and* blank strings (the Telco file hides missing values as ' ')."""
    rows = []
    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        n_blank = 0
        if not pd.api.types.is_numeric_dtype(s):
            n_blank = int((s.dropna().astype(str).str.strip() == "").sum())
        total = n_null + n_blank
        rows.append({"column": col, "missing": total, "missing_pct": round(100 * total / max(len(df), 1), 2)})
    return pd.DataFrame(rows).sort_values("missing", ascending=False).reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe.

    * strips whitespace and turns blank strings into NaN
    * converts TotalCharges to numeric (it is stored as text in the raw file)
    * fills missing TotalCharges deterministically: tenure * MonthlyCharges
      (the 11 affected customers have tenure == 0, so the value is 0)
    * drops the customer id and duplicate customers
    * encodes the target: Yes -> 1 (churn), No -> 0 (stay)

    Any remaining gaps are imputed *inside* the sklearn pipeline (fit on training data only),
    which prevents data leakage.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col]
            df[col] = s.where(s.isna(), s.astype(str).str.strip())
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"TotalCharges", "tenure", "MonthlyCharges"} <= set(df.columns):
        df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=ID_COL).drop(columns=ID_COL)

    if TARGET in df.columns:
        mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
        df[TARGET] = df[TARGET].astype(str).str.strip().str.lower().map(mapping)
        if df[TARGET].isna().all():
            raise ValueError(f"Could not interpret the '{TARGET}' column (expected Yes/No).")
        df = df.dropna(subset=[TARGET])
        df[TARGET] = df[TARGET].astype(int)

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Feature engineering (FR-04) - stateless, so it can live inside the pipeline
# --------------------------------------------------------------------------------------
def _to_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add derived churn-relevant features to a dataframe of raw customer columns."""
    X = X.copy()
    X["SeniorCitizen"] = _to_flag(X["SeniorCitizen"])
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X["TotalCharges"] = X["TotalCharges"].fillna(X["tenure"] * X["MonthlyCharges"])

    X["tenure_group"] = pd.cut(
        X["tenure"], bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12 m", "13-24 m", "25-48 m", "49+ m"],
    ).astype(object)
    X["n_addon_services"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
    X["has_internet"] = (X["InternetService"] != "No").astype(int)
    X["avg_monthly_spend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
    X["charge_delta"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
    X["auto_pay"] = X["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)
    X["has_family"] = ((X["Partner"] == "Yes") | (X["Dependents"] == "Yes")).astype(int)
    return X


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_pipeline(model) -> Pipeline:
    """raw columns -> feature engineering -> preprocessing -> model."""
    return Pipeline([
        ("features", FunctionTransformer(engineer_features, validate=False)),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])


def split_data(df: pd.DataFrame):
    """Stratified train/test split of a cleaned dataframe."""
    X = df[RAW_FEATURES].reset_index(drop=True)
    y = df[TARGET].reset_index(drop=True)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
