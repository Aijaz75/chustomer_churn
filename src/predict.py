"""Load the saved model and predict churn with probabilities (FR-07, FR-08)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.preprocessing import RAW_FEATURES, clean_data

HIGH_RISK_CUTOFF = 0.60


def load_model(path: str | Path | None = None) -> dict:
    path = Path(path) if path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def risk_level(p: float, threshold: float) -> str:
    """Low = predicted to stay; Medium/High = predicted to leave, graded by probability."""
    if p < threshold:
        return "Low"
    return "High" if p >= max(HIGH_RISK_CUTOFF, threshold) else "Medium"


def predict_dataframe(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Predict for many customers. Accepts raw Telco-style columns (Churn/customerID optional)."""
    bundle = bundle or load_model()
    data = clean_data(df.drop(columns=["Churn"], errors="ignore"))
    missing = [c for c in RAW_FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    proba = bundle["pipeline"].predict_proba(data[RAW_FEATURES])[:, 1]
    thr = bundle["threshold"]
    out = data[RAW_FEATURES].copy()
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"] = (proba >= thr).astype(int)          # 0 = stay, 1 = likely to leave
    out["risk_level"] = [risk_level(p, thr) for p in proba]
    return out


def predict_customer(customer: dict, bundle: dict | None = None) -> dict:
    """Predict for one customer given as a dict of raw columns."""
    bundle = bundle or load_model()
    row = predict_dataframe(pd.DataFrame([customer]), bundle).iloc[0]
    return {
        "churn_probability": float(row["churn_probability"]),
        "stay_probability": float(1 - row["churn_probability"]),
        "prediction": int(row["churn_prediction"]),
        "label": "Likely to leave" if row["churn_prediction"] == 1 else "Likely to stay",
        "risk_level": row["risk_level"],
        "threshold": float(bundle["threshold"]),
    }


EXAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.7, "TotalCharges": 191.4,
}


def main():
    parser = argparse.ArgumentParser(description="Predict churn.")
    parser.add_argument("--csv", help="CSV of customers to score (default: built-in example customer)")
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()
    if args.csv:
        result = predict_dataframe(pd.read_csv(args.csv))
        result.to_csv(args.out, index=False)
        print(result[["churn_probability", "churn_prediction", "risk_level"]].head(10))
        print(f"Saved {len(result)} predictions to {args.out}")
    else:
        print(predict_customer(EXAMPLE_CUSTOMER))


if __name__ == "__main__":
    main()
