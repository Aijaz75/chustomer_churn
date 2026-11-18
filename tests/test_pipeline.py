import numpy as np
import pandas as pd
import pytest

from src.preprocessing import RAW_FEATURES, build_pipeline, clean_data, engineer_features
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "customerID": [f"id{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n), "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n), "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": rng.integers(0, 72, n), "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)"], n),
        "MonthlyCharges": rng.uniform(20, 110, n).round(2),
        "Churn": rng.choice(["Yes", "No"], n),
    })
    for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[c] = rng.choice(["Yes", "No", "No internet service"], n)
    df["TotalCharges"] = (df.tenure * df.MonthlyCharges).astype(str)
    df.loc[df.tenure == 0, "TotalCharges"] = " "          # the real file's blank values
    return df


def test_clean_fixes_blank_total_charges(raw):
    clean = clean_data(raw)
    assert clean["TotalCharges"].isna().sum() == 0
    assert "customerID" not in clean.columns
    assert set(clean["Churn"].unique()) <= {0, 1}


def test_engineered_features_present(raw):
    eng = engineer_features(clean_data(raw)[RAW_FEATURES])
    for col in ["tenure_group", "n_addon_services", "avg_monthly_spend", "auto_pay"]:
        assert col in eng.columns


def test_pipeline_outputs_valid_probabilities(raw):
    clean = clean_data(raw)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(clean[RAW_FEATURES], clean["Churn"])
    p = pipe.predict_proba(clean[RAW_FEATURES])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_pipeline_handles_unseen_category(raw):
    clean = clean_data(raw)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(clean[RAW_FEATURES], clean["Churn"])
    row = clean[RAW_FEATURES].head(1).copy()
    row["Contract"] = "Five year"
    assert pipe.predict_proba(row).shape == (1, 2)
