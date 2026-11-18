"""Streamlit dashboard - Customer Churn Prediction.   Run:  streamlit run app/app.py"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import eda
from src.config import DATA_PATH, FIGURES_DIR, MODEL_PATH, REPORTS_DIR
from src.predict import load_model, predict_customer, predict_dataframe
from src.preprocessing import (
    ENGINEERED_DESCRIPTIONS, RAW_FEATURES, clean_data, engineer_features, missing_value_report,
)
from src.train import train_and_compare

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")


# ------------------------------------------------------------------ helpers
def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource
def cached_model(path: str, mtime: float):
    return load_model(path)


def get_model():
    if not MODEL_PATH.exists():
        return None
    return cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def get_raw():
    """Uploaded dataframe, otherwise the default CSV in data/."""
    if "raw" in st.session_state:
        return st.session_state["raw"]
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def get_clean():
    raw = get_raw()
    if raw is None:
        return None
    if "clean" not in st.session_state or st.session_state.get("clean_src") is not raw:
        st.session_state["clean"] = clean_data(raw)
        st.session_state["clean_src"] = raw
    return st.session_state["clean"]


def need_data():
    st.warning("No dataset loaded. Upload the Telco CSV on the **Data Upload & Cleaning** page.")
    st.stop()


# ------------------------------------------------------------------ pages
def page_data():
    st.header("1. Data upload & cleaning")
    st.caption("FR-01 Upload customer dataset  |  FR-02 Clean missing values")
    up = st.file_uploader("Upload the IBM Telco Customer Churn CSV", type="csv")
    if up is not None:
        st.session_state["raw"] = pd.read_csv(up)
        st.session_state.pop("clean", None)
        st.success(f"Loaded {up.name}")
    raw = get_raw()
    if raw is None:
        need_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw.shape[0]:,}")
    c2.metric("Columns", raw.shape[1])
    c3.metric("Target column", "Churn" if "Churn" in raw.columns else "missing!")
    st.subheader("First 5 records")
    st.dataframe(raw.head())

    st.subheader("Missing values (before cleaning)")
    rep = missing_value_report(raw)
    st.dataframe(rep[rep["missing"] > 0] if (rep["missing"] > 0).any() else rep.head(0))
    if (rep["missing"] > 0).any():
        st.info("Blank strings count as missing. In this dataset `TotalCharges` is blank for brand-new customers "
                "(tenure = 0), so it is filled with tenure x MonthlyCharges = 0.")
    else:
        st.success("No missing values found.")

    clean = get_clean()
    st.subheader("After cleaning")
    st.write(f"Shape: **{clean.shape[0]:,} x {clean.shape[1]}** - `customerID` dropped, `TotalCharges` numeric, "
             "`Churn` encoded (Yes=1, No=0).")
    st.dataframe(clean.head())
    st.write("Remaining missing values:", int(clean.isna().sum().sum()))
    st.dataframe(eda.data_summary(clean))


def page_eda():
    st.header("2. Customer analytics (EDA)")
    st.caption("FR-03 Perform exploratory data analysis")
    df = get_clean()
    if df is None:
        need_data()
    if "Churn" not in df.columns:
        st.error("The dataset needs a Churn column for analytics.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['Churn'].mean():.1%}")
    c3.metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():.2f}")

    t1, t2, t3, t4 = st.tabs(["Churn & demographics", "Services vs churn", "Numeric features", "Key findings"])
    with t1:
        a, b = st.columns([1, 2])
        with a:
            show_fig(eda.plot_churn_distribution(df))
        with b:
            show_fig(eda.plot_demographics(df))
    with t2:
        show_fig(eda.plot_churn_rate_by(df, [
            "Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
            "PaperlessBilling", "StreamingTV", "MultipleLines", "SeniorCitizen"]))
        col = st.selectbox("Explore any column", [c for c in df.columns if df[c].nunique() <= 10 and c != "Churn"])
        st.bar_chart(eda.churn_rate_by(df, col))
    with t3:
        show_fig(eda.plot_numeric_distributions(df))
        show_fig(eda.plot_correlation(df))
    with t4:
        for f in eda.eda_findings(df):
            st.markdown(f"- {f}")


def page_features():
    st.header("3. Feature engineering")
    st.caption("FR-04 Perform feature engineering")
    df = get_clean()
    if df is None:
        need_data()
    st.write("These features are computed *inside* the model pipeline, so predictions for new customers "
             "use exactly the same logic as training.")
    st.table(pd.DataFrame({"feature": list(ENGINEERED_DESCRIPTIONS), "meaning": list(ENGINEERED_DESCRIPTIONS.values())}))
    eng = engineer_features(df[RAW_FEATURES])
    new_cols = list(ENGINEERED_DESCRIPTIONS)
    st.subheader("Preview")
    st.dataframe(eng[new_cols].assign(Churn=df["Churn"]).head(10))
    st.subheader("Churn rate by tenure group")
    st.bar_chart(eng.assign(Churn=df["Churn"]).groupby("tenure_group")["Churn"].mean().mul(100))


def page_train():
    st.header("4. Train & compare models")
    st.caption("FR-05 Train multiple models  |  FR-06 Compare performance  |  FR-10 Save the trained model")
    df = get_clean()
    if df is None:
        need_data()
    st.write("Trains Logistic Regression, Decision Tree, Random Forest and Gradient Boosting with 5-fold "
             "cross-validation, tunes each decision threshold, selects the best model on cross-validation "
             "(not on the test set) and saves it to `models/churn_model.pkl`.")
    if st.button("Train models", type="primary"):
        bar = st.progress(0.0)
        status = st.empty()

        def cb(frac, msg):
            bar.progress(min(float(frac), 1.0))
            status.text(msg)

        with st.spinner("Training... this takes about a minute"):
            summary = train_and_compare(df, save=True, progress=cb)
        st.success(f"Done. Selected model: **{summary['best_model']}** (threshold {summary['threshold']:.2f}). "
                   f"Saved to {MODEL_PATH.name}.")
        st.session_state["trained"] = True
    show_comparison()


def show_comparison():
    csv = REPORTS_DIR / "model_comparison.csv"
    meta = REPORTS_DIR / "metrics.json"
    if not csv.exists():
        st.info("No results yet - click **Train models**.")
        return
    res = pd.read_csv(csv)
    best = json.loads(meta.read_text())["best_model"] if meta.exists() else None
    cols = ["model", "cv_roc_auc", "cv_f1", "threshold", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_roc_auc"]
    st.subheader("Model comparison")
    st.dataframe(res[cols].round(3))
    if best:
        st.success(f"Selected model: **{best}**")
    img = FIGURES_DIR / "model_comparison.png"
    if img.exists():
        st.image(str(img))


def page_predict():
    st.header("5. Predict customer churn")
    st.caption("FR-07 Predict churn  |  FR-08 Display prediction probabilities")
    bundle = get_model()
    if bundle is None:
        st.warning("No trained model found. Go to **Train & Compare Models** first.")
        st.stop()
    st.caption(f"Model: {bundle['model_name']}  |  decision threshold: {bundle['threshold']:.2f}")

    tab1, tab2 = st.tabs(["Single customer", "Batch (CSV)"])
    with tab1:
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Profile**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with b:
            st.markdown("**Services**")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            lines = "No phone service" if phone == "No" else st.selectbox("Multiple lines", ["No", "Yes"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            addons = {}
            for name in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                addons[name] = "No internet service" if internet == "No" else st.selectbox(name, ["No", "Yes"])
        with c:
            st.markdown("**Billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                      "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 0.5)
            total = st.number_input("Total charges ($)", 0.0, 10000.0, float(round(tenure * monthly, 2)), 1.0,
                                    key=f"total_{tenure}_{monthly}")

        if st.button("Predict churn", type="primary"):
            customer = {
                "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
                "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
                "InternetService": internet, **addons, "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
            }
            r = predict_customer(customer, bundle)
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", r["label"])
            m2.metric("Churn probability", f"{r['churn_probability']:.1%}")
            m3.metric("Risk level", r["risk_level"])
            fig, ax = plt.subplots(figsize=(7, 1.3))
            ax.barh([""], [r["stay_probability"]], color="#2a9d8f", label="Stay")
            ax.barh([""], [r["churn_probability"]], left=[r["stay_probability"]], color="#e76f51", label="Churn")
            ax.axvline(1 - r["threshold"], color="k", ls="--")
            ax.set_xlim(0, 1)
            ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.4))
            ax.set_title("Probability of staying vs leaving (dashed line = decision threshold)", fontsize=9)
            show_fig(fig)
            if r["prediction"] == 1:
                st.error("Recommended action: contact this customer with a retention offer "
                         "(e.g. contract upgrade discount, free tech support).")
            else:
                st.success("Low risk - no action needed.")

    with tab2:
        up = st.file_uploader("CSV with the same columns as the Telco dataset", type="csv", key="batch")
        if up is not None:
            try:
                out = predict_dataframe(pd.read_csv(up), bundle)
            except Exception as e:
                st.error(f"Could not score file: {e}")
            else:
                st.write(f"Scored {len(out):,} customers - {int(out['churn_prediction'].sum()):,} flagged as likely to leave.")
                st.dataframe(out.sort_values("churn_probability", ascending=False).head(100))
                st.download_button("Download predictions", out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")


def page_performance():
    st.header("6. Model performance & reports")
    st.caption("FR-09 Generate visual reports")
    meta = REPORTS_DIR / "metrics.json"
    if not meta.exists():
        st.info("Train a model first.")
        st.stop()
    summary = json.loads(meta.read_text())
    res = pd.DataFrame(summary["results"])
    best = res[res["model"] == summary["best_model"]].iloc[0]
    st.subheader(f"Selected model: {summary['best_model']}")
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("Accuracy", "test_accuracy"), ("Precision", "test_precision"),
                                         ("Recall", "test_recall"), ("F1", "test_f1"), ("ROC-AUC", "test_roc_auc")]):
        col.metric(label, f"{best[key]:.3f}")
    st.caption(f"Threshold {summary['threshold']:.2f}. Selected by {summary['selection_criterion']}.")
    show_comparison()
    for title, f in [("ROC curves", "roc_curves.png"), ("Confusion matrices", "confusion_matrices.png"),
                     ("Feature importance", "feature_importance.png"),
                     ("Probability distribution", "probability_distribution.png")]:
        p = FIGURES_DIR / f
        if p.exists():
            st.subheader(title)
            st.image(str(p))
    rep = REPORTS_DIR / "model_report.md"
    if rep.exists():
        st.download_button("Download model report (Markdown)", rep.read_bytes(), "model_report.md")


PAGES = {
    "1. Data Upload & Cleaning": page_data,
    "2. Customer Analytics (EDA)": page_eda,
    "3. Feature Engineering": page_features,
    "4. Train & Compare Models": page_train,
    "5. Predict Churn": page_predict,
    "6. Model Performance & Reports": page_performance,
}

st.sidebar.title("📉 Churn Prediction")
choice = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.markdown("---")
st.sidebar.caption("Trained model: " + ("✅ found" if MODEL_PATH.exists() else "❌ not trained yet"))
PAGES[choice]()
