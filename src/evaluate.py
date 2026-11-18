"""Model evaluation: metrics, threshold tuning, charts (FR-06, FR-09) and the markdown report."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, fbeta_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
)

from src.config import BETA, FIGURES_DIR

LABELS = ["Stay", "Churn"]


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """All headline metrics for one set of predicted probabilities."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=BETA, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y_true, proba, beta: float = BETA) -> tuple[float, float]:
    """Decision threshold that maximises F-beta (recall-weighted) on the given predictions."""
    y_true = np.asarray(y_true)
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        score = fbeta_score(y_true, (proba >= t).astype(int), beta=beta, zero_division=0)
        if score > best_score:
            best_t, best_score = float(round(t, 2)), float(score)
    return best_t, best_score


# --------------------------------------------------------------------------------------
# Charts (all return a matplotlib Figure)
# --------------------------------------------------------------------------------------
def plot_model_comparison(results: pd.DataFrame):
    metrics = ["test_recall", "test_precision", "test_f1", "test_roc_auc"]
    names = ["Recall", "Precision", "F1", "ROC-AUC"]
    x = np.arange(len(results))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (m, n) in enumerate(zip(metrics, names)):
        ax.bar(x + (i - 1.5) * width, results[m], width, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels(results["model"], rotation=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Model comparison (tuned decision threshold)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_roc_curves(curves: dict):
    """curves: {model_name: (y_true, proba)}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(matrices: dict, best: str | None = None):
    """matrices: {model_name: 2x2 array [[tn, fp], [fn, tp]]}"""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, matrices.items()):
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(name + (" *" if name == best else ""), fontsize=10)
    fig.suptitle("Confusion matrices (test set, * = selected model)", y=1.02)
    fig.tight_layout()
    return fig


def plot_feature_importance(imp: pd.DataFrame, top: int = 12):
    imp = imp.sort_values("importance", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#2a6f97")
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("What drives churn predictions? (permutation importance)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_probability_distribution(y_true, proba, threshold: float):
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(proba[y_true == 0], bins=30, alpha=0.6, label="Stayed", color="#2a9d8f")
    ax.hist(proba[y_true == 1], bins=30, alpha=0.6, label="Churned", color="#e76f51")
    ax.axvline(threshold, color="k", ls="--", label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Customers")
    ax.set_title("Predicted probabilities by actual outcome (test set)")
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig, name: str, directory: Path = FIGURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------------------
def write_model_report(summary: dict, path: Path) -> Path:
    res = pd.DataFrame(summary["results"])
    best = summary["best_model"]
    b = res[res["model"] == best].iloc[0]
    ds = summary["dataset"]

    def row(r):
        return (f"| {r['model']} | {r['cv_roc_auc']:.3f} ± {r['cv_roc_auc_std']:.3f} | {r['cv_f2_tuned']:.3f} | "
                f"{r['threshold']:.2f} | {r['test_accuracy']:.3f} | {r['test_precision']:.3f} | "
                f"{r['test_recall']:.3f} | {r['test_f1']:.3f} | {r['test_roc_auc']:.3f} |")

    table = "\n".join(row(r) for _, r in res.iterrows())
    top = summary.get("feature_importance", [])[:8]
    top_lines = "\n".join(f"{i + 1}. **{f['feature']}** (ROC-AUC drop {f['importance']:.3f})" for i, f in enumerate(top))
    n_test = ds["n_test"]
    caught = b["test_tp"]
    churners = b["test_tp"] + b["test_fn"]

    md = f"""# Model Report - Customer Churn Prediction

_Generated automatically on {datetime.now():%Y-%m-%d %H:%M} by `src/train.py`._

## 1. Data
- Customers: **{ds['n_rows']:,}** | Churn rate: **{ds['churn_rate']:.1%}** (imbalanced: ~3 stayers per churner)
- Train / test split: {ds['n_train']:,} / {n_test:,} (stratified, `random_state={ds['random_state']}`)
- Cleaning: blank `TotalCharges` (new customers, tenure 0) converted to 0; `customerID` dropped; target encoded Yes=1/No=0.
- Feature engineering: tenure group, number of add-on services, average monthly spend, charge delta, auto-pay flag, family flag, internet flag.

## 2. Methodology
- One sklearn `Pipeline` (feature engineering -> imputation/scaling/one-hot -> model), so no information leaks from the test set.
- {ds['cv_folds']}-fold stratified cross-validation on the training data for every model.
- Because the classes are imbalanced, accuracy alone is misleading. The decision threshold for each model is tuned on
  **out-of-fold training predictions** to maximise **F{summary['beta']:g}** (recall weighted {summary['beta']:g}x more than precision).
- The model is **selected by cross-validated F{summary['beta']:g}, never by test results**; the test set is only used once for the final report.

**Why F{summary['beta']:g}?** Losing a customer costs far more than a retention discount sent to someone who would have stayed anyway,
so the business wants to *catch* most churners (high recall) while keeping precision at a workable level.

## 3. Results

| Model | CV ROC-AUC | CV F{summary['beta']:g} (tuned) | Threshold | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
{table}

![Model comparison](figures/model_comparison.png)
![ROC curves](figures/roc_curves.png)

## 4. Selected model: **{best}**
- Decision threshold: **{b['threshold']:.2f}** (a customer is flagged as "likely to leave" when P(churn) >= this value).
- On the held-out test set ({n_test:,} customers) it identifies **{int(caught)} of {int(churners)} churners ({b['test_recall']:.1%} recall)**
  with precision {b['test_precision']:.1%} (**{int(b['test_fp'])} loyal customers** flagged unnecessarily).
- ROC-AUC {b['test_roc_auc']:.3f}, accuracy {b['test_accuracy']:.1%}.

![Confusion matrices](figures/confusion_matrices.png)
![Probability distribution](figures/probability_distribution.png)

## 5. What drives churn?
{top_lines}

![Feature importance](figures/feature_importance.png)

## 6. Limitations & next steps
- Accuracy of ~80% is typical for this dataset; the data lacks usage/behavioural signals (calls, complaints, data usage), which limits any model.
- The threshold assumes a 2:1 cost trade-off. If the real cost of an offer vs. a lost customer is known, change `BETA` in `src/config.py`.
- Possible improvements: hyper-parameter search, probability calibration, SHAP explanations, monitoring for drift after deployment.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return path
