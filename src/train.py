"""Reproducible training pipeline (FR-05, FR-06, FR-10).

Run from the project root:
    python -m src.train
    python -m src.train --data path/to/file.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):                       # allow `python src/train.py` too
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src import evaluate as ev
from src.config import (
    BETA, CV_FOLDS, DATA_PATH, FIGURES_DIR, MODEL_PATH, RANDOM_STATE, REPORTS_DIR, TARGET, TEST_SIZE,
)
from src.preprocessing import build_pipeline, clean_data, load_data, split_data

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_models() -> dict:
    """The four candidate algorithms required by the brief."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                        subsample=0.8, random_state=RANDOM_STATE),
    }


def cross_validate_pipeline(pipeline, X: pd.DataFrame, y: pd.Series):
    """Stratified k-fold CV. Returns mean/std of fold metrics and out-of-fold probabilities."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    folds = []
    for tr, va in skf.split(X, y):
        model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = proba
        folds.append(ev.compute_metrics(y.iloc[va], proba, 0.5))
    stats = {}
    for k in METRIC_KEYS:
        vals = [f[k] for f in folds]
        stats[f"cv_{k}"] = float(np.mean(vals))
        stats[f"cv_{k}_std"] = float(np.std(vals))
    return stats, oof


def train_and_compare(df: pd.DataFrame, save: bool = True, progress=None) -> dict:
    """Clean dataframe in -> trained, evaluated and (optionally) saved model out.

    `progress` is an optional callback(fraction, message) used by the Streamlit app.
    """
    def report(frac, msg):
        if progress:
            progress(frac, msg)

    X_train, X_test, y_train, y_test = split_data(df)
    models = get_models()
    rows, fitted, test_proba = [], {}, {}

    for i, (name, model) in enumerate(models.items()):
        report(i / len(models), f"Cross-validating and training {name} ...")
        pipe = build_pipeline(model)
        cv_stats, oof = cross_validate_pipeline(pipe, X_train, y_train)
        threshold, cv_f2 = ev.best_threshold(y_train, oof)

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        tuned = ev.compute_metrics(y_test, proba, threshold)
        default = ev.compute_metrics(y_test, proba, 0.5)

        row = {"model": name, **cv_stats, "threshold": threshold, "cv_f2_tuned": cv_f2}
        row.update({f"test_{k}": v for k, v in tuned.items()})
        row.update({f"test_default_{k}": default[k] for k in ("accuracy", "precision", "recall", "f1")})
        rows.append(row)
        fitted[name], test_proba[name] = pipe, proba

    results = pd.DataFrame(rows)
    # Model selection uses ONLY cross-validation on training data (never the test set).
    best_name = results.loc[results["cv_f2_tuned"].idxmax(), "model"]
    best_pipe = fitted[best_name]
    best_row = results[results["model"] == best_name].iloc[0]

    report(0.9, "Computing feature importance and charts ...")
    perm = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc",
                                  n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    importance = (pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean,
                                "std": perm.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    summary = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best_name,
        "threshold": float(best_row["threshold"]),
        "beta": BETA,
        "selection_criterion": f"highest cross-validated F{BETA:g} at the tuned threshold",
        "dataset": {
            "n_rows": int(len(df)), "churn_rate": float(df[TARGET].mean()),
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "cv_folds": CV_FOLDS, "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        },
        "results": results.to_dict(orient="records"),
        "feature_importance": importance.to_dict(orient="records"),
        "sklearn_version": sklearn.__version__,
    }

    # ---- charts (FR-09) -------------------------------------------------------------
    figures = {
        "model_comparison.png": ev.plot_model_comparison(results),
        "roc_curves.png": ev.plot_roc_curves({n: (y_test, p) for n, p in test_proba.items()}),
        "confusion_matrices.png": ev.plot_confusion_matrices(
            {r["model"]: [[r["test_tn"], r["test_fp"]], [r["test_fn"], r["test_tp"]]] for r in rows}, best=best_name),
        "feature_importance.png": ev.plot_feature_importance(importance),
        "probability_distribution.png": ev.plot_probability_distribution(
            y_test, test_proba[best_name], summary["threshold"]),
    }

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": best_pipe,
            "model_name": best_name,
            "threshold": summary["threshold"],
            "metrics": {k: float(best_row[f"test_{k}"]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            "trained_at": summary["trained_at"],
            "sklearn_version": sklearn.__version__,
        }, MODEL_PATH)                                                       # FR-10
        results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
        (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for name, fig in figures.items():
            ev.save_figure(fig, name)
        ev.write_model_report(summary, REPORTS_DIR / "model_report.md")

    import matplotlib.pyplot as plt
    for fig in figures.values():
        plt.close(fig)
    report(1.0, "Done")
    return summary


def run_training(data_path: str | Path | None = None, save: bool = True, progress=None) -> dict:
    """Load -> clean -> train -> evaluate -> save."""
    df = clean_data(load_data(data_path or DATA_PATH))
    return train_and_compare(df, save=save, progress=progress)


def main():
    parser = argparse.ArgumentParser(description="Train the churn prediction models.")
    parser.add_argument("--data", default=None, help="Path to the Telco CSV (default: data/customer_data.csv)")
    args = parser.parse_args()

    summary = run_training(args.data, progress=lambda f, m: print(f"[{f:4.0%}] {m}"))
    res = pd.DataFrame(summary["results"])
    cols = ["model", "cv_roc_auc", "cv_f2_tuned", "threshold", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_roc_auc"]
    print("\n" + res[cols].round(3).to_string(index=False))
    print(f"\nSelected model : {summary['best_model']}  (threshold {summary['threshold']:.2f})")
    print(f"Selection rule : {summary['selection_criterion']}")
    print(f"Model saved to : {MODEL_PATH}")
    print(f"Report saved to: {REPORTS_DIR / 'model_report.md'}")


if __name__ == "__main__":
    main()
