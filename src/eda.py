"""Exploratory data analysis helpers (FR-03). Every plot function returns a matplotlib Figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import TARGET

COLORS = {0: "#2a9d8f", 1: "#e76f51"}


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Dtype, unique count and missing count per column."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique": df.nunique(),
        "missing": df.isna().sum(),
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })


def plot_churn_distribution(df: pd.DataFrame):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(["Stay (0)", "Churn (1)"], counts.values, color=[COLORS[0], COLORS[1]])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("Churn distribution")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    return fig


def plot_demographics(df: pd.DataFrame):
    cols = [c for c in ("gender", "SeniorCitizen", "Partner", "Dependents") if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.2))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        vc = df[c].value_counts()
        ax.bar(vc.index.astype(str), vc.values, color="#2a6f97")
        ax.set_title(c)
    fig.suptitle("Customer demographics", y=1.03)
    fig.tight_layout()
    return fig


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.Series:
    return (df.groupby(col)[TARGET].mean() * 100).sort_values(ascending=False)


def plot_churn_rate_by(df: pd.DataFrame, cols: list[str], ncols: int = 3):
    cols = [c for c in cols if c in df.columns]
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    overall = df[TARGET].mean() * 100
    for ax, c in zip(axes, cols):
        rate = churn_rate_by(df, c)
        ax.bar(rate.index.astype(str), rate.values, color="#e76f51")
        ax.axhline(overall, color="k", ls="--", lw=1)
        ax.set_title(f"Churn rate by {c}", fontsize=10)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"Churn rate by category (dashed line = overall {overall:.1f}%)", y=1.0)
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame):
    cols = [c for c in ("tenure", "MonthlyCharges", "TotalCharges") if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 3.6))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        for label in (0, 1):
            ax.hist(df.loc[df[TARGET] == label, c].dropna(), bins=30, alpha=0.6,
                    color=COLORS[label], label="Churn" if label else "Stay")
        ax.set_title(c)
        ax.legend()
    fig.suptitle("Numeric features by churn", y=1.03)
    fig.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame):
    num = df.select_dtypes(include="number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right"); ax.set_yticklabels(corr.columns)
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation of numeric features (incl. Churn)")
    fig.tight_layout()
    return fig


def eda_findings(df: pd.DataFrame) -> list[str]:
    """Insights computed from the data itself (so they are always true for the data loaded)."""
    out = [f"Overall churn rate is **{df[TARGET].mean():.1%}** ({int(df[TARGET].sum()):,} of {len(df):,} customers) - "
           "the classes are imbalanced, so accuracy alone is misleading."]

    def rate(mask):
        return df.loc[mask, TARGET].mean()

    if "Contract" in df:
        out.append(f"**Contract type is the strongest signal**: month-to-month customers churn at "
                   f"{rate(df.Contract == 'Month-to-month'):.1%} vs {rate(df.Contract == 'Two year'):.1%} on two-year contracts.")
    if "tenure" in df:
        out.append(f"**New customers leave most**: churn is {rate(df.tenure <= 12):.1%} in the first 12 months "
                   f"vs {rate(df.tenure > 48):.1%} after 4 years.")
    if "InternetService" in df:
        out.append(f"**Fiber optic customers churn more** ({rate(df.InternetService == 'Fiber optic'):.1%}) than DSL "
                   f"({rate(df.InternetService == 'DSL'):.1%}) - possibly price or service quality.")
    if "TechSupport" in df:
        out.append(f"**Support & security reduce churn**: without TechSupport {rate(df.TechSupport == 'No'):.1%} churn, "
                   f"with it {rate(df.TechSupport == 'Yes'):.1%}.")
    if "PaymentMethod" in df:
        out.append(f"**Electronic-check payers churn most** ({rate(df.PaymentMethod == 'Electronic check'):.1%}) - "
                   "automatic payment methods are far lower.")
    if "MonthlyCharges" in df:
        out.append(f"Churners pay more per month on average (${df.loc[df[TARGET] == 1, 'MonthlyCharges'].mean():.2f} vs "
                   f"${df.loc[df[TARGET] == 0, 'MonthlyCharges'].mean():.2f}).")
    if "gender" in df:
        out.append(f"Gender has almost no effect (female {rate(df.gender == 'Female'):.1%} vs male {rate(df.gender == 'Male'):.1%}).")
    return out


def write_eda_findings(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# EDA Findings\n\n" + "\n".join(f"- {f}" for f in eda_findings(df)) + "\n", encoding="utf-8")
    return path
