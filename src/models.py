"""
Model comparison: supervised vs. unsupervised (brief step 3).

Supervised detectors learn the fraud/legit boundary from labels:
    * Logistic regression (balanced)  — linear baseline
    * Random forest      (balanced)   — non-linear, robust
    * XGBoost (scale_pos_weight)      — gradient boosting, usually the winner

Unsupervised detectors ignore the labels and flag statistical outliers, which
matters operationally because genuinely novel fraud patterns have no historical
labels:
    * Isolation Forest   — isolates anomalies with random partitioning

All models output a continuous score so they can be compared on the same
PR / ROC footing. For Isolation Forest the negative anomaly score is used as the
"fraud-ness" score.

Run:
    python src/models.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from xgboost import XGBClassifier

from data import RANDOM_STATE
from evaluation import results_table, score_probabilities
from preprocessing import get_splits

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def train_supervised(X_tr, y_tr) -> dict:
    """Fit the three supervised models and return them keyed by name."""
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    models = {
        "LogReg (balanced)": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=pos_weight, eval_metric="aucpr",
            tree_method="hist", random_state=RANDOM_STATE),
    }
    for name, m in models.items():
        print(f"[models] training {name}...")
        m.fit(X_tr, y_tr)
    return models


def train_isolation_forest(X_tr, y_tr, contamination: float) -> IsolationForest:
    """Fit Isolation Forest on the (mostly-legit) training data."""
    print("[models] training IsolationForest (unsupervised)...")
    iso = IsolationForest(
        n_estimators=300, contamination=contamination,
        random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_tr)
    return iso


def run(save: bool = True) -> pd.DataFrame:
    X_tr, X_te, y_tr, y_te, _ = get_splits()
    contamination = float(y_tr.mean())

    supervised = train_supervised(X_tr, y_tr)
    iso = train_isolation_forest(X_tr, y_tr, contamination)

    rows, pr_curves, roc_curves = [], {}, {}
    for name, m in supervised.items():
        proba = m.predict_proba(X_te)[:, 1]
        rows.append(score_probabilities(y_te, proba, 0.5, name))
        p, r, _ = precision_recall_curve(y_te, proba)
        pr_curves[name] = (r, p)
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_curves[name] = (fpr, tpr)

    # Isolation Forest: higher score = more anomalous. score_samples is the
    # opposite sign, so negate to get a "fraud-ness" score.
    iso_score = -iso.score_samples(X_te)
    rows.append(score_probabilities(
        y_te, iso_score, np.quantile(iso_score, 1 - contamination),
        "IsolationForest (unsup.)"))
    p, r, _ = precision_recall_curve(y_te, iso_score)
    pr_curves["IsolationForest (unsup.)"] = (r, p)
    fpr, tpr, _ = roc_curve(y_te, iso_score)
    roc_curves["IsolationForest (unsup.)"] = (fpr, tpr)

    table = results_table(rows)
    print("\n=== Supervised vs unsupervised (test set) ===")
    print(table.to_string(index=False))

    if save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for name, (r, p) in pr_curves.items():
            ap = table.loc[table.model == name, "PR_AUC"].iloc[0]
            axes[0].plot(r, p, label=f"{name} ({ap:.3f})")
        axes[0].set(xlabel="Recall", ylabel="Precision",
                    title="Précision–Rappel (PR-AUC)")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

        for name, (fpr, tpr) in roc_curves.items():
            auc = table.loc[table.model == name, "ROC_AUC"].iloc[0]
            axes[1].plot(fpr, tpr, label=f"{name} ({auc:.3f})")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[1].set(xlabel="FPR", ylabel="TPR", title="ROC (AUC)")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "model_comparison.png", dpi=130)
        plt.close()
        print(f"[models] saved figure + csv to {REPORT_DIR}")

    return table


if __name__ == "__main__":
    run()
