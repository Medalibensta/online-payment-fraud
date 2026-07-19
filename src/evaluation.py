"""
Shared evaluation helpers.

For a 0.17 %-positive problem, accuracy and even ROC-AUC are misleading (a
model that predicts "never fraud" scores 99.83 % accuracy). We therefore lead
with **PR-AUC** (average precision) and report precision / recall / F1 at a
chosen operating threshold, plus the confusion matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def score_probabilities(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    label: str = "model",
) -> dict:
    """Return a dict of imbalance-aware metrics for one model's scores."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": label,
        "PR_AUC": average_precision_score(y_true, y_score),
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble scored rows into a sorted, display-ready dataframe."""
    df = pd.DataFrame(rows)
    ordered = ["model", "PR_AUC", "ROC_AUC", "precision", "recall", "f1",
               "threshold", "TP", "FP", "FN", "TN"]
    df = df[[c for c in ordered if c in df.columns]]
    return df.sort_values("PR_AUC", ascending=False).reset_index(drop=True)
