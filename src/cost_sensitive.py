"""
Cost-sensitive threshold selection (brief step 4).

A fraud model's probability threshold is a BUSINESS decision, not a statistical
one. This module makes the trade-off explicit with a simple cost model:

    * missed fraud (FN)  — the bank refunds the transaction: cost = the actual
                           amount of that transaction (amount-aware), floored at
                           a minimum charge-back handling fee;
    * false alarm (FP)   — an analyst manually reviews the flagged transaction
                           (and the customer may be briefly inconvenienced):
                           flat review cost per flagged transaction;
    * true positive (TP) — the fraud is stopped, but the review cost is still
                           paid.

Total cost(threshold) = Σ_FN amount_i  +  c_review × (#TP + #FP)

The threshold sweep shows that the cost-minimising operating point sits far
below the default 0.5 — the model should flag many more transactions than a
naive deployment would.

Run:
    python src/cost_sensitive.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from data import RANDOM_STATE, load_raw
from preprocessing import get_splits

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

REVIEW_COST = 5.0      # € — analyst review of one flagged transaction
MIN_FN_COST = 50.0     # € — floor for charge-back handling on missed fraud


def fit_xgboost(X_tr, y_tr) -> XGBClassifier:
    """Train the champion model from models.py with identical settings."""
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        tree_method="hist", random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    return model


def business_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
) -> dict:
    """Compute the euro cost of operating the model at a given threshold."""
    flagged = y_score >= threshold
    fn_mask = (~flagged) & (y_true == 1)
    fn_cost = float(np.maximum(amounts[fn_mask], MIN_FN_COST).sum())
    review_cost = float(REVIEW_COST * flagged.sum())
    return {
        "threshold": threshold,
        "missed_fraud_cost": fn_cost,
        "review_cost": review_cost,
        "total_cost": fn_cost + review_cost,
        "n_flagged": int(flagged.sum()),
        "n_missed": int(fn_mask.sum()),
    }


def sweep(y_true, y_score, amounts,
          thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Evaluate the business cost across a grid of thresholds."""
    if thresholds is None:
        thresholds = np.concatenate([
            np.linspace(0.001, 0.099, 50),      # fine grid at low thresholds
            np.linspace(0.10, 0.99, 90),
        ])
    return pd.DataFrame(
        [business_cost(y_true, y_score, amounts, float(t)) for t in thresholds]
    )


def run(save: bool = True) -> pd.DataFrame:
    X_tr, X_te, y_tr, y_te, _ = get_splits()
    model = fit_xgboost(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    # Recover the raw (unscaled) amounts for the SAME test rows via the index.
    raw = load_raw()
    amounts = raw.loc[X_te.index, "Amount"].to_numpy()

    grid = sweep(y_te.to_numpy(), proba, amounts)
    best = grid.loc[grid.total_cost.idxmin()]
    naive = business_cost(y_te.to_numpy(), proba, amounts, 0.5)
    no_model = float(np.maximum(amounts[y_te == 1], MIN_FN_COST).sum())

    print("\n=== Cost-sensitive threshold analysis (test set, euros) ===")
    print(f"no model at all      : {no_model:>12,.0f} EUR (all fraud missed)")
    print(f"default threshold 0.5: {naive['total_cost']:>12,.0f} EUR "
          f"({naive['n_missed']} missed, {naive['n_flagged']} flagged)")
    print(f"optimal threshold {best.threshold:.3f}: {best.total_cost:>12,.0f} EUR "
          f"({best.n_missed:.0f} missed, {best.n_flagged:.0f} flagged)")
    saving = naive["total_cost"] - best.total_cost
    print(f"saving vs 0.5        : {saving:>12,.0f} EUR "
          f"({saving / naive['total_cost'] * 100:.1f}% of model@0.5 cost)")

    if save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        grid.to_csv(REPORT_DIR / "cost_threshold_sweep.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(grid.threshold, grid.total_cost, lw=2, label="coût total")
        ax.plot(grid.threshold, grid.missed_fraud_cost, "--",
                label="fraude manquée (FN)")
        ax.plot(grid.threshold, grid.review_cost, ":",
                label="coût de revue (TP+FP)")
        ax.axvline(best.threshold, color="crimson", alpha=0.7,
                   label=f"seuil optimal = {best.threshold:.3f}")
        ax.axvline(0.5, color="gray", alpha=0.5, label="seuil naïf = 0.5")
        ax.set(xlabel="Seuil de décision", ylabel="Coût (EUR)",
               title="Coût métier vs seuil de décision (XGBoost)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cost_vs_threshold.png", dpi=130)
        plt.close()
        print(f"[cost] saved figure + csv to {REPORT_DIR}")

    return grid


if __name__ == "__main__":
    run()
