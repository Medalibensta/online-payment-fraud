"""
SHAP interpretability for the champion XGBoost model (brief step 6).

Risk teams cannot act on a black-box score: an analyst reviewing a flagged
transaction needs to know WHY it was flagged. TreeSHAP gives exact, per-
transaction attributions for tree ensembles:

    * beeswarm summary — which PCA components drive fraud predictions globally,
      and in which direction;
    * bar plot         — mean |SHAP| global feature importance;
    * waterfall        — full decision breakdown for the highest-risk fraud in
      the test set (the "explain this alert" analyst view).

The V-features are anonymised PCA components, so interpretation is necessarily
structural ("component V14 low → fraud-like") rather than semantic — a real
deployment on raw features would name merchant category, hour, country, etc.

Run:
    python src/interpretability.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from cost_sensitive import fit_xgboost
from preprocessing import get_splits

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def run(sample_size: int = 5000, save: bool = True) -> None:
    X_tr, X_te, y_tr, y_te, _ = get_splits()
    model = fit_xgboost(X_tr, y_tr)

    # TreeSHAP is fast, but 71k rows is still needless — explain a sample that
    # keeps every fraud plus a random slice of legit transactions.
    rng = np.random.default_rng(42)
    fraud_idx = np.flatnonzero(y_te.to_numpy() == 1)
    legit_idx = rng.choice(np.flatnonzero(y_te.to_numpy() == 0),
                           size=sample_size - len(fraud_idx), replace=False)
    sub = np.sort(np.r_[fraud_idx, legit_idx])
    X_sub = X_te.iloc[sub]

    print(f"[shap] computing TreeSHAP values on {len(X_sub):,} test rows...")
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sub)

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure()
        shap.plots.beeswarm(explanation, max_display=12, show=False)
        plt.title("SHAP — impact des features sur le score de fraude")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=130,
                    bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.plots.bar(explanation, max_display=12, show=False)
        plt.title("SHAP — importance globale (moyenne |SHAP|)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_importance.png", dpi=130,
                    bbox_inches="tight")
        plt.close()

        # Waterfall for the highest-scored true fraud: the analyst view.
        proba_sub = model.predict_proba(X_sub)[:, 1]
        is_fraud_sub = y_te.iloc[sub].to_numpy() == 1
        top_fraud_local = int(np.argmax(np.where(is_fraud_sub, proba_sub, -1)))
        plt.figure()
        shap.plots.waterfall(explanation[top_fraud_local], max_display=12,
                             show=False)
        plt.title("SHAP — décomposition d'une alerte fraude (score max)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_waterfall_fraud.png", dpi=130,
                    bbox_inches="tight")
        plt.close()
        print(f"[shap] saved 3 figures to {FIG_DIR}")

    # Console: top-5 global features.
    mean_abs = np.abs(explanation.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:5]
    cols = X_sub.columns.to_numpy()
    print("[shap] top-5 features by mean |SHAP|:")
    for i in order:
        print(f"    {cols[i]:<10} {mean_abs[i]:.4f}")


if __name__ == "__main__":
    run()
