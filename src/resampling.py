"""
Comparison of class-imbalance strategies (brief step 2).

Holds the estimator fixed (a fast, linear logistic regression) and varies only
the way the 0.17 %-positive training set is rebalanced, so the effect of the
*strategy* is isolated from the effect of the model:

    1. none            — raw imbalance, default logistic regression
    2. class_weight    — cost-reweighting inside the loss ("balanced")
    3. undersample     — RandomUnderSampler of the majority class
    4. SMOTE           — synthetic minority oversampling
    5. SMOTE+Tomek     — oversample then clean borderline majority points

Every strategy is evaluated on the SAME untouched test split with PR-AUC as the
headline metric. Resampling is applied inside an imblearn Pipeline so it only
ever touches the training folds — never the test data.

Run:
    python src/resampling.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from data import RANDOM_STATE
from evaluation import results_table, score_probabilities
from preprocessing import get_splits

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def _logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)


def build_strategies() -> dict[str, ImbPipeline]:
    """Return one imblearn pipeline per rebalancing strategy."""
    return {
        "none": ImbPipeline([("clf", _logreg())]),
        "class_weight": ImbPipeline([
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "undersample": ImbPipeline([
            ("res", RandomUnderSampler(random_state=RANDOM_STATE)),
            ("clf", _logreg()),
        ]),
        "SMOTE": ImbPipeline([
            ("res", SMOTE(random_state=RANDOM_STATE)),
            ("clf", _logreg()),
        ]),
        "SMOTE+Tomek": ImbPipeline([
            ("res", SMOTETomek(random_state=RANDOM_STATE)),
            ("clf", _logreg()),
        ]),
    }


def run(save: bool = True) -> pd.DataFrame:
    X_tr, X_te, y_tr, y_te, _ = get_splits()
    rows, curves = [], {}
    for name, pipe in build_strategies().items():
        print(f"[resampling] fitting '{name}'...")
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        rows.append(score_probabilities(y_te, proba, threshold=0.5, label=name))
        prec, rec, _ = precision_recall_curve(y_te, proba)
        curves[name] = (rec, prec)

    table = results_table(rows)
    print("\n=== Resampling strategy comparison (test set) ===")
    print(table.to_string(index=False))

    if save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORT_DIR / "resampling_comparison.csv", index=False)

        plt.figure(figsize=(7, 6))
        for name, (rec, prec) in curves.items():
            ap = table.loc[table.model == name, "PR_AUC"].iloc[0]
            plt.plot(rec, prec, label=f"{name} (PR-AUC={ap:.3f})")
        plt.xlabel("Recall (fraudes détectées)")
        plt.ylabel("Precision")
        plt.title("Précision–Rappel selon la stratégie de rééquilibrage")
        plt.legend(loc="upper right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "resampling_pr_curves.png", dpi=130)
        plt.close()
        print(f"[resampling] saved figure + csv to {REPORT_DIR}")

    return table


if __name__ == "__main__":
    run()
