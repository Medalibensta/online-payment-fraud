"""
Preprocessing for the fraud dataset.

The V1..V28 features are already PCA components (roughly centred and scaled), so
the only variable that needs attention is the single original one:

    * Amount — heavily right-skewed → log1p, then standardised.

(The OpenML mirror of the ULB dataset drops the original `Time` column, so no
time-of-day features are derived.)

Scaling statistics are always fit on the TRAINING split only and applied to the
test split, to avoid leaking test information into the features.

Run:
    python src/preprocessing.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import RANDOM_STATE, TARGET_COL, load_raw

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED_DIR = DATA_DIR / "processed"

V_COLS = [f"V{i}" for i in range(1, 29)]


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the log-transformed transaction amount."""
    df = df.copy()
    df["LogAmount"] = np.log1p(df["Amount"])
    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Stratified train/test split + train-fit scaling of the non-PCA features.

    Returns X_train, X_test, y_train, y_test, feature_names. The V-columns are
    passed through untouched; LogAmount is standardised.
    """
    df = add_amount_features(df)
    scaled_cols = ["LogAmount"]
    feature_cols = V_COLS + scaled_cols
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train[scaled_cols] = scaler.fit_transform(X_train[scaled_cols])
    X_test[scaled_cols] = scaler.transform(X_test[scaled_cols])

    return X_train, X_test, y_train, y_test, feature_cols


def get_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Convenience loader: raw → features → split → scaled."""
    return split_and_scale(load_raw())


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, cols = get_splits()
    print(f"train: {X_tr.shape}  fraud={y_tr.mean()*100:.3f}%")
    print(f"test:  {X_te.shape}  fraud={y_te.mean()*100:.3f}%")
    print(f"{len(cols)} features: {cols}")
