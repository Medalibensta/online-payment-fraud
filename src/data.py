"""
Data acquisition for the online-payment fraud project.

Primary source is the ULB "Credit Card Fraud Detection" dataset (Dal Pozzolo
et al., 2015): 284 807 European card transactions over two days in September
2013, of which only 492 are fraudulent (0.172 %). The 28 features V1..V28 are
the result of a PCA transformation applied by the data owners to protect
confidentiality; only `Amount` is original. (The OpenML mirror used here drops
the original `Time` column, so this project works from V1..V28 + Amount.)

Because the raw file is ~150 MB it is NOT committed to git. This module fetches
it reproducibly from OpenML (data_id=1597, no account required) and caches it as
a compact parquet file under data/raw/. If the download is unavailable (offline
CI, no network), it falls back to a documented synthetic surrogate with the same
schema and imbalance so the whole pipeline still runs end-to-end.

Run:
    python src/data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PARQUET = RAW_DIR / "creditcard.parquet"
OPENML_ID = 1597  # ULB CreditCardFraudDetection on OpenML

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"


def _from_openml() -> pd.DataFrame:
    """Download the genuine ULB dataset from OpenML."""
    from sklearn.datasets import fetch_openml

    print(f"[data] downloading ULB dataset from OpenML (data_id={OPENML_ID})...")
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    # OpenML labels the target "Class" with string dtype {'0','1'}; normalise.
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    print(f"[data] downloaded {len(df):,} rows, "
          f"{df[TARGET_COL].mean() * 100:.3f}% fraud.")
    return df


def _synthetic(n: int = 284_807, fraud_rate: float = 0.00172) -> pd.DataFrame:
    """
    Reproducible fallback that mimics the ULB schema and imbalance.

    Legit and fraud transactions are drawn from distinct multivariate-normal
    clouds in the 28-dim PCA space (fraud shifted on a handful of components,
    exactly the structure that makes the real problem separable), plus a
    right-skewed Amount.
    """
    print("[data] OpenML unavailable — generating synthetic surrogate.")
    rng = np.random.default_rng(RANDOM_STATE)
    n_fraud = max(1, int(round(n * fraud_rate)))
    n_legit = n - n_fraud

    legit = rng.standard_normal((n_legit, 28))
    fraud = rng.standard_normal((n_fraud, 28))
    # Shift a subset of components for the fraud cloud (weak, overlapping signal).
    shift_cols = [1, 2, 3, 9, 10, 11, 13, 16]
    fraud[:, shift_cols] += rng.uniform(1.5, 3.0, size=len(shift_cols))

    X = np.vstack([legit, fraud])
    y = np.r_[np.zeros(n_legit, int), np.ones(n_fraud, int)]

    amount = np.r_[
        rng.lognormal(3.0, 1.2, n_legit),
        rng.lognormal(4.0, 1.3, n_fraud),  # fraud skews to larger amounts
    ]
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = amount
    df[TARGET_COL] = y
    return df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """
    Return the raw transactions dataframe, using the local parquet cache when
    present. Set force_download=True to refetch from OpenML.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PARQUET.exists() and not force_download:
        return pd.read_parquet(RAW_PARQUET)

    try:
        df = _from_openml()
    except Exception as exc:  # noqa: BLE001 - network/import errors → fallback
        print(f"[data] OpenML fetch failed ({exc!s}).")
        df = _synthetic()

    # Keep a stable column order.
    df = df[FEATURE_COLS + [TARGET_COL]]
    df.to_parquet(RAW_PARQUET, index=False)
    print(f"[data] cached -> {RAW_PARQUET.relative_to(DATA_DIR.parent)}")
    return df


if __name__ == "__main__":
    frame = load_raw(force_download=True)
    print(frame.shape)
    print(frame[TARGET_COL].value_counts())
    print(frame.head())
