"""
Shared, dataset-agnostic helpers used by both EDA.ipynb (conceptually) and
Augmentation.ipynb. Factored out so the "one transferable pipeline for both
datasets" requirement is enforced by code reuse, not by convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
LABEL_COL = "is_anomaly"


def split_feature_types(df: pd.DataFrame, label_col: str = LABEL_COL) -> dict:
    """Continuous vs. binary vs. categorical split (same rule as the starter kit)."""
    feature_cols = [c for c in df.columns if c != label_col]
    categorical_cols = [
        c for c in feature_cols
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    binary_cols = [
        c for c in numeric_cols
        if df[c].dropna().nunique() <= 2 and set(df[c].dropna().unique()).issubset({0, 1})
    ]
    continuous_cols = [c for c in numeric_cols if c not in binary_cols]
    return {
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
        "continuous_cols": continuous_cols,
    }


def bucket_rare_categories(series: pd.Series, min_freq: float = 0.01, other_label: str = "Other") -> pd.Series:
    """Fold long-tail categories below min_freq into a single bucket.

    Keeps a generator from having to learn (and later reproduce) categories it
    only saw a handful of times, which is what the EDA flagged for high-
    cardinality columns such as NSL-KDD's `service` and UNSW-NB15's `proto`.
    """
    freq = series.value_counts(normalize=True)
    rare = freq[freq < min_freq].index
    if len(rare) == 0:
        return series.copy()
    return series.where(~series.isin(rare), other_label)


def scrub_normal(df: pd.DataFrame, continuous_cols: list, label_col: str = LABEL_COL,
                  contamination: float = 0.02, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Isolate the normal (label==0) rows, drop IsolationForest within-class
    outliers, then drop exact duplicates. Mirrors EDA.ipynb Section 1.4/2.4 so
    the generator is trained on the same "clean normal" definition the EDA
    validated, not the raw normal subset.
    """
    normal_df = df[df[label_col] == 0].copy()
    X = normal_df[continuous_cols].fillna(0).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                           random_state=random_state, n_jobs=-1)
    pred = iso.fit_predict(Xs)
    normal_df["_iso_outlier"] = pred == -1
    scrubbed = (
        normal_df.loc[~normal_df["_iso_outlier"], normal_df.columns.drop("_iso_outlier")]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return scrubbed


def nearest_pd_correlation(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project a noisy/ill-conditioned correlation matrix onto the nearest
    valid (positive-definite, unit-diagonal) correlation matrix by clipping
    negative eigenvalues. Needed because per-category-group correlation
    matrices, estimated from a few hundred/thousand rows, are not guaranteed
    to be positive semi-definite.
    """
    corr = np.nan_to_num((corr + corr.T) / 2, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, eps, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def apply_numeric_constraints(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply the EDA-derived, machine-readable constraint spec (min/max clip,
    or snap-to-nearest-allowed-value) to a synthetic dataframe. `spec` is the
    dict loaded from `eda_outputs/<slug>_constraints.json`.
    """
    df = df.copy()
    for rule_name, cols_spec in spec.items():
        if rule_name.startswith("drop_columns") or not isinstance(cols_spec, dict):
            continue
        for col, bounds in cols_spec.items():
            if col not in df.columns:
                continue
            if "allowed" in bounds:
                try:
                    allowed = np.array(bounds["allowed"], dtype=float)
                except (TypeError, ValueError):
                    continue  # non-numeric allowed set (e.g. protocol_type strings) handled separately
                vals = df[col].to_numpy(dtype=float)
                nearest_idx = np.abs(vals[:, None] - allowed[None, :]).argmin(axis=1)
                df[col] = allowed[nearest_idx]
            else:
                lo = bounds.get("min", -np.inf)
                hi = bounds.get("max", np.inf)
                df[col] = df[col].clip(lo, hi)
    return df
