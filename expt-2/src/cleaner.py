"""
Normal data extraction, deduplication, and outlier scrubbing for Experiment 2.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

LABEL_COL = "is_anomaly"
DEFAULT_SEED = 42


def scrub_normal_data(
    df: pd.DataFrame,
    continuous_cols: List[str],
    label_col: str = LABEL_COL,
    contamination: float = 0.02,
    random_state: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract normal instances, detect and remove within-class outliers via Isolation Forest,
    and eliminate exact duplicate records.
    
    Returns:
        scrubbed_df: Cleaned dataframe ready for generative modeling
        audit_stats: Dictionary containing diagnostic counters
    """
    normal_df = df[df[label_col] == 0].copy()
    raw_count = len(normal_df)
    
    # 1. Feature matrix for continuous outlier detection
    X_cont = normal_df[continuous_cols].fillna(0).to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X_cont)
    
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    preds = iso.fit_predict(X_scaled)
    is_outlier = preds == -1
    num_outliers = int(np.sum(is_outlier))
    
    # Keep inliers
    normal_inliers = normal_df.loc[~is_outlier, [c for c in normal_df.columns if c != label_col]].copy()
    inlier_count = len(normal_inliers)
    
    # 2. Deduplication
    scrubbed_df = normal_inliers.drop_duplicates().reset_index(drop=True)
    num_duplicates = inlier_count - len(scrubbed_df)
    
    audit_stats = {
        "raw_normal_count": raw_count,
        "outliers_removed": num_outliers,
        "outlier_pct": float(num_outliers / raw_count * 100),
        "duplicates_removed": num_duplicates,
        "final_scrubbed_count": len(scrubbed_df),
        "contamination_param": contamination,
    }
    
    return scrubbed_df, audit_stats


def nearest_pd_correlation(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Project an empirical or shrunk correlation matrix onto the nearest valid
    positive-definite, unit-diagonal correlation matrix by spectral eigenvalue clipping.
    """
    local_corr = np.array(corr, copy=True, dtype=float)
    local_corr = np.nan_to_num((local_corr + local_corr.T) / 2.0, nan=0.0)
    np.fill_diagonal(local_corr, 1.0)
    vals, vecs = np.linalg.eigh(local_corr)
    vals = np.clip(vals, eps, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed

