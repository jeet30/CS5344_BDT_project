"""
Data loading and schema classification utilities for Experiment 2.
Enforces the project brief requirement: 'Use one transferable pipeline for both datasets'.
"""
from __future__ import annotations

import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd

LABEL_COL = "is_anomaly"

# Official competition submission constraints
SUBMISSION_SPECS = {
    "nslkdd": {
        "raw_path": "train_nslkdd.csv",
        "target_n": 40000,
        "constant_cols": ["num_outbound_cmds", "is_host_login"],
    },
    "unsw": {
        "raw_path": "train_unsw.csv",
        "target_n": 70000,
        "constant_cols": [],
    },
}


def load_dataset(name: str, root_dir: str = ".") -> pd.DataFrame:
    """Load dataset by name ('nslkdd' or 'unsw')."""
    key = name.lower()
    if key not in SUBMISSION_SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(SUBMISSION_SPECS.keys())}")
    
    csv_path = os.path.join(root_dir, SUBMISSION_SPECS[key]["raw_path"])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Dataset at {csv_path} is missing target label column '{LABEL_COL}'")
    
    return df


def classify_features(df: pd.DataFrame, label_col: str = LABEL_COL) -> Dict[str, Any]:
    """
    Rigorously partition feature columns into categorical, continuous, integer, binary,
    and constant features.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    
    categorical_cols = [
        c for c in feature_cols
        if pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    
    binary_cols = []
    integer_cols = []
    continuous_cols = []
    
    for c in numeric_cols:
        if c in constant_cols:
            continue
        vals = df[c].dropna().to_numpy()
        unique_vals = np.unique(vals)
        if set(unique_vals).issubset({0, 1}):
            binary_cols.append(c)
        elif np.all(np.isclose(vals, np.round(vals))) and len(unique_vals) <= 100:
            integer_cols.append(c)
        else:
            continuous_cols.append(c)
            
    return {
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
        "integer_cols": integer_cols,
        "continuous_cols": continuous_cols,
        "constant_cols": constant_cols,
    }

