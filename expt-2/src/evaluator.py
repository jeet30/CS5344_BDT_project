"""
Evaluation engine for Experiment 2:
Track A: Standard proxy score (ECOD + 10 seeds x 500 trees IsolationForest).
Track B: Pseudo-novelty stress-testing (unsupervised attack clustering to test novel-anomaly generalization).
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from pyod.models.ecod import ECOD

# Force matplotlib temporary directory inside workspace to avoid permission/cache warnings
os.environ["MPLCONFIGDIR"] = "/tmp"

LABEL_COL = "is_anomaly"
N_IFOREST_SEEDS = 10
N_IFOREST_TREES = 500


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert dataframes into aligned scaled numeric + one-hot categorical matrices."""
    # Scale numerics based on train
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numeric_cols].fillna(0).to_numpy(dtype=float))
    X_test_num = scaler.transform(test_df[numeric_cols].fillna(0).to_numpy(dtype=float))
    
    # Categoricals
    train_cat_dummies = pd.get_dummies(train_df[cat_cols])
    ref_columns = train_cat_dummies.columns.tolist()
    
    X_train_cat = train_cat_dummies.to_numpy(dtype=float)
    X_test_cat = (
        pd.get_dummies(test_df[cat_cols])
        .reindex(columns=ref_columns, fill_value=0)
        .to_numpy(dtype=float)
    )
    
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])
    return X_train, X_test, ref_columns


def score_detectors(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_seeds: int = N_IFOREST_SEEDS,
    n_trees: int = N_IFOREST_TREES,
) -> Dict[str, Any]:
    """Train ECOD and 10-seed IsolationForest on X_train, evaluate AUPRC on X_test."""
    # 1. ECOD
    ecod = ECOD()
    ecod.fit(X_train)
    ecod_scores = ecod.decision_function(X_test)
    auprc_ecod = float(average_precision_score(y_test, ecod_scores))
    
    # 2. Isolation Forest ensemble
    iforest_seed_scores = []
    for seed in range(n_seeds):
        clf = IsolationForest(n_estimators=n_trees, random_state=seed, n_jobs=-1)
        clf.fit(X_train)
        # Note: score_samples returns opposite of anomaly score (higher = normal), so invert with negative
        pred_scores = -clf.score_samples(X_test)
        auprc_seed = float(average_precision_score(y_test, pred_scores))
        iforest_seed_scores.append(auprc_seed)
        
    auprc_iforest = float(np.mean(iforest_seed_scores))
    s_score = (auprc_ecod + auprc_iforest) / 2.0
    
    return {
        "AUPRC_ECOD": auprc_ecod,
        "AUPRC_IForest": auprc_iforest,
        "S": s_score,
        "IForest_seed_std": float(np.std(iforest_seed_scores)),
        "IForest_seed_scores": iforest_seed_scores,
    }


def cluster_abnormal_traffic(
    abnormal_df: pd.DataFrame,
    continuous_cols: List[str],
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """
    Cluster abnormal samples (y=1) in continuous feature space to discover
    latent attack categories (A_seen sub-types).
    """
    X = abnormal_df[continuous_cols].fillna(0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)
    return labels, km


def evaluate_pseudo_novelty(
    synthetic_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    continuous_cols: List[str],
    n_attack_clusters: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate synthetic data against both:
    1. Overall holdout (standard proxy)
    2. Per-attack-cluster AUPRC (measuring performance across distinct attack modes)
    """
    y_holdout = real_holdout[LABEL_COL].to_numpy()
    X_train, X_test, _ = build_feature_matrices(synthetic_train, real_holdout, numeric_cols, cat_cols)
    
    # Base evaluation
    base_res = score_detectors(X_train, X_test, y_holdout)
    
    # Cluster abnormal holdout
    abnormal_mask = y_holdout == 1
    abnormal_indices = np.where(abnormal_mask)[0]
    normal_indices = np.where(~abnormal_mask)[0]
    
    abnormal_df = real_holdout.iloc[abnormal_indices]
    cluster_labels, _ = cluster_abnormal_traffic(
        abnormal_df, continuous_cols, n_clusters=n_attack_clusters, random_state=random_state
    )
    
    # Per-cluster AUPRC: evaluate detector on (all normal + single attack cluster)
    ecod = ECOD()
    ecod.fit(X_train)
    ecod_scores = ecod.decision_function(X_test)
    
    clf = IsolationForest(n_estimators=300, random_state=random_state, n_jobs=-1)
    clf.fit(X_train)
    iforest_scores = -clf.score_samples(X_test)
    
    cluster_scores = {}
    for c_id in range(n_attack_clusters):
        c_sub_idx = abnormal_indices[cluster_labels == c_id]
        sub_eval_indices = np.concatenate([normal_indices, c_sub_idx])
        y_sub = y_holdout[sub_eval_indices]
        
        c_ecod = float(average_precision_score(y_sub, ecod_scores[sub_eval_indices]))
        c_ifo = float(average_precision_score(y_sub, iforest_scores[sub_eval_indices]))
        c_s = (c_ecod + c_ifo) / 2.0
        
        cluster_scores[f"cluster_{c_id}"] = {
            "size": int(len(c_sub_idx)),
            "pct_abnormal": float(len(c_sub_idx) / len(abnormal_indices) * 100),
            "AUPRC_ECOD": c_ecod,
            "AUPRC_IForest": c_ifo,
            "S": c_s,
        }
        
    mean_cluster_s = float(np.mean([cs["S"] for cs in cluster_scores.values()]))
    mean_cluster_ecod = float(np.mean([cs["AUPRC_ECOD"] for cs in cluster_scores.values()]))
    mean_cluster_ifo = float(np.mean([cs["AUPRC_IForest"] for cs in cluster_scores.values()]))
    
    return {
        "overall": base_res,
        "attack_clusters": cluster_scores,
        "mean_cluster_s": mean_cluster_s,
        "mean_cluster_auprc_ecod": mean_cluster_ecod,
        "mean_cluster_auprc_iforest": mean_cluster_ifo,
    }

