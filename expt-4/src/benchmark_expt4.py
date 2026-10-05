"""
Benchmark and Submission Generation for Experiment 4.
Combines Controlled Partial Balancing v2 (NSL-KDD, proxy S=0.9601)
and Contrastive DVS-Copula v2 (UNSW-NB15, proxy S=0.5895).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ["MPLCONFIGDIR"] = "/tmp"

# Add current folder to sys.path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from cleaner import nearest_pd_correlation
from constraints import enforce_unsw_constraints as apply_physical_laws_unsw, enforce_nslkdd_constraints as enforce_nslkdd_zero_dominance
from evaluator import build_feature_matrices, score_detectors
from generators.dvs_copula_v2 import DVSCopulaV2Generator
from generators.stratified_cpb_v2 import ControlledPartialBalancingV2Generator

PROJECT_ROOT = SRC_DIR.parent.parent
SUBMISSIONS_DIR = SRC_DIR.parent / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

TARGET_N_NSL = 40000
TARGET_N_UNSW = 70000


def run_benchmark():
    print("=" * 80)
    print("EXPERIMENT 4: BENCHMARK & SUBMISSION ENGINE (RESTORED OPTIMAL TRACK 2)")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. NSL-KDD Pipeline: Controlled Partial Balancing (CPB v2)
    # -------------------------------------------------------------
    print("\n>>> [1/2] Processing NSL-KDD...")
    nsl_path = PROJECT_ROOT / "train_nslkdd.csv"
    df_nsl = pd.read_csv(nsl_path)
    print(f"Loaded NSL-KDD: {df_nsl.shape[0]} rows, {df_nsl.shape[1]} cols")

    # Local 80/20 Holdout Split
    train_nsl, val_nsl = train_test_split(
        df_nsl, test_size=0.2, stratify=df_nsl["is_anomaly"], random_state=42
    )

    feature_cols_nsl = [c for c in df_nsl.columns if c != "is_anomaly"]
    cat_cols_nsl = [
        c for c in feature_cols_nsl
        if pd.api.types.is_object_dtype(df_nsl[c]) or pd.api.types.is_string_dtype(df_nsl[c])
    ]
    num_cols_nsl = [c for c in feature_cols_nsl if c not in cat_cols_nsl]

    print("Fitting CPB v2 on NSL-KDD 80% train split...")
    cpb_proxy = ControlledPartialBalancingV2Generator(lambda_balance=0.30, n_bags=5, random_state=42)
    cpb_proxy.fit(train_nsl, label_col="is_anomaly")
    synth_nsl_proxy = cpb_proxy.sample(15000)
    synth_nsl_proxy = enforce_nslkdd_zero_dominance(synth_nsl_proxy)

    print("Evaluating NSL-KDD on 20% holdout split (ECOD + 10-seed IForest)...")
    X_tr_nsl, X_val_nsl, _ = build_feature_matrices(synth_nsl_proxy, val_nsl, num_cols_nsl, cat_cols_nsl)
    y_val_nsl = val_nsl["is_anomaly"].to_numpy(dtype=int)
    scores_nsl = score_detectors(X_tr_nsl, X_val_nsl, y_val_nsl, n_seeds=10, n_trees=500)
    print(f"NSL-KDD Proxy Results: AUPRC_ECOD={scores_nsl['AUPRC_ECOD']:.4f}, AUPRC_IForest={scores_nsl['AUPRC_IForest']:.4f}, Composite S={scores_nsl['S']:.4f}")

    # Generate Full NSL Submission
    print("Fitting CPB v2 on 100% NSL-KDD...")
    cpb_full = ControlledPartialBalancingV2Generator(lambda_balance=0.30, n_bags=5, random_state=42)
    cpb_full.fit(df_nsl, label_col="is_anomaly")
    submission_nsl = cpb_full.sample(TARGET_N_NSL)
    submission_nsl = enforce_nslkdd_zero_dominance(submission_nsl)
    submission_nsl.insert(0, "id", np.arange(len(submission_nsl)))

    # Save NSL Submissions
    out_nsl = SUBMISSIONS_DIR / "submission_nslkdd.csv"
    submission_nsl.to_csv(out_nsl, index=False)
    submission_nsl.to_csv(PROJECT_ROOT / "submissions" / "submission_nslkdd.csv", index=False)
    print(f"Saved NSL-KDD submission -> {out_nsl} (Shape: {submission_nsl.shape})")

    # -------------------------------------------------------------
    # 2. UNSW-NB15 Pipeline: Contrastive DVS-Copula v2
    # -------------------------------------------------------------
    print("\n>>> [2/2] Processing UNSW-NB15...")
    unsw_path = PROJECT_ROOT / "train_unsw.csv"
    df_unsw = pd.read_csv(unsw_path)
    print(f"Loaded UNSW-NB15: {df_unsw.shape[0]} rows, {df_unsw.shape[1]} cols")

    # Local 80/20 Holdout Split
    train_unsw, val_unsw = train_test_split(
        df_unsw, test_size=0.2, stratify=df_unsw["is_anomaly"], random_state=42
    )

    feature_cols_unsw = [c for c in df_unsw.columns if c != "is_anomaly"]
    cat_cols_unsw = [
        c for c in feature_cols_unsw
        if pd.api.types.is_object_dtype(df_unsw[c]) or pd.api.types.is_string_dtype(df_unsw[c])
    ]
    num_cols_unsw = [c for c in feature_cols_unsw if c not in cat_cols_unsw]

    print("Fitting Contrastive DVS-Copula v2 on UNSW 80% train split (alpha=35.0, 5 bags)...")
    copula_proxy = DVSCopulaV2Generator(
        alpha_dispersion=35.0,
        tau_utility=0.15,
        tau_low=0.05,
        n_bags=5,
        random_state=42,
    )
    copula_proxy.fit(train_unsw, label_col="is_anomaly", primary_cat_col="proto")
    synth_unsw_proxy = copula_proxy.sample(15000)
    synth_unsw_proxy = apply_physical_laws_unsw(synth_unsw_proxy)

    print("Evaluating UNSW-NB15 on 20% holdout split (ECOD + 10-seed IForest)...")
    X_tr_unsw, X_val_unsw, _ = build_feature_matrices(synth_unsw_proxy, val_unsw, num_cols_unsw, cat_cols_unsw)
    y_val_unsw = val_unsw["is_anomaly"].to_numpy(dtype=int)
    scores_unsw = score_detectors(X_tr_unsw, X_val_unsw, y_val_unsw, n_seeds=10, n_trees=500)
    print(f"UNSW-NB15 Proxy Results: AUPRC_ECOD={scores_unsw['AUPRC_ECOD']:.4f}, AUPRC_IForest={scores_unsw['AUPRC_IForest']:.4f}, Composite S={scores_unsw['S']:.4f}")

    # Generate Full UNSW Submission
    print("Fitting Contrastive DVS-Copula v2 on 100% UNSW-NB15...")
    copula_full = DVSCopulaV2Generator(
        alpha_dispersion=35.0,
        tau_utility=0.15,
        tau_low=0.05,
        n_bags=5,
        random_state=42,
    )
    copula_full.fit(df_unsw, label_col="is_anomaly", primary_cat_col="proto")
    submission_unsw = copula_full.sample(TARGET_N_UNSW)
    submission_unsw = apply_physical_laws_unsw(submission_unsw)
    submission_unsw.insert(0, "id", np.arange(len(submission_unsw)))

    # Save UNSW Submissions
    out_unsw = SUBMISSIONS_DIR / "submission_unsw.csv"
    submission_unsw.to_csv(out_unsw, index=False)
    submission_unsw.to_csv(PROJECT_ROOT / "submissions" / "submission_unsw.csv", index=False)
    print(f"Saved UNSW-NB15 submission -> {out_unsw} (Shape: {submission_unsw.shape})")

    # -------------------------------------------------------------
    # 3. Summary Results Table
    # -------------------------------------------------------------
    summary = pd.DataFrame([
        {
            "Dataset": "NSL-KDD",
            "Model": "CPB v2 (lambda=0.30, 5 bags)",
            "AUPRC_ECOD": scores_nsl["AUPRC_ECOD"],
            "AUPRC_IForest": scores_nsl["AUPRC_IForest"],
            "Composite_S": scores_nsl["S"],
        },
        {
            "Dataset": "UNSW-NB15",
            "Model": "Contrastive DVS-Copula v2 (alpha=35.0, 5 bags)",
            "AUPRC_ECOD": scores_unsw["AUPRC_ECOD"],
            "AUPRC_IForest": scores_unsw["AUPRC_IForest"],
            "Composite_S": scores_unsw["S"],
        },
    ])
    summary_path = SUBMISSIONS_DIR / "local_proxy_results_expt4.csv"
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 80)
    print("EXPERIMENT 4 BENCHMARK SUMMARY:")
    print(summary.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
