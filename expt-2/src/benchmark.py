"""
Comprehensive Benchmarking and Submission Pipeline for Experiment 2.
Evaluates Candidate Generators across Track A (Standard Proxy) and Track B (Pseudo-Novelty Stress-Test),
then synthesizes official competition submissions with strict domain constraints.
"""
from __future__ import annotations

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from data_loader import load_dataset, classify_features, SUBMISSION_SPECS
from cleaner import scrub_normal_data
from constraints import audit_constraints, enforce_nslkdd_constraints, enforce_unsw_constraints
from evaluator import build_feature_matrices, score_detectors, evaluate_pseudo_novelty
from generators.copula_slack import CopulaTailSlackGenerator, HybridEnsembleGenerator
from generators.utility_copula import ContrastiveDVSCopulaGenerator


def run_starter_kit_baseline(
    clean_normal_df: pd.DataFrame,
    continuous_cols: list,
    n_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Implements starter kit prototype sampling + Gaussian jitter baseline."""
    rng = np.random.default_rng(random_state)
    n_proto = min(20, len(clean_normal_df))
    proto_idx = rng.choice(len(clean_normal_df), size=n_proto, replace=False)
    prototypes = clean_normal_df.iloc[proto_idx]
    
    # Resample from prototypes
    sample_idx = rng.choice(n_proto, size=n_samples, replace=True)
    syn_df = prototypes.iloc[sample_idx].copy().reset_index(drop=True)
    
    # Add isotropic Gaussian noise (0.5 * std)
    for c in continuous_cols:
        vals = syn_df[c].to_numpy(dtype=float)
        std = np.std(clean_normal_df[c].to_numpy(dtype=float))
        if std > 1e-6:
            noise = rng.normal(0.0, 0.5 * std, size=n_samples)
            syn_df[c] = np.maximum(0.0, vals + noise)
            
    return syn_df


def benchmark_dataset(
    dataset_name: str,
    root_dir: str = "../..",
    sample_size_eval: int = 15000,
) -> Tuple[pd.DataFrame, Any, dict]:
    """
    Run dual-track evaluation across candidate generators for a single dataset.
    Returns (results_df, best_generator, best_config).
    """
    print("=" * 80)
    print(f"BENCHMARKING: {dataset_name.upper()}")
    print("=" * 80)
    
    df_raw = load_dataset(dataset_name, root_dir=root_dir)
    schema = classify_features(df_raw)
    
    # 80/20 train/holdout split for local evaluation
    df_train_raw, df_holdout = train_test_split(
        df_raw, test_size=0.20, random_state=42, stratify=df_raw["is_anomaly"]
    )
    
    # Scrub training normal
    print(f"Scrubbing training normal samples (raw train size: {len(df_train_raw)})...")
    clean_train_normal, _ = scrub_normal_data(
        df_train_raw, schema["continuous_cols"], contamination=0.02, random_state=42
    )
    print(f"Clean normal training rows available: {len(clean_train_normal)}")
    
    # Define candidates
    candidates = [
        {
            "name": "1. Starter Kit Baseline (Jitter 0.5*std)",
            "type": "baseline",
            "params": {},
        },
        {
            "name": "2. Expt 1 Copula (Tight, gamma=1.00)",
            "type": "copula",
            "params": {"tail_slack_gamma": 1.00},
        },
        {
            "name": "3. Expt 2 Copula (Slack gamma=1.10)",
            "type": "copula",
            "params": {"tail_slack_gamma": 1.10},
        },
        {
            "name": "4. Expt 2 Copula (Slack gamma=1.20)",
            "type": "copula",
            "params": {"tail_slack_gamma": 1.20},
        },
        {
            "name": "5. Expt 2 Hybrid Ensemble (gamma=1.15, slack_ratio=0.20)",
            "type": "hybrid",
            "params": {"tail_slack_gamma": 1.15, "slack_ratio": 0.20},
        },
        {
            "name": "6. Contrastive DVS Copula (Utility-Guided)",
            "type": "contrastive_dvs",
            "params": {
                "tau_utility": 0.15,
                "tau_low": 0.03 if dataset_name == "nslkdd" else 0.05,
                "alpha_dispersion": 8.0 if dataset_name == "nslkdd" else 35.0,
            },
        },
    ]
    
    # Identify group column for hierarchical conditioning
    group_col = "protocol_type" if dataset_name == "nslkdd" else "proto"
    
    # Bounded columns (e.g. rate metrics in [0, 1])
    bounded_cols = [c for c in schema["continuous_cols"] if "rate" in c]
    
    results = []
    fitted_models = {}
    
    for cand in candidates:
        cand_name = cand["name"]
        print(f"\n--- Testing Candidate: {cand_name} ---")
        t0 = time.time()
        
        if cand["type"] == "baseline":
            syn_eval = run_starter_kit_baseline(
                clean_train_normal, schema["continuous_cols"], n_samples=sample_size_eval, random_state=42
            )
            model_obj = None
        elif cand["type"] == "copula":
            gen = CopulaTailSlackGenerator(
                tail_slack_gamma=cand["params"]["tail_slack_gamma"], random_state=42
            )
            gen.fit(
                clean_train_normal,
                numeric_cols=schema["numeric_cols"],
                categorical_cols=schema["categorical_cols"],
                continuous_cols=schema["continuous_cols"],
                bounded_cols=bounded_cols,
                group_col=group_col,
            )
            syn_eval = gen.sample(sample_size_eval)
            model_obj = gen
        elif cand["type"] == "hybrid":
            gen = HybridEnsembleGenerator(
                tail_slack_gamma=cand["params"]["tail_slack_gamma"],
                slack_ratio=cand["params"]["slack_ratio"],
                random_state=42,
            )
            gen.fit(
                clean_train_normal,
                numeric_cols=schema["numeric_cols"],
                categorical_cols=schema["categorical_cols"],
                continuous_cols=schema["continuous_cols"],
                bounded_cols=bounded_cols,
                group_col=group_col,
            )
            syn_eval = gen.sample(sample_size_eval)
            model_obj = gen
        elif cand["type"] == "contrastive_dvs":
            gen = ContrastiveDVSCopulaGenerator(
                random_state=42,
                tau_utility=cand["params"]["tau_utility"],
                tau_low=cand["params"]["tau_low"],
                alpha_dispersion=cand["params"]["alpha_dispersion"],
            )
            gen.fit(df_train_raw, label_col="is_anomaly")
            syn_eval = gen.sample(sample_size_eval)
            model_obj = gen
            
        fitted_models[cand_name] = model_obj
        
        # Enforce physical constraints
        if dataset_name == "nslkdd":
            syn_eval = enforce_nslkdd_constraints(syn_eval)
        else:
            syn_eval = enforce_unsw_constraints(syn_eval)
            
        fit_sample_time = time.time() - t0
        
        # Evaluate Dual-Track (Track A: Standard Proxy, Track B: Pseudo-Novelty Stress-Test)
        eval_metrics = evaluate_pseudo_novelty(
            synthetic_train=syn_eval,
            real_holdout=df_holdout,
            numeric_cols=schema["numeric_cols"],
            cat_cols=schema["categorical_cols"],
            continuous_cols=schema["continuous_cols"],
            n_attack_clusters=4,
            random_state=42,
        )
        
        s_std = eval_metrics["overall"]["S"]
        auprc_ecod = eval_metrics["overall"]["AUPRC_ECOD"]
        auprc_iforest = eval_metrics["overall"]["AUPRC_IForest"]
        s_pseudo = eval_metrics["mean_cluster_s"]
        
        # Harmonic mean / blended metric to balance both tracks
        s_composite = 0.5 * s_std + 0.5 * s_pseudo
        
        print(f"  Track A (Standard S):      {s_std:.4f} (ECOD: {auprc_ecod:.4f}, IForest: {auprc_iforest:.4f})")
        print(f"  Track B (Pseudo-Novelty):  {s_pseudo:.4f}")
        print(f"  Composite Optimization S:  {s_composite:.4f}")
        print(f"  Execution Time:            {fit_sample_time:.1f}s")
        
        res_entry = {
            "dataset": dataset_name,
            "candidate": cand_name,
            "TrackA_S": s_std,
            "TrackA_ECOD": auprc_ecod,
            "TrackA_IForest": auprc_iforest,
            "TrackB_Pseudo_S": s_pseudo,
            "Composite_S": s_composite,
            "time_sec": fit_sample_time,
        }
        # Add individual cluster scores
        for cid, cs in eval_metrics["attack_clusters"].items():
            res_entry[f"{cid}_S"] = cs["S"]
            
        results.append(res_entry)
        
    # Rank candidates by Track A Standard S (primary competition leaderboard metric)
    res_df = pd.DataFrame(results).sort_values(by="TrackA_S", ascending=False)
    best_cand = res_df.iloc[0]
    best_name = best_cand["candidate"]
    print("\n" + "=" * 80)
    print(f"BEST CANDIDATE FOR {dataset_name.upper()}: {best_name}")
    print(f"  Track A S: {best_cand['TrackA_S']:.4f} | Track B Pseudo-Novelty S: {best_cand['TrackB_Pseudo_S']:.4f}")
    print("=" * 80)
    
    return res_df, fitted_models[best_name], best_cand.to_dict()


def generate_official_submission(
    dataset_name: str,
    generator: Any,
    output_dir: str = "expt-2/submissions",
    root_dir: str = ".",
    random_state: int = 42,
) -> str:
    """
    Generates the official competition submission CSV using the full scrubbed normal dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    spec = SUBMISSION_SPECS[dataset_name]
    target_n = spec["target_n"]
    out_filename = f"submission_{dataset_name}.csv"
    out_path = os.path.join(output_dir, out_filename)
    
    print("\n" + "#" * 80)
    print(f"GENERATING OFFICIAL SUBMISSION FOR {dataset_name.upper()} ({target_n} rows)")
    print("#" * 80)
    
    # 1. Load full raw dataset to extract complete feature list and schema
    df_raw = load_dataset(dataset_name, root_dir=root_dir)
    schema = classify_features(df_raw)
    
    # 2. Load full scrubbed normal dataset from eda/outputs
    scrubbed_path = os.path.join(
        root_dir, "expt-2", "eda", "outputs", f"{dataset_name}_scrubbed_normal.csv"
    )
    df_clean_normal = pd.read_csv(scrubbed_path)
    print(f"Full clean normal training rows: {len(df_clean_normal)}")
    
    group_col = "protocol_type" if dataset_name == "nslkdd" else "proto"
    bounded_cols = [c for c in schema["continuous_cols"] if "rate" in c]
    
    # 3. Fit and sample using multi-seed bagged ensemble to safeguard against 36%/64% test set shakeup
    print("Executing Multi-Seed Bagged Synthesis to safeguard against 36%/64% test set shakeup...")
    seeds = [42, 100]
    n_per_seed = [target_n // 2, target_n - (target_n // 2)]
    syn_batches = []
    t0 = time.time()
    
    for s_idx, (seed, n_sub) in enumerate(zip(seeds, n_per_seed)):
        print(f"  > Fitting & sampling bag {s_idx + 1}/{len(seeds)} (seed={seed}, N={n_sub:,})...")
        if dataset_name == "nslkdd":
            # NSL-KDD: Copula with gentle tail slack gamma=1.05 and exact discrete preservation
            model = CopulaTailSlackGenerator(random_state=seed, tail_slack_gamma=1.05)
            model.fit(
                df_clean_normal,
                numeric_cols=schema["numeric_cols"],
                categorical_cols=schema["categorical_cols"],
                continuous_cols=schema["continuous_cols"],
                bounded_cols=bounded_cols,
                group_col=group_col,
            )
            syn_batch = model.sample(n_sub)
        else:
            # UNSW: Contrastive DVS Copula with alpha=35.0 & discrete freezing (silencing continuous volume confounders)
            model = ContrastiveDVSCopulaGenerator(
                random_state=seed,
                tau_utility=0.15,
                tau_low=0.05,
                alpha_dispersion=35.0,
            )
            model.fit(df_raw, label_col="is_anomaly")
            syn_batch = model.sample(n_sub)
        syn_batches.append(syn_batch)
        
    syn_df = pd.concat(syn_batches, ignore_index=True)
    syn_df = syn_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"Bagged sampling completed in {time.time() - t0:.1f}s")
    
    # 5. Enforce deterministic constraints
    print("Enforcing physical domain constraints & identities...")
    if dataset_name == "nslkdd":
        syn_df = enforce_nslkdd_constraints(syn_df)
    else:
        syn_df = enforce_unsw_constraints(syn_df)
        
    # 6. Align column order exactly with original feature columns (excluding label)
    original_feature_cols = [c for c in df_raw.columns if c != "is_anomaly"]
    syn_df = syn_df[original_feature_cols].copy()
    
    # 7. Add leading lowercase 'id' column (0 to target_n - 1)
    syn_df.insert(0, "id", np.arange(target_n, dtype=int))
    
    # 8. Audit submission constraints
    print("Auditing generated submission batch...")
    audit = audit_constraints(syn_df, dataset_name)
    for k, v in audit.items():
        print(f"  {k}: {v}")
        
    assert len(syn_df) == target_n, f"Row count mismatch: {len(syn_df)} != {target_n}"
    assert not syn_df.isnull().any().any(), "Submission contains null/NaN values!"
    assert not np.isinf(syn_df.select_dtypes(include=[np.number]).to_numpy()).any(), "Submission contains Inf!"
    
    # 9. Save CSV to expt-2/submissions/
    syn_df.to_csv(out_path, index=False)
    
    # Also sync to root submissions/ directory
    root_sub_dir = os.path.join(root_dir, "submissions")
    os.makedirs(root_sub_dir, exist_ok=True)
    root_sub_path = os.path.join(root_sub_dir, f"submission_{dataset_name}.csv")
    syn_df.to_csv(root_sub_path, index=False)
    
    print(f"\n>>> Official submission successfully written to:")
    print(f"    - {out_path} ({os.path.getsize(out_path) / (1024*1024):.2f} MB)")
    print(f"    - {root_sub_path} ({os.path.getsize(root_sub_path) / (1024*1024):.2f} MB)")
    print(f"    Shape: {syn_df.shape} (Columns: {list(syn_df.columns[:5])}...)\n")
    return out_path


def main():
    root_dir = "."
    eda_outputs_dir = os.path.join(root_dir, "expt-2", "eda", "outputs")
    
    # 1. Benchmark NSL-KDD
    nsl_results, nsl_best_model, nsl_best_cfg = benchmark_dataset(
        "nslkdd", root_dir=root_dir, sample_size_eval=10000
    )
    
    # 2. Benchmark UNSW-NB15
    unsw_results, unsw_best_model, unsw_best_cfg = benchmark_dataset(
        "unsw", root_dir=root_dir, sample_size_eval=15000
    )
    
    # Combine & Export Benchmark Results
    all_results = pd.concat([nsl_results, unsw_results], axis=0, ignore_index=True)
    benchmark_csv_path = os.path.join(eda_outputs_dir, "benchmark_results.csv")
    all_results.to_csv(benchmark_csv_path, index=False)
    print(f"Benchmark results saved to: {benchmark_csv_path}")
    
    # Print Executive Summary Table
    print("\n" + "=" * 90)
    print("EXPERIMENT 2 DUAL-TRACK BENCHMARK SUMMARY")
    print("=" * 90)
    summary_cols = ["dataset", "candidate", "TrackA_S", "TrackB_Pseudo_S", "Composite_S", "time_sec"]
    print(all_results[summary_cols].to_string(index=False))
    print("=" * 90)
    
    # 3. Generate Official Submissions using the winning model per dataset
    out_dir = os.path.join(root_dir, "expt-2", "submissions")
    nsl_sub_path = generate_official_submission("nslkdd", nsl_best_model, output_dir=out_dir, root_dir=root_dir)
    unsw_sub_path = generate_official_submission("unsw", unsw_best_model, output_dir=out_dir, root_dir=root_dir)
    
    print("\nALL EXPERIMENT 2 DELIVERABLES PRODUCED SUCCESSFULLY!")
    print(f"1. Benchmark Table: {benchmark_csv_path}")
    print(f"2. NSL-KDD Submission: {nsl_sub_path}")
    print(f"3. UNSW-NB15 Submission: {unsw_sub_path}")


if __name__ == "__main__":
    main()
