"""
Milestone 2 Dual-Track Benchmark & Official Submission Pipeline.
Evaluates Hybrid CSPC + DD-ME against prior benchmarks and generates official Kaggle submissions.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Tuple, Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure matplotlib temporary directory is set safely
os.environ["MPLCONFIGDIR"] = "/tmp"

sys.path.append(os.path.dirname(__file__))
from data_loader import load_dataset, classify_features, SUBMISSION_SPECS
from constraints import audit_constraints, enforce_nslkdd_constraints, enforce_unsw_constraints
from evaluator import build_feature_matrices, score_detectors, evaluate_pseudo_novelty
from generators.copula_slack import CopulaTailSlackGenerator
from generators.utility_copula import ContrastiveDVSCopulaGenerator
from generators.hybrid_cspc import HybridCSPCGenerator


def run_starter_kit_baseline(
    clean_normal_df: pd.DataFrame,
    continuous_cols: list,
    n_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Starter kit prototype sampling + Gaussian jitter baseline."""
    rng = np.random.default_rng(random_state)
    n_proto = min(20, len(clean_normal_df))
    proto_idx = rng.choice(len(clean_normal_df), size=n_proto, replace=False)
    prototypes = clean_normal_df.iloc[proto_idx]
    
    sample_idx = rng.choice(n_proto, size=n_samples, replace=True)
    syn_df = prototypes.iloc[sample_idx].copy().reset_index(drop=True)
    
    for c in continuous_cols:
        vals = syn_df[c].to_numpy(dtype=float)
        std = np.std(clean_normal_df[c].to_numpy(dtype=float))
        if std > 1e-6:
            noise = rng.normal(0.0, 0.5 * std, size=n_samples)
            syn_df[c] = np.maximum(0.0, vals + noise)
            
    return syn_df


def benchmark_dataset_m2(
    dataset_name: str,
    root_dir: str = ".",
    sample_size_eval: int = 15000,
) -> Tuple[pd.DataFrame, Any, dict]:
    """
    Run dual-track evaluation for Milestone 2 on a single dataset.
    Returns (results_df, best_generator, best_config).
    """
    print("=" * 80)
    print(f"MILESTONE 2 BENCHMARKING: {dataset_name.upper()}")
    print("=" * 80)

    df_raw = load_dataset(dataset_name, root_dir=root_dir)
    schema = classify_features(df_raw)

    # 80/20 train/holdout split
    df_train_raw, df_holdout = train_test_split(
        df_raw, test_size=0.20, random_state=42, stratify=df_raw["is_anomaly"]
    )
    df_train_norm = df_train_raw[df_train_raw["is_anomaly"] == 0].copy()

    # Define Candidate Models
    candidates = [
        {
            "name": "1. Starter Kit Baseline (Jitter 0.5*std)",
            "type": "baseline",
            "params": {},
        },
        {
            "name": "2. Milestone 1 Best (DVS / Copula Slack)",
            "type": "m1_best",
            "params": {
                "tau_utility": 0.15,
                "tau_low": 0.03 if dataset_name == "nslkdd" else 0.05,
                "alpha_dispersion": 8.0 if dataset_name == "nslkdd" else 35.0,
            },
        },
        {
            "name": "3. Milestone 2 Hybrid CSPC + DD-ME (Single-Bag B=1)",
            "type": "hybrid_cspc",
            "params": {
                "tau_utility": 0.15,
                "tau_low": 0.03 if dataset_name == "nslkdd" else 0.05,
                "alpha_dispersion": 8.0 if dataset_name == "nslkdd" else 35.0,
                "purge_trojan_states": False if dataset_name == "nslkdd" else True,
                "n_bags": 1,
            },
        },
        {
            "name": "4. Milestone 2 Hybrid CSPC + DD-ME (Multi-Seed Bagging B=5)",
            "type": "hybrid_cspc",
            "params": {
                "tau_utility": 0.15,
                "tau_low": 0.03 if dataset_name == "nslkdd" else 0.05,
                "alpha_dispersion": 8.0 if dataset_name == "nslkdd" else 35.0,
                "purge_trojan_states": False if dataset_name == "nslkdd" else True,
                "n_bags": 5,
            },
        },
    ]

    results = []
    trained_models = {}

    for cand in candidates:
        name = cand["name"]
        ctype = cand["type"]
        params = cand["params"]
        print(f"\n--- Evaluating Candidate: {name} ---")
        t0 = time.time()

        if ctype == "baseline":
            syn_df = run_starter_kit_baseline(
                df_train_norm, schema["continuous_cols"], n_samples=sample_size_eval, random_state=42
            )
            trained_models[name] = None
        elif ctype == "m1_best":
            if dataset_name == "nslkdd":
                model = CopulaTailSlackGenerator(random_state=42, tail_slack_gamma=1.05)
                model.fit(
                    df_train_norm,
                    numeric_cols=schema["numeric_cols"],
                    categorical_cols=schema["categorical_cols"],
                    continuous_cols=schema["continuous_cols"],
                    bounded_cols=[c for c in schema["continuous_cols"] if "rate" in c],
                    group_col="protocol_type",
                )
            else:
                model = ContrastiveDVSCopulaGenerator(random_state=42, **params)
                model.fit(df_train_raw, label_col="is_anomaly")
            syn_df = model.sample(sample_size_eval)
            trained_models[name] = model
        elif ctype == "hybrid_cspc":
            model = HybridCSPCGenerator(random_state=42, **params)
            model.fit(df_train_raw, label_col="is_anomaly", dataset_name=dataset_name)
            syn_df = model.sample(sample_size_eval)
            trained_models[name] = model

        # Enforce physical constraints
        if dataset_name == "nslkdd":
            syn_df = enforce_nslkdd_constraints(syn_df)
        else:
            syn_df = enforce_unsw_constraints(syn_df)

        gen_time = time.time() - t0

        # Track A: Standard proxy score (ECOD + 10 seeds x 500 trees IsolationForest)
        print("  Scoring Track A (ECOD + 10 seeds IForest)...")
        X_train, X_test, _ = build_feature_matrices(
            syn_df, df_holdout, schema["numeric_cols"], schema["categorical_cols"]
        )
        y_holdout = df_holdout["is_anomaly"].to_numpy()
        track_a_res = score_detectors(X_train, X_test, y_holdout, n_seeds=10, n_trees=500)

        # Track B: Pseudo-Novelty Stress-Testing
        print("  Scoring Track B (Pseudo-Novelty Attack Clusters)...")
        track_b_res = evaluate_pseudo_novelty(
            syn_df,
            df_holdout,
            schema["numeric_cols"],
            schema["categorical_cols"],
            schema["continuous_cols"],
            n_attack_clusters=4,
            random_state=42,
            X_train=X_train,
            X_test=X_test,
            base_res=track_a_res,
        )

        composite_s = (track_a_res["S"] + track_b_res["pseudo_novelty_S"]) / 2.0
        print(f"  > Track A S: {track_a_res['S']:.5f} (ECOD: {track_a_res['AUPRC_ECOD']:.5f}, IForest: {track_a_res['AUPRC_IForest']:.5f})")
        print(f"  > Track B S: {track_b_res['pseudo_novelty_S']:.5f} | Composite S: {composite_s:.5f} (Time: {gen_time:.1f}s)")

        results.append({
            "dataset": dataset_name,
            "candidate": name,
            "TrackA_ECOD": track_a_res["AUPRC_ECOD"],
            "TrackA_IForest": track_a_res["AUPRC_IForest"],
            "TrackA_S": track_a_res["S"],
            "TrackB_Pseudo_S": track_b_res["pseudo_novelty_S"],
            "Composite_S": composite_s,
            "time_sec": gen_time,
            "params": str(params),
        })

    res_df = pd.DataFrame(results).sort_values("Composite_S", ascending=False).reset_index(drop=True)
    best_candidate_name = res_df.iloc[0]["candidate"]
    best_model = trained_models[best_candidate_name]
    best_cfg = next(c for c in candidates if c["name"] == best_candidate_name)

    print(f"\nWinning Model for {dataset_name.upper()}: {best_candidate_name} (Composite S = {res_df.iloc[0]['Composite_S']:.5f})")
    return res_df, best_model, best_cfg


def generate_official_submission_m2(
    dataset_name: str,
    best_config: dict,
    output_dir: str = "expt-2/submissions",
    root_dir: str = ".",
    random_state: int = 42,
) -> str:
    """
    Fits winning Milestone 2 Hybrid CSPC model on full labeled dataset,
    samples target N records, enforces constraints, audits violations, and writes CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    spec = SUBMISSION_SPECS[dataset_name]
    target_n = spec["target_n"]
    out_filename = f"submission_{dataset_name}.csv"
    out_path = os.path.join(output_dir, out_filename)

    print("\n" + "#" * 80)
    print(f"GENERATING OFFICIAL MILESTONE 2 SUBMISSION FOR {dataset_name.upper()} ({target_n:,} rows)")
    print("#" * 80)

    # 1. Load full raw dataset
    df_raw = load_dataset(dataset_name, root_dir=root_dir)

    # 2. Fit Hybrid CSPC on full dataset
    if best_config.get("type") == "hybrid_cspc":
        params = best_config["params"]
    else:
        params = {
            "tau_utility": 0.15,
            "tau_low": 0.03 if dataset_name == "nslkdd" else 0.05,
            "alpha_dispersion": 8.0 if dataset_name == "nslkdd" else 35.0,
            "purge_trojan_states": False if dataset_name == "nslkdd" else True,
            "n_bags": 5,
        }
    print(f"Fitting HybridCSPCGenerator with parameters: {params}...")
    t0 = time.time()
    generator = HybridCSPCGenerator(random_state=random_state, **params)
    generator.fit(df_raw, label_col="is_anomaly", dataset_name=dataset_name)

    # 3. Sample target_n records
    print(f"Synthesizing {target_n:,} records across {generator.n_bags} bags...")
    syn_df = generator.sample(target_n)
    print(f"Synthesis completed in {time.time() - t0:.1f}s")

    # 4. Enforce deterministic constraints & physical invariants
    print("Enforcing physical domain constraints...")
    if dataset_name == "nslkdd":
        syn_df = enforce_nslkdd_constraints(syn_df)
    else:
        syn_df = enforce_unsw_constraints(syn_df)

    # 5. Column ordering & id creation
    original_feature_cols = [c for c in df_raw.columns if c != "is_anomaly"]
    syn_df = syn_df[original_feature_cols].copy()
    syn_df.insert(0, "id", np.arange(target_n, dtype=int))

    # 6. Audit submission
    print("Auditing generated submission batch...")
    audit = audit_constraints(syn_df, dataset_name)
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if len(audit) == 0:
        print("  -> Zero domain constraint violations detected!")

    assert len(syn_df) == target_n, f"Row count mismatch: {len(syn_df)} != {target_n}"
    assert not syn_df.isnull().any().any(), "Submission contains null/NaN values!"
    assert not np.isinf(syn_df.select_dtypes(include=[np.number]).to_numpy()).any(), "Submission contains Inf!"

    # 7. Write to expt-2/submissions/
    syn_df.to_csv(out_path, index=False)

    # 8. Sync to root submissions/ directory
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
    nsl_results, _, nsl_best_cfg = benchmark_dataset_m2(
        "nslkdd", root_dir=root_dir, sample_size_eval=15000
    )

    # 2. Benchmark UNSW-NB15
    unsw_results, _, unsw_best_cfg = benchmark_dataset_m2(
        "unsw", root_dir=root_dir, sample_size_eval=20000
    )

    # Combine & Export Benchmark Results
    all_results = pd.concat([nsl_results, unsw_results], axis=0, ignore_index=True)
    benchmark_csv_path = os.path.join(eda_outputs_dir, "benchmark_results_milestone2.csv")
    all_results.to_csv(benchmark_csv_path, index=False)
    print(f"\nBenchmark results saved to: {benchmark_csv_path}")

    # Print Summary Table
    print("\n" + "=" * 90)
    print("MILESTONE 2 DUAL-TRACK BENCHMARK SUMMARY")
    print("=" * 90)
    summary_cols = ["dataset", "candidate", "TrackA_S", "TrackB_Pseudo_S", "Composite_S", "time_sec"]
    print(all_results[summary_cols].to_string(index=False))
    print("=" * 90)

    # 3. Generate Official Submissions
    out_dir = os.path.join(root_dir, "expt-2", "submissions")
    nsl_sub_path = generate_official_submission_m2("nslkdd", nsl_best_cfg, output_dir=out_dir, root_dir=root_dir)
    unsw_sub_path = generate_official_submission_m2("unsw", unsw_best_cfg, output_dir=out_dir, root_dir=root_dir)

    print("\nALL MILESTONE 2 DELIVERABLES PRODUCED SUCCESSFULLY!")
    print(f"1. Benchmark Table: {benchmark_csv_path}")
    print(f"2. NSL-KDD Submission: {nsl_sub_path}")
    print(f"3. UNSW-NB15 Submission: {unsw_sub_path}")


if __name__ == "__main__":
    main()
