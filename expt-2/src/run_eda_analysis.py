"""
Comprehensive EDA computation and export runner for Experiment 2.
Generates statistical profiles, domain constraint JSONs, scrubbed normal datasets,
and latent anomaly cluster profiles.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataset, classify_features, SUBMISSION_SPECS
from cleaner import scrub_normal_data
from constraints import audit_constraints, enforce_nslkdd_constraints, enforce_unsw_constraints

os.environ["MPLCONFIGDIR"] = "/tmp"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "eda", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_feature_profiles(df: pd.DataFrame, schema: dict, dataset_name: str) -> pd.DataFrame:
    """Compute comprehensive statistical profile and normal-vs-abnormal divergence."""
    records = []
    normal_df = df[df["is_anomaly"] == 0]
    abnormal_df = df[df["is_anomaly"] == 1]
    
    num_cols = schema["continuous_cols"] + schema["integer_cols"] + schema["binary_cols"]
    
    for col in num_cols:
        all_vals = df[col].dropna().to_numpy(dtype=float)
        norm_vals = normal_df[col].dropna().to_numpy(dtype=float)
        abn_vals = abnormal_df[col].dropna().to_numpy(dtype=float)
        
        # KS-Test between normal and abnormal
        ks_stat, ks_pval = stats.ks_2samp(norm_vals, abn_vals)
        
        # Skew & Kurtosis on normal
        norm_skew = float(stats.skew(norm_vals)) if len(norm_vals) > 0 else 0.0
        norm_kurt = float(stats.kurtosis(norm_vals)) if len(norm_vals) > 0 else 0.0
        
        # Zero inflation in normal
        zero_pct = float(np.mean(norm_vals == 0.0) * 100.0) if len(norm_vals) > 0 else 0.0
        
        # Quantiles in normal
        q05, q25, q50, q75, q95 = np.percentile(norm_vals, [5, 25, 50, 75, 95])
        
        col_type = "binary" if col in schema["binary_cols"] else ("integer" if col in schema["integer_cols"] else "continuous")
        
        records.append({
            "dataset": dataset_name,
            "feature": col,
            "type": col_type,
            "norm_mean": float(np.mean(norm_vals)),
            "norm_std": float(np.std(norm_vals)),
            "norm_median": float(q50),
            "norm_iqr": float(q75 - q25),
            "norm_min": float(np.min(norm_vals)),
            "norm_max": float(np.max(norm_vals)),
            "norm_q05": float(q05),
            "norm_q95": float(q95),
            "norm_skewness": norm_skew,
            "norm_kurtosis": norm_kurt,
            "norm_zero_pct": zero_pct,
            "abn_mean": float(np.mean(abn_vals)),
            "abn_median": float(np.median(abn_vals)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "heavy_tailed": abs(norm_skew) > 3.0 or norm_kurt > 10.0,
        })
        
    return pd.DataFrame(records)


def analyze_anomaly_clusters(df: pd.DataFrame, schema: dict, dataset_name: str, n_clusters: int = 4) -> pd.DataFrame:
    """Cluster abnormal instances to discover distinct attack archetypes."""
    abnormal_df = df[df["is_anomaly"] == 1].copy()
    cont_cols = schema["continuous_cols"]
    
    X = abnormal_df[cont_cols].fillna(0).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(Xs)
    abnormal_df["cluster"] = cluster_labels
    
    profiles = []
    for c_id in range(n_clusters):
        c_df = abnormal_df[abnormal_df["cluster"] == c_id]
        row_dict = {
            "dataset": dataset_name,
            "cluster_id": c_id,
            "count": len(c_df),
            "pct_of_abnormal": round(len(c_df) / len(abnormal_df) * 100, 2),
        }
        
        # Summarize key behavioral characteristics
        if "nsl" in dataset_name.lower():
            row_dict["mean_duration"] = round(c_df["duration"].mean(), 2)
            row_dict["mean_src_bytes"] = round(c_df["src_bytes"].mean(), 2)
            row_dict["mean_dst_bytes"] = round(c_df["dst_bytes"].mean(), 2)
            row_dict["mean_count"] = round(c_df["count"].mean(), 2)
            row_dict["mean_serror_rate"] = round(c_df["serror_rate"].mean(), 4)
            row_dict["mean_rerror_rate"] = round(c_df["rerror_rate"].mean(), 4)
            row_dict["top_service"] = c_df["service"].mode()[0] if "service" in c_df.columns else "N/A"
            row_dict["top_flag"] = c_df["flag"].mode()[0] if "flag" in c_df.columns else "N/A"
            
            # Label archetype based on characteristics
            if row_dict["mean_serror_rate"] > 0.5:
                archetype = "SYN-Flood / DoS (high serror)"
            elif row_dict["mean_rerror_rate"] > 0.5:
                archetype = "Rejection Scan / Port Sweep"
            elif row_dict["mean_count"] > 100:
                archetype = "High-Volume Flood"
            else:
                archetype = "Stealth / Exploit / Content"
            row_dict["archetype"] = archetype
            
        elif "unsw" in dataset_name.lower():
            row_dict["mean_dur"] = round(c_df["dur"].mean(), 4)
            row_dict["mean_sbytes"] = round(c_df["sbytes"].mean(), 2)
            row_dict["mean_dbytes"] = round(c_df["dbytes"].mean(), 2)
            row_dict["mean_rate"] = round(c_df["rate"].mean(), 2)
            row_dict["mean_sttl"] = round(c_df["sttl"].mean(), 2)
            row_dict["mean_ct_dst_sport_ltm"] = round(c_df["ct_dst_sport_ltm"].mean(), 2)
            row_dict["top_proto"] = c_df["proto"].mode()[0] if "proto" in c_df.columns else "N/A"
            row_dict["top_state"] = c_df["state"].mode()[0] if "state" in c_df.columns else "N/A"
            
            if row_dict["mean_rate"] > 50000:
                archetype = "High-Rate DoS / Volumetric"
            elif row_dict["mean_sttl"] > 100:
                archetype = "Remote Infiltration / Exploit"
            elif row_dict["mean_ct_dst_sport_ltm"] > 10:
                archetype = "Port / Host Scan (Surveillance)"
            else:
                archetype = "Low-and-Slow Attack"
            row_dict["archetype"] = archetype
            
        profiles.append(row_dict)
        
    return pd.DataFrame(profiles)


def generate_domain_constraints_spec(df: pd.DataFrame, schema: dict, dataset_name: str) -> dict:
    """Generate exact machine-readable domain constraints spec."""
    normal_df = df[df["is_anomaly"] == 0]
    spec = {}
    
    if "nsl" in dataset_name.lower():
        # Rate columns strictly in [0, 1]
        rate_cols = [c for c in normal_df.columns if c.endswith("_rate")]
        spec["rate_features_in_[0,1]"] = {c: {"min": 0.0, "max": 1.0} for c in rate_cols}
        
        # Bounded non-negative numerics
        spec["non_negative_counters"] = {}
        for c in ["duration", "src_bytes", "dst_bytes", "count", "srv_count", "dst_host_count", "dst_host_srv_count"]:
            if c in normal_df.columns:
                spec["non_negative_counters"][c] = {
                    "min": 0.0,
                    "max_observed_normal": float(normal_df[c].max()),
                }
                
        # Binary flags
        binary_cols = ["land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"]
        spec["binary_flags"] = {c: {"allowed": [0, 1]} for c in binary_cols if c in normal_df.columns}
        
        # Invariant / drop columns
        spec["constant_zero_columns"] = [c for c in ["num_outbound_cmds", "is_host_login"] if c in df.columns]
        
    elif "unsw" in dataset_name.lower():
        spec["non_negative_metrics"] = {}
        for c in ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit"]:
            if c in normal_df.columns:
                spec["non_negative_metrics"][c] = {
                    "min": 0.0,
                    "max_observed_normal": float(normal_df[c].max()),
                }
                
        spec["ttl_range_[0,255]"] = {c: {"min": 0, "max": 255} for c in ["sttl", "dttl"] if c in normal_df.columns}
        spec["binary_indicators"] = {c: {"allowed": [0, 1]} for c in ["is_sm_ips_ports", "is_ftp_login"] if c in normal_df.columns}
        
        # TCP-specific identities and bounds
        spec["tcp_handshake_identity"] = {
            "identity": "tcprtt == synack + ackdat",
            "max_observed_tcprtt": float(normal_df["tcprtt"].max()),
            "max_observed_synack": float(normal_df["synack"].max()),
            "max_observed_ackdat": float(normal_df["ackdat"].max()),
        }
        spec["tcp_specific_zero_for_non_tcp"] = ["swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat"]
        
    return spec


def main():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE EXPERIMENT 2 EDA & ARTIFACT GENERATION")
    print("=" * 70)
    
    root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    
    # 1. Load Datasets
    print("\n[1/5] Loading and classifying datasets...")
    df_nsl = load_dataset("nslkdd", root_dir=root_dir)
    df_unsw = load_dataset("unsw", root_dir=root_dir)
    
    schema_nsl = classify_features(df_nsl)
    schema_unsw = classify_features(df_unsw)
    
    print(f"NSL-KDD: Shape {df_nsl.shape}, Normal: {(df_nsl['is_anomaly']==0).sum()}, Abnormal: {(df_nsl['is_anomaly']==1).sum()}")
    print(f"  Continuous: {len(schema_nsl['continuous_cols'])}, Integer: {len(schema_nsl['integer_cols'])}, Binary: {len(schema_nsl['binary_cols'])}, Categorical: {len(schema_nsl['categorical_cols'])}, Constant: {len(schema_nsl['constant_cols'])}")
    
    print(f"UNSW-NB15: Shape {df_unsw.shape}, Normal: {(df_unsw['is_anomaly']==0).sum()}, Abnormal: {(df_unsw['is_anomaly']==1).sum()}")
    print(f"  Continuous: {len(schema_unsw['continuous_cols'])}, Integer: {len(schema_unsw['integer_cols'])}, Binary: {len(schema_unsw['binary_cols'])}, Categorical: {len(schema_unsw['categorical_cols'])}, Constant: {len(schema_unsw['constant_cols'])}")
    
    # 2. Outlier Scrubbing & Normal Extraction
    print("\n[2/5] Outlier scrubbing and normal dataset extraction (alpha=0.02)...")
    scrubbed_nsl, stats_nsl = scrub_normal_data(df_nsl, schema_nsl["continuous_cols"], contamination=0.02)
    scrubbed_unsw, stats_unsw = scrub_normal_data(df_unsw, schema_unsw["continuous_cols"], contamination=0.02)
    
    nsl_scrub_path = os.path.join(OUTPUT_DIR, "nslkdd_scrubbed_normal.csv")
    unsw_scrub_path = os.path.join(OUTPUT_DIR, "unsw_scrubbed_normal.csv")
    scrubbed_nsl.to_csv(nsl_scrub_path, index=False)
    scrubbed_unsw.to_csv(unsw_scrub_path, index=False)
    print(f"  NSL-KDD scrubbed: {stats_nsl['raw_normal_count']} -> {stats_nsl['final_scrubbed_count']} (Outliers: {stats_nsl['outliers_removed']}, Dups: {stats_nsl['duplicates_removed']}) -> Saved to {nsl_scrub_path}")
    print(f"  UNSW scrubbed: {stats_unsw['raw_normal_count']} -> {stats_unsw['final_scrubbed_count']} (Outliers: {stats_unsw['outliers_removed']}, Dups: {stats_unsw['duplicates_removed']}) -> Saved to {unsw_scrub_path}")
    
    # 3. Feature Profiles & Statistical Divergence
    print("\n[3/5] Computing statistical feature profiles and normal-vs-abnormal KS divergence...")
    prof_nsl = compute_feature_profiles(df_nsl, schema_nsl, "NSL-KDD")
    prof_unsw = compute_feature_profiles(df_unsw, schema_unsw, "UNSW-NB15")
    all_profiles = pd.concat([prof_nsl, prof_unsw], ignore_index=True)
    
    profiles_path = os.path.join(OUTPUT_DIR, "eda_feature_profiles.csv")
    all_profiles.to_csv(profiles_path, index=False)
    print(f"  Feature profiles saved to {profiles_path}")
    
    # Show Top Sentinel & Stealth Features
    for ds_name, p_df in [("NSL-KDD", prof_nsl), ("UNSW-NB15", prof_unsw)]:
        print(f"\n  Top 5 Sentinel Features (Max KS Separation) for {ds_name}:")
        top_sentinel = p_df.sort_values(by="ks_statistic", ascending=False).head(5)
        for _, r in top_sentinel.iterrows():
            print(f"    - {r['feature']} (type: {r['type']}): KS={r['ks_statistic']:.4f}, Norm-Median={r['norm_median']:.2f}, Abn-Median={r['abn_median']:.2f}")
            
        print(f"  Top 3 Stealth / High-Overlap Features (Min KS Separation) for {ds_name}:")
        stealth = p_df[p_df["type"] != "binary"].sort_values(by="ks_statistic", ascending=True).head(3)
        for _, r in stealth.iterrows():
            print(f"    - {r['feature']} (type: {r['type']}): KS={r['ks_statistic']:.4f}, Norm-Median={r['norm_median']:.2f}, Abn-Median={r['abn_median']:.2f}")
            
    # 4. Latent Anomaly Clustering (Pseudo-Novelty Harness Foundation)
    print("\n[4/5] Performing unsupervised clustering on abnormal instances (K=4)...")
    clust_nsl = analyze_anomaly_clusters(df_nsl, schema_nsl, "NSL-KDD", n_clusters=4)
    clust_unsw = analyze_anomaly_clusters(df_unsw, schema_unsw, "UNSW-NB15", n_clusters=4)
    all_clusters = pd.concat([clust_nsl, clust_unsw], ignore_index=True)
    
    cluster_path = os.path.join(OUTPUT_DIR, "anomaly_cluster_profiles.csv")
    all_clusters.to_csv(cluster_path, index=False)
    print(f"  Anomaly cluster profiles saved to {cluster_path}")
    print("\n  Discovered Attack Archetypes in Training Abnormal Traffic:")
    for _, r in all_clusters.iterrows():
        print(f"    [{r['dataset']}] Cluster {r['cluster_id']}: {r['archetype']} ({r['count']} flows, {r['pct_of_abnormal']}%)")
        
    # 5. Domain Constraints Specification
    print("\n[5/5] Exporting domain constraints specifications...")
    spec_nsl = generate_domain_constraints_spec(df_nsl, schema_nsl, "NSL-KDD")
    spec_unsw = generate_domain_constraints_spec(df_unsw, schema_unsw, "UNSW-NB15")
    
    spec_nsl_path = os.path.join(OUTPUT_DIR, "nslkdd_domain_constraints.json")
    spec_unsw_path = os.path.join(OUTPUT_DIR, "unsw_domain_constraints.json")
    
    with open(spec_nsl_path, "w") as f:
        json.dump(spec_nsl, f, indent=2)
    with open(spec_unsw_path, "w") as f:
        json.dump(spec_unsw, f, indent=2)
        
    print(f"  NSL-KDD constraints JSON saved to {spec_nsl_path}")
    print(f"  UNSW-NB15 constraints JSON saved to {spec_unsw_path}")
    print("\nEDA ANALYSIS RUN COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()

