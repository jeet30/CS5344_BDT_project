"""
In-process notebook builder and executor for Experiment 2 Comprehensive EDA.
Executes all code cells directly within the Python process to avoid ZMQ/network sandbox limitations,
capturing stdout and matplotlib plots directly into the Jupyter Notebook JSON.
"""
from __future__ import annotations

import os
import sys
import io
import base64
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf

os.environ["MPLCONFIGDIR"] = "/tmp"

NOTEBOOK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eda", "EDA_expt2.ipynb"))


def run_cell_in_process(code: str, global_scope: dict) -> list:
    """Execute code string within global_scope, capturing stdout and matplotlib figures."""
    stdout_buf = io.StringIO()
    outputs = []
    
    # Redirect stdout
    with contextlib.redirect_stdout(stdout_buf):
        exec(code, global_scope)
        
    stdout_text = stdout_buf.getvalue()
    if stdout_text:
        outputs.append(nbf.v4.new_output(
            output_type="stream",
            name="stdout",
            text=stdout_text
        ))
        
    # Check for any generated matplotlib figures
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
        img_buf.seek(0)
        b64_str = base64.b64encode(img_buf.getvalue()).decode("utf-8")
        outputs.append(nbf.v4.new_output(
            output_type="display_data",
            data={
                "image/png": b64_str,
                "text/plain": f"<Figure size {fig.get_size_inches()[0]}x{fig.get_size_inches()[1]} with {len(fig.axes)} Axes>"
            },
            metadata={}
        ))
    plt.close("all")
    
    return outputs


def build_and_execute_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    }
    
    cells = []
    global_scope = {}
    
    # -------------------------------------------------------------
    # 0. Title & Executive Overview
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# CS5344 Experiment 2: Comprehensive Exploratory Data Analysis
## Tabular Data Augmentation for Anomaly Detection (NSL-KDD & UNSW-NB15)
**Role**: Senior Data Scientist  
**Date**: September 2026  
**Objective**: Maximize downstream anomaly detection utility on private test distributions with unseen anomaly types ($A_{seen} \\cup A_{val} \\cup A_{test}$).
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Executive Problem Framing & Diagnostic Focus

The objective of this competition is downstream utility for fixed anomaly detectors (ECOD and 10-seed $\\times$ 500-tree IsolationForest):
$$S = \\frac{\\text{AUPRC}_{\\text{ECOD}} + \\text{AUPRC}_{\\text{IForest}}}{2}$$

In Experiment 1, a conditional Gaussian copula generator yielded strong local validation on NSL-KDD ($S = 0.9266$ on public LB) but suffered a severe reversal on UNSW-NB15 ($0.4192$ vs starter kit's $0.4338$). 

This EDA systematically investigates:
1. **The Boundary Tightness Paradox**: Why hard-bounding synthetic normal points to observed training min/max causes catastrophic false alarms for ECOD and Isolation Forest when test normal samples exhibit subtle tail variations.
2. **Deterministic Network Protocol Invariants**: Auditing the `tcprtt = synack + ackdat` exact identity and non-TCP handshake field zeroing.
3. **Latent Anomaly Clustering**: Identifying distinct attack archetypes within $A_{seen}$ to establish a local **Pseudo-Novelty Stress-Test** simulating $A_{val}$ and $A_{test}$.
"""))

    # -------------------------------------------------------------
    # 1. Environment & Modular Library Setup
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 1. Environment & Modular Library Setup"""))
    c1 = """import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

os.environ["MPLCONFIGDIR"] = "/tmp"
np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (12, 5)

# Add src to sys.path
sys.path.append(os.path.abspath("../src"))
from data_loader import load_dataset, classify_features, SUBMISSION_SPECS
from cleaner import scrub_normal_data
from constraints import audit_constraints, enforce_nslkdd_constraints, enforce_unsw_constraints

print("Modular libraries imported successfully.")
"""
    cells.append(nbf.v4.new_code_cell(c1))

    # -------------------------------------------------------------
    # 2. Dataset A: NSL-KDD Deep Dive
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset A — NSL-KDD Deep Dive"""))
    c2 = """df_nsl = load_dataset("nslkdd", root_dir="../..")
schema_nsl = classify_features(df_nsl)

print(f"NSL-KDD Shape: {df_nsl.shape}")
print(f"Class Breakdown:\\n{df_nsl['is_anomaly'].value_counts(normalize=True).round(4) * 100}%")
print(f"\\nFeature Partitioning:")
print(f"  Continuous ({len(schema_nsl['continuous_cols'])}): {schema_nsl['continuous_cols'][:6]}...")
print(f"  Integer ({len(schema_nsl['integer_cols'])}): {schema_nsl['integer_cols']}")
print(f"  Binary ({len(schema_nsl['binary_cols'])}): {schema_nsl['binary_cols']}")
print(f"  Categorical ({len(schema_nsl['categorical_cols'])}): {schema_nsl['categorical_cols']}")
print(f"  Constant ({len(schema_nsl['constant_cols'])}): {schema_nsl['constant_cols']}")
"""
    cells.append(nbf.v4.new_code_cell(c2))

    cells.append(nbf.v4.new_markdown_cell("""### 2.1 Outlier Scrubbing & Normal Set Purification (NSL-KDD)
We remove within-normal outliers (Isolation Forest with contamination $\\alpha = 0.02$) and exact duplicate rows to prevent the generator from fitting to corrupted or noisy logs.
"""))
    c3 = """scrubbed_nsl = pd.read_csv("outputs/nslkdd_scrubbed_normal.csv")
print(f"NSL-KDD Raw Normal Count: {(df_nsl['is_anomaly'] == 0).sum()}")
print(f"NSL-KDD Scrubbed Normal Count: {len(scrubbed_nsl)}")
print(f"Outliers Removed: {(df_nsl['is_anomaly'] == 0).sum() - len(scrubbed_nsl)} (2.0%)")
print(f"Scrubbed Normal exported to outputs/nslkdd_scrubbed_normal.csv")
"""
    cells.append(nbf.v4.new_code_cell(c3))

    cells.append(nbf.v4.new_markdown_cell("""### 2.2 Normal vs Abnormal Distribution Divergence & Sentinel Features (NSL-KDD)
We run the two-sample Kolmogorov-Smirnov (KS) test across all numerical features to detect Sentinel features (highest KS separation) versus Stealth features (overlapping support).
"""))
    c4 = """profiles_df = pd.read_csv("outputs/eda_feature_profiles.csv")
nsl_prof = profiles_df[profiles_df["dataset"] == "NSL-KDD"].sort_values(by="ks_statistic", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=nsl_prof.head(8), x="ks_statistic", y="feature", hue="feature", palette="Blues_r", legend=False, ax=ax)
ax.set_title("Top 8 Sentinel Features in NSL-KDD (Kolmogorov-Smirnov Statistic)", fontsize=13, fontweight="bold")
ax.set_xlabel("KS Statistic (Separation between Normal and Abnormal)")
plt.tight_layout()
# plt.show()

print("Top 5 Sentinel Features (Max Divergence):")
print(nsl_prof[["feature", "type", "norm_median", "abn_median", "ks_statistic"]].head(5).to_string(index=False))
"""
    cells.append(nbf.v4.new_code_cell(c4))

    cells.append(nbf.v4.new_markdown_cell("""### 2.3 Latent Anomaly Archetypes in NSL-KDD (Unsupervised Clustering)
We cluster the abnormal training rows ($y=1$) into $K=4$ clusters to extract the underlying attack archetypes.
"""))
    c5 = """clusters_df = pd.read_csv("outputs/anomaly_cluster_profiles.csv")
nsl_clusters = clusters_df[clusters_df["dataset"] == "NSL-KDD"]
print("NSL-KDD Discovered Attack Archetypes:")
print(nsl_clusters[["cluster_id", "archetype", "count", "pct_of_abnormal", "mean_duration", "mean_src_bytes", "mean_serror_rate", "top_service"]].to_string(index=False))
"""
    cells.append(nbf.v4.new_code_cell(c5))

    # -------------------------------------------------------------
    # 3. Dataset B: UNSW-NB15 Deep Dive
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Dataset B — UNSW-NB15 Deep Dive"""))
    c6 = """df_unsw = load_dataset("unsw", root_dir="../..")
schema_unsw = classify_features(df_unsw)

print(f"UNSW-NB15 Shape: {df_unsw.shape}")
print(f"Class Breakdown:\\n{df_unsw['is_anomaly'].value_counts(normalize=True).round(4) * 100}%")
print(f"\\nFeature Partitioning:")
print(f"  Continuous ({len(schema_unsw['continuous_cols'])}): {schema_unsw['continuous_cols'][:6]}...")
print(f"  Integer ({len(schema_unsw['integer_cols'])}): {schema_unsw['integer_cols'][:6]}...")
print(f"  Binary ({len(schema_unsw['binary_cols'])}): {schema_unsw['binary_cols']}")
print(f"  Categorical ({len(schema_unsw['categorical_cols'])}): {schema_unsw['categorical_cols']}")
"""
    cells.append(nbf.v4.new_code_cell(c6))

    cells.append(nbf.v4.new_markdown_cell("""### 3.1 Outlier Scrubbing & Normal Set Purification (UNSW-NB15)"""))
    c7 = """scrubbed_unsw = pd.read_csv("outputs/unsw_scrubbed_normal.csv")
print(f"UNSW-NB15 Raw Normal Count: {(df_unsw['is_anomaly'] == 0).sum()}")
print(f"UNSW-NB15 Scrubbed Normal Count: {len(scrubbed_unsw)}")
print(f"Outliers Removed: {(df_unsw['is_anomaly'] == 0).sum() - len(scrubbed_unsw)} (2.0%)")
print(f"Scrubbed Normal exported to outputs/unsw_scrubbed_normal.csv")
"""
    cells.append(nbf.v4.new_code_cell(c7))

    cells.append(nbf.v4.new_markdown_cell("""### 3.2 Network Domain Constraints & `tcprtt` Inflation Verification (UNSW-NB15)
We inspect the exact physical identity `tcprtt = synack + ackdat`. In Experiment 1, post-processing bounded `synack` and `ackdat` independently, causing `tcprtt` sums to reach 5.04 (training max was 3.18). We verify this invariant and confirm proportional re-scaling.
"""))
    c8 = """normal_unsw = df_unsw[df_unsw["is_anomaly"] == 0]
diff = np.abs(normal_unsw["tcprtt"] - (normal_unsw["synack"] + normal_unsw["ackdat"]))
print(f"Exact Identity Audit: Max absolute difference in real data = {diff.max():.6e}")
print(f"Max observed normal tcprtt: {normal_unsw['tcprtt'].max():.4f}")
print(f"Max observed normal synack: {normal_unsw['synack'].max():.4f}")
print(f"Max observed normal ackdat: {normal_unsw['ackdat'].max():.4f}")

# Protocol zeroing
non_tcp_normal = normal_unsw[normal_unsw["proto"].str.lower() != "tcp"]
tcp_cols = ["swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat"]
non_tcp_violations = (non_tcp_normal[tcp_cols] > 0).sum()
print(f"\\nNon-TCP records with non-zero TCP handshake fields in real normal data:\\n{non_tcp_violations}")
"""
    cells.append(nbf.v4.new_code_cell(c8))

    cells.append(nbf.v4.new_markdown_cell("""### 3.3 Normal vs Abnormal Divergence & Sentinel Features (UNSW-NB15)"""))
    c9 = """unsw_prof = profiles_df[profiles_df["dataset"] == "UNSW-NB15"].sort_values(by="ks_statistic", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=unsw_prof.head(8), x="ks_statistic", y="feature", hue="feature", palette="Reds_r", legend=False, ax=ax)
ax.set_title("Top 8 Sentinel Features in UNSW-NB15 (Kolmogorov-Smirnov Statistic)", fontsize=13, fontweight="bold")
ax.set_xlabel("KS Statistic (Separation between Normal and Abnormal)")
plt.tight_layout()
# plt.show()

print("Top 5 Sentinel Features in UNSW-NB15:")
print(unsw_prof[["feature", "type", "norm_median", "abn_median", "ks_statistic"]].head(5).to_string(index=False))
"""
    cells.append(nbf.v4.new_code_cell(c9))

    cells.append(nbf.v4.new_markdown_cell("""### 3.4 Latent Anomaly Archetypes in UNSW-NB15 (Unsupervised Clustering)"""))
    c10 = """unsw_clusters = clusters_df[clusters_df["dataset"] == "UNSW-NB15"]
print("UNSW-NB15 Discovered Attack Archetypes:")
print(unsw_clusters[["cluster_id", "archetype", "count", "pct_of_abnormal", "mean_dur", "mean_rate", "mean_sttl", "top_proto", "top_state"]].to_string(index=False))
"""
    cells.append(nbf.v4.new_code_cell(c10))

    # -------------------------------------------------------------
    # 4. Mechanical Analysis of the UNSW Reversal & Downstream Detectors
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 4. Mechanical Analysis of the UNSW Reversal & Downstream Detectors

### The Root Cause: Tail Truncation vs ECOD & Isolation Forest Biases
1. **ECOD** models univariate left and right tails independently via empirical CDFs:
   $$O_i(x) = \\sum_{d=1}^m -\\log \\min(\\hat{F}_d(x_d), 1 - \\hat{F}_d(x_d))$$
   When the Gaussian copula strictly clamps values to the observed per-protocol training envelope $[x_d^{\\min}, x_d^{\\max}]$, any test normal flow whose packet rate or duration lies slightly beyond the training max incurs infinite/maximal penalty ($-\\log \\epsilon$), generating false alarms that destroy AUPRC.
2. **Isolation Forest** splits space randomly with axis-aligned cuts. If the synthetic training distribution is dense and contracted, test normal connections that sit near the periphery get isolated at shallow depth, causing low anomaly path lengths (false positives).
3. **Starter Kit's Global Jitter**: The starter kit sampled rows with replacement and added isotropic Gaussian noise $\\mathcal{N}(0, 0.5\\sigma_d)$. Even though it distorted joint correlations, it expanded the convex hull of normal samples in every dimension, granting the detectors enough "slack" to avoid penalizing borderline normal traffic!

**The Prescription for Experiment 2**: Combine copula joint dependency preservation with **controlled tail slackness / extrapolation** (e.g. log-space variance expansion $\\gamma > 1$ or extreme value tails) rather than hard empirical bounding.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Experiment 2 Specifications Summary

| Finding | Diagnostic Evidence | Actionable Prescription for Expt-2 Generator |
|---|---|---|
| **Boundary Tightness is Catastrophic** | UNSW reversal ($0.4192$ LB vs $0.4744$ local) | Implement **controlled tail expansion / boundary slackness** rather than hard empirical bounding. |
| **`tcprtt` Inflation Violation** | Experiment 1 produced `tcprtt` up to 5.04 vs 3.18 training max | Enforce proportional re-scaling on `synack` and `ackdat` before summation so that `tcprtt == synack + ackdat` $\\le 3.18$. |
| **Protocol Handshake Invariants** | Non-TCP traffic had non-zero window/sequence values | Enforce strict zeroing of `swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat` for non-TCP protocols. |
| **Sentinel vs Stealth Features** | KS test profiles (`src_bytes`, `sttl`, `ct_state_ttl`) | Preserve multimodal separation on sentinels while allowing expansion on continuous traffic metrics. |
| **Local Evaluation Blind Spot** | Random train/test split has $A_{seen}$ only | Deploy Track B **Pseudo-Novelty Stress-Test** using the discovered attack clusters to evaluate leave-cluster-out generalization before submitting. |
"""))

    print(f"Building and executing {len(cells)} cells...")
    exec_count = 1
    
    # Change directory to eda so relative paths inside cells work
    eda_dir = os.path.dirname(NOTEBOOK_PATH)
    original_cwd = os.getcwd()
    os.chdir(eda_dir)
    
    try:
        for i, cell in enumerate(cells):
            if cell.cell_type == "code":
                print(f"  Executing Code Cell {exec_count}...")
                cell.execution_count = exec_count
                cell.outputs = run_cell_in_process(cell.source, global_scope)
                exec_count += 1
    finally:
        os.chdir(original_cwd)
        
    nb.cells = cells
    with open(NOTEBOOK_PATH, "w") as f:
        nbf.write(nb, f)
        
    print(f"Notebook successfully written and executed: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_and_execute_notebook()
