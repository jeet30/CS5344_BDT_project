"""
Notebook builder and executor for Experiment 2 Comprehensive EDA.
Builds and executes expt-2/eda/EDA_expt2.ipynb using nbformat and nbclient.
"""
from __future__ import annotations

import os
import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "eda", "EDA_expt2.ipynb")


def build_notebook():
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
    
    # -------------------------------------------------------------
    # Title & Metadata
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# CS5344 Experiment 2: Comprehensive Exploratory Data Analysis
## Tabular Data Augmentation for Anomaly Detection (NSL-KDD & UNSW-NB15)
**Author**: Senior Data Scientist  
**Date**: September 2026  
**Context**: CS5344 Course Project — Downstream utility optimization for ECOD & Isolation Forest under private unseen anomaly distributions ($A_{val}, A_{test}$).
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Executive Problem Framing & Goals of Experiment 2 EDA

The objective in this competition is **not** to blindly maximize statistical fidelity (e.g. KS-distance or Wasserstein distance to training normal rows), but to generate synthetic normal data $\\tilde{D}$ ($N=40,000$ for NSL-KDD, $N=70,000$ for UNSW-NB15) that **maximizes downstream anomaly detection utility**:
$$S = \\frac{\\text{AUPRC}_{\\text{ECOD}} + \\text{AUPRC}_{\\text{IForest}}}{2}$$

Crucially, as established in the problem formulation:
$$\\{\\text{abnormal in } D_{train}\\} = A_{seen}$$
$$\\{\\text{abnormal in } D_{valid}\\} = A_{seen} \\cup A_{val}$$
$$\\{\\text{abnormal in } D_{test}\\} = A_{seen} \\cup A_{val} \\cup A_{test}$$

In Experiment 1, a conditional Gaussian copula generator performed well locally on NSL-KDD ($S=0.9266$ on public LB) but suffered a severe reversal on UNSW-NB15 ($0.4192$ vs starter kit's $0.4338$). This EDA systematically investigates the root causes:
1. **The Boundary Tightness Paradox**: How ECOD and Isolation Forest penalize overly-contracted normal manifolds when novel attack shapes appear.
2. **Deterministic Network Protocol Identities**: Auditing `tcprtt = synack + ackdat` and protocol-specific zeroing.
3. **Latent Attack Clustering**: Extracting distinct attack archetypes to power a local **Pseudo-Novelty Stress-Test** that mimics $A_{val}$ and $A_{test}$.
"""))

    # -------------------------------------------------------------
    # Section 0: Setup & Imports
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 0. Environment Setup & Modular Library Imports"""))
    cells.append(nbf.v4.new_code_cell("""import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ensure reproducibility and headless plotting
os.environ["MPLCONFIGDIR"] = "/tmp"
np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 10

# Add src to sys.path
sys.path.append(os.path.abspath("../src"))
from data_loader import load_dataset, classify_features, SUBMISSION_SPECS
from cleaner import scrub_normal_data
from constraints import audit_constraints, enforce_nslkdd_constraints, enforce_unsw_constraints
from evaluator import build_feature_matrices, score_detectors, cluster_abnormal_traffic

print("Experiment 2 environment initialized successfully.")
"""))

    # -------------------------------------------------------------
    # Section 1: NSL-KDD Deep Dive
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset A — NSL-KDD Deep Dive"""))
    cells.append(nbf.v4.new_code_cell("""df_nsl = load_dataset("nslkdd", root_dir="../..")
schema_nsl = classify_features(df_nsl)

print(f"NSL-KDD Shape: {df_nsl.shape}")
print(f"Class Breakdown:\\n{df_nsl['is_anomaly'].value_counts(normalize=True).round(4) * 100}%")
print(f"\\nFeature Categories:")
print(f"  Continuous ({len(schema_nsl['continuous_cols'])}): {schema_nsl['continuous_cols'][:5]}...")
print(f"  Integer ({len(schema_nsl['integer_cols'])}): {schema_nsl['integer_cols']}")
print(f"  Binary ({len(schema_nsl['binary_cols'])}): {schema_nsl['binary_cols']}")
print(f"  Categorical ({len(schema_nsl['categorical_cols'])}): {schema_nsl['categorical_cols']}")
print(f"  Constant / Invariant ({len(schema_nsl['constant_cols'])}): {schema_nsl['constant_cols']}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.1 Outlier Scrubbing & Normal Isolation (NSL-KDD)"""))
    cells.append(nbf.v4.new_code_cell("""scrubbed_nsl, stats_nsl = scrub_normal_data(df_nsl, schema_nsl["continuous_cols"], contamination=0.02)
print("NSL-KDD Normal Scrubbing Audit:")
for k, v in stats_nsl.items():
    print(f"  {k}: {v}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.2 Anomaly Divergence & Sentinel Features (NSL-KDD)
We quantify the distribution separation between normal ($y=0$) and attack ($y=1$) connections using the two-sample Kolmogorov-Smirnov (KS) test.
"""))
    cells.append(nbf.v4.new_code_cell("""profiles_path = "outputs/eda_feature_profiles.csv"
profiles_df = pd.read_csv(profiles_path)
nsl_prof = profiles_df[profiles_df["dataset"] == "NSL-KDD"].sort_values(by="ks_statistic", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=nsl_prof.head(10), x="ks_statistic", y="feature", palette="Blues_r", ax=ax)
ax.set_title("Top 10 Sentinel Features in NSL-KDD (Kolmogorov-Smirnov Statistic)", fontsize=13, fontweight="bold")
ax.set_xlabel("KS Statistic (Separation between Normal and Abnormal)")
plt.tight_layout()
plt.show()

print("Top 5 Sentinel Features:")
print(nsl_prof[["feature", "type", "norm_median", "abn_median", "ks_statistic"]].head(5).to_string(index=False))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.3 Latent Attack Archetypes in NSL-KDD (Pseudo-Novelty Foundation)
Because the validation and test sets include unseen attack categories ($A_{val}, A_{test}$), we perform unsupervised clustering on the abnormal training rows ($y=1$) to discover the natural attack archetypes within $A_{seen}$.
"""))
    cells.append(nbf.v4.new_code_cell("""clusters_path = "outputs/anomaly_cluster_profiles.csv"
clusters_df = pd.read_csv(clusters_path)
nsl_clusters = clusters_df[clusters_df["dataset"] == "NSL-KDD"]

print("NSL-KDD Discovered Attack Clusters:")
print(nsl_clusters[["cluster_id", "archetype", "count", "pct_of_abnormal", "mean_duration", "mean_src_bytes", "mean_serror_rate", "top_service"]].to_string(index=False))
"""))

    # -------------------------------------------------------------
    # Section 2: UNSW-NB15 Deep Dive
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset B — UNSW-NB15 Deep Dive"""))
    cells.append(nbf.v4.new_code_cell("""df_unsw = load_dataset("unsw", root_dir="../..")
schema_unsw = classify_features(df_unsw)

print(f"UNSW-NB15 Shape: {df_unsw.shape}")
print(f"Class Breakdown:\\n{df_unsw['is_anomaly'].value_counts(normalize=True).round(4) * 100}%")
print(f"\\nFeature Categories:")
print(f"  Continuous ({len(schema_unsw['continuous_cols'])}): {schema_unsw['continuous_cols'][:5]}...")
print(f"  Integer ({len(schema_unsw['integer_cols'])}): {schema_unsw['integer_cols'][:5]}...")
print(f"  Binary ({len(schema_unsw['binary_cols'])}): {schema_unsw['binary_cols']}")
print(f"  Categorical ({len(schema_unsw['categorical_cols'])}): {schema_unsw['categorical_cols']}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.1 Outlier Scrubbing & Normal Isolation (UNSW-NB15)"""))
    cells.append(nbf.v4.new_code_cell("""scrubbed_unsw, stats_unsw = scrub_normal_data(df_unsw, schema_unsw["continuous_cols"], contamination=0.02)
print("UNSW-NB15 Normal Scrubbing Audit:")
for k, v in stats_unsw.items():
    print(f"  {k}: {v}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.2 Network Physical Invariants & `tcprtt` Inflation Audit (UNSW-NB15)
In Experiment 1, post-processing recomputed `tcprtt = synack + ackdat`, but bounded `synack` and `ackdat` to their independent historical maxes. Consequently, synthetic rows produced `tcprtt` sums as high as 5.04, whereas the true maximum training `tcprtt` is 3.18! We verify this domain identity and enforce proportional scaling.
"""))
    cells.append(nbf.v4.new_code_cell("""normal_unsw = df_unsw[df_unsw["is_anomaly"] == 0]
diff = np.abs(normal_unsw["tcprtt"] - (normal_unsw["synack"] + normal_unsw["ackdat"]))
print(f"Real Data Identity Verification: Max absolute mismatch = {diff.max():.6e}")
print(f"Max observed normal tcprtt: {normal_unsw['tcprtt'].max():.4f}")
print(f"Max observed normal synack: {normal_unsw['synack'].max():.4f}")
print(f"Max observed normal ackdat: {normal_unsw['ackdat'].max():.4f}")

# Non-TCP zeroing verification
non_tcp_normal = normal_unsw[normal_unsw["proto"].str.lower() != "tcp"]
tcp_cols = ["swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat"]
non_tcp_violations = (non_tcp_normal[tcp_cols] > 0).sum()
print(f"\\nNon-TCP records with non-zero TCP handshake fields in real normal data:\\n{non_tcp_violations}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.3 Anomaly Divergence & Sentinel Features (UNSW-NB15)"""))
    cells.append(nbf.v4.new_code_cell("""unsw_prof = profiles_df[profiles_df["dataset"] == "UNSW-NB15"].sort_values(by="ks_statistic", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=unsw_prof.head(10), x="ks_statistic", y="feature", palette="Reds_r", ax=ax)
ax.set_title("Top 10 Sentinel Features in UNSW-NB15 (Kolmogorov-Smirnov Statistic)", fontsize=13, fontweight="bold")
ax.set_xlabel("KS Statistic (Separation between Normal and Abnormal)")
plt.tight_layout()
plt.show()

print("Top 5 Sentinel Features in UNSW-NB15:")
print(unsw_prof[["feature", "type", "norm_median", "abn_median", "ks_statistic"]].head(5).to_string(index=False))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.4 Latent Attack Archetypes in UNSW-NB15 (Pseudo-Novelty Foundation)"""))
    cells.append(nbf.v4.new_code_cell("""unsw_clusters = clusters_df[clusters_df["dataset"] == "UNSW-NB15"]
print("UNSW-NB15 Discovered Attack Clusters:")
print(unsw_clusters[["cluster_id", "archetype", "count", "pct_of_abnormal", "mean_dur", "mean_rate", "mean_sttl", "top_proto", "top_state"]].to_string(index=False))
"""))

    # -------------------------------------------------------------
    # Section 3: Inductive Biases & UNSW Reversal Breakdown
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Mathematical Breakdown of the UNSW Reversal & Downstream Detector Biases

Why did the Gaussian copula beat the starter kit locally on UNSW-NB15 ($0.4744$ vs $0.4606$), but lose on the public Kaggle leaderboard ($0.4192$ vs $0.4338$)?

### 1. Inductive Bias of ECOD (Empirical Cumulative Distribution Outlier Detection)
ECOD calculates outlier scores by estimating univariate empirical CDFs in both tails:
$$O_i = \\sum_{d=1}^m -\\log \\left( \\min \\left( \\hat{F}_d(x_{id}), 1 - \\hat{F}_d(x_{id}) \\right) \\right)$$
- If the synthetic normal dataset $\\tilde{D}$ strictly clips or binds to the narrow training range $[x_d^{\\min}, x_d^{\\max}]$ without tail coverage, $\\hat{F}_d$ evaluates to extreme values ($0$ or $1$) on real test normal connections that sit slightly outside the training envelope.
- As a result, ECOD assigns massive outlier scores ($-\\log \\epsilon$) to valid normal test points, creating hundreds of false positives that decimate AUPRC!

### 2. Inductive Bias of Isolation Forest
Isolation Forest partitions space with random axis-aligned splits:
$$s(x, n) = 2^{-\\frac{E(h(x))}{c(n)}}$$
- When synthetic data is tightly packed around the training core, peripheral regions become completely empty.
- Any test normal flow that exhibits slight protocol variance falls into empty peripheral space and is isolated at extremely shallow depth $h(x)$, triggering false alarms.

### 3. The Accident of the Starter Kit's Global Jitter
The starter kit sampled prototypes and added global Gaussian noise:
$$\\tilde{x}_d = x_d + \\mathcal{N}(0, 0.5 \\cdot \\sigma_d)$$
Although this broke correlations and physical constraints, it inadvertently **widened the normal support** in every dimension! This wider boundary gave the downstream detectors slack to accommodate slightly atypical normal traffic and novel attack configurations without premature false alarms.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Key Conclusions & Augmentation Strategy for Experiment 2

| Finding | Diagnostic Evidence | Actionable Prescription for Expt-2 Generator |
|---|---|---|
| **Boundary Tightness is Catastrophic for Novel Generalization** | UNSW reversal ($0.4192$ LB vs $0.4744$ local) | Implement **controlled tail expansion / boundary slackness** (e.g. variance inflation or Generalized Pareto tail extrapolation) rather than hard empirical bounding. |
| **`tcprtt` Domain Identity Violation** | Experiment 1 produced `tcprtt` up to 5.04 vs 3.18 training max | Enforce proportional re-scaling on `synack` and `ackdat` before summation so that `tcprtt == synack + ackdat` and never exceeds empirical valid ranges. |
| **Protocol Handshake Invariants** | Non-TCP traffic had non-zero window/sequence values | Enforce strict zeroing of `swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat` for non-TCP protocols. |
| **Sentinel vs Stealth Features** | KS test profiles (`src_bytes`, `sttl`, `ct_state_ttl`) | Preserve multimodal separation on sentinels while allowing expansion on continuous traffic metrics. |
| **Local Evaluation Blind Spot** | Random train/test split has $A_{seen}$ only | Deploy Track B **Pseudo-Novelty Stress-Test** using the discovered attack clusters to evaluate leave-cluster-out generalization before submitting. |
"""))

    nb.cells = cells
    
    with open(NOTEBOOK_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {NOTEBOOK_PATH}")
    
    # Execute notebook
    print("Executing notebook with nbclient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": os.path.dirname(NOTEBOOK_PATH)}})
    client.execute()
    
    with open(NOTEBOOK_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"Notebook successfully executed and saved to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()

