# Experiment 4: Restored Contrastive DVS-Copula & Multi-Track Generative Architecture

**Date:** 5 October 2026  
**Authors:** Aayush Majumdar (A0352046L) & Vadiraj Rao (A0353463A) — Group 42  

---

## 1. Executive Summary & Forensic Audit of the Track 2 Dip

### 1.1 Direct Resolution of the Score Discrepancy
In our interim diagnostic run, Track 2 reported a local proxy composite score of **$S_{\text{proxy}} = 0.4919$** on UNSW-NB15. This was indeed lower than our original Phase 7 Contrastive DVS-Copula benchmark ($S_{\text{proxy}} = 0.5870$, Kaggle Public LB = **$0.53307$**). 

The user's observation was **100% accurate**: the interim Track 2 configuration had regressed because an essential component of the original Phase 7 pipeline had been omitted.

### 1.2 Forensic Root-Cause Analysis: Why the Interim Score Dropped to 0.4919
Following the regression in Experiment 3 ($0.49966$ on Kaggle LB), our initial diagnosis wrongly hypothesized that *any* Contrastive Likelihood-Ratio (CLR) weighting had damaged the generator. In response, the interim Track 2 script completely removed CLR purification and modeled the raw normal training distribution directly.

This caused an immediate, severe breakdown:
1. **Raw Normal Mode Contamination:** In UNSW-NB15, normal traffic heavily overlaps with attack traffic across key discriminative features (`dur`, `rate`, `sload`, `dinpkt`, `trans_depth`). In raw training data, legitimate sessions frequently exhibit characteristics that mimic recon and generic attacks.
2. **Isolation Forest Boundary Collapse:** When synthetic data is generated from raw unweighted normal distributions, the generated points populate ambiguous, overlapping regions. Isolation Forest—which relies on finding isolated partitions—splits indiscriminately on these overlapping boundaries. In the interim run, **$\text{AUPRC}_{\text{IForest}}$ collapsed from $0.6341$ down to $0.4784$** (a massive $-0.1557$ drop), dragging composite $S$ down to **$0.4919$**.
3. **What Actually Broke Experiment 3 ($0.49966$):** The regression in Experiment 3 was **not** caused by the original mathematical CLR weighting. It was caused by the aggressive manual heuristic pruning added in `hybrid_cspc.py`:
   - Hard deletion of `state == 'INT'` (which wiped out 27% of valid normal traffic, causing catastrophic ECOD false alarms on WAN traffic).
   - Quantile uniform histogram smoothing that distorted empirical densities.
   - K-Means sample-level concatenation (blending 50% real rows reintroduced the rigid empirical maximum volume ceilings, recreating ECOD's right-tail penalty cliff).

### 1.3 The Rectification: Restoring Pure Contrastive DVS-Copula
By restoring the pure, unadulterated Phase 7 mathematical formulation—combining discrete Contrastive Likelihood-Ratio weighting, protocol-stratified Gaussian copulas, Discriminative Variance Scaling ($\alpha = 35.0$), closed-loop physical constraints, and 5-seed anti-shakeup bagging—Experiment 4 achieves:
- **UNSW-NB15:** $\text{AUPRC}_{\text{ECOD}} = 0.5384$, $\text{AUPRC}_{\text{IForest}} = 0.6392$, **Composite $S = 0.5888$** (surpasses the original Phase 7 benchmark of $0.5870$ and recovers $+0.0969$ over the interim dip).
- **NSL-KDD:** $\text{AUPRC}_{\text{ECOD}} = 0.8997$, $\text{AUPRC}_{\text{IForest}} = 0.9583$, **Composite $S = 0.9290$**.

---

## 2. Theoretical Architecture of Experiment 4

Experiment 4 establishes two distinct, production-grade tracks:
- **Track 2 (Immediate Local Submission):** CPU-efficient, mathematically rigorous Contrastive DVS-Copula v2 (for UNSW-NB15) and Controlled Partial Balancing v2 with strict integer clamping (for NSL-KDD).
- **Track 1 (Heavy-Compute GPU Cluster):** TabSyn-style Latent Denoising Diffusion Probabilistic Model (DDPM) conditioned on contrastive purity weights, prepared for execution on the NUS SoC GPU cluster via Slurm.

```
+----------------------------------------------------------------------------------------------------+
|                                    EXPERIMENT 4 DUAL-TRACK ARCHITECTURE                            |
+----------------------------------------------------------------------------------------------------+
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
     [Track 2: Analytical & Copula]                                   [Track 1: SoC GPU Cluster]
                 │                                                                 │
      ┌──────────┴──────────┐                                         ┌────────────┴────────────┐
      ▼                     ▼                                         ▼                         ▼
 [NSL-KDD]             [UNSW-NB15]                               [Tabular VAE]            [Latent DDPM]
  CPB v2                Contrastive DVS-Copula v2                Mixed Cont/Cat           Score Matching
  λ = 0.30              Discrete CLR Purification (Top-K)        Embeddings + MLP         Reverse Diffusion
  Exact Int Clamping    Proto-Stratified Gaussian Copula         Latent Dimension=64      T=1000 Timesteps
  5-Seed Bagging        Variance Expansion (α=35.0)              Sinkhorn/ELBO Loss       Conditioned Purity
                        Physical Law Projection (tcprtt)
                        5-Seed Anti-Shakeup Bagging
                 │                                                                 │
                 ▼                                                                 ▼
   [Local Proxy: S = 0.9290 | S = 0.5888]                              [SoC Slurm: sbatch run_cluster.sh]
   [Submissions: 40k NSL / 70k UNSW]                                   [Submissions: TabSyn 40k / 70k]
```

### 2.1 Track 2 Generator Details

#### A. UNSW-NB15: Contrastive DVS-Copula v2 (`dvs_copula_v2.py`)
1. **Discrete Marginal Likelihood-Ratio Purification:**
   For each training sample $x_i$, a purity score $\mathcal{L}(x_i)$ is computed across the most discriminative discrete features $\mathcal{S} = \{\text{proto}, \text{service}, \text{state}\}$:
   $$\log \mathcal{L}(x_i) = \sum_{j \in \mathcal{S}} \left[ \log \frac{P(X_{ij} = x_{ij} \mid y=0) + \epsilon}{P(X_{ij} = x_{ij} \mid y=1) + \epsilon} \right]$$
   Normal rows with high positive log-likelihood ratios represent uncontaminated normal network behavior. Sampling probabilities are assigned via softmax temperature scaling:
   $$w_i = \frac{\exp(\beta \cdot \log \mathcal{L}(x_i))}{\sum_k \exp(\beta \cdot \log \mathcal{L}(x_k))}$$
2. **Protocol Stratification & Latent Gaussian Copula:**
   Traffic is partitioned into primary protocol strata (TCP, UDP, and other). Within each stratum, marginal empirical cumulative distributions $\hat{F}_j$ are mapped to standard normal variates:
   $$Z_j = \Phi^{-1}(\hat{F}_j(X_j))$$
   The latent covariance matrix $\Sigma = \text{Corr}(Z)$ is regularized via Higham's nearest positive-definite projection.
3. **Discriminative Variance Scaling (DVS):**
   To eliminate false positives caused by ECOD's right-tail penalty on flow volume confounders (`sbytes`, `dbytes`, `sload`, `dload`, `spkts`, `dpkts`), the latent standard deviation of volume features is scaled by factor $\alpha = 35.0$. This smooths the synthetic right tail, preventing test-set legitimate heavy downloads from falling beyond the synthetic empirical boundary.
4. **Closed-Loop Physical Law Projection:**
   Synthetic TCP records are projected onto physical network invariants:
   $$\text{tcprtt} = \text{synack} + \text{ackdat}$$
   $$\text{rate} \le \frac{\text{sbytes} + \text{dbytes}}{\max(\text{dur}, 10^{-6})}$$
5. **Anti-Shakeup Bagging (5 Seeds):**
   To guard against leaderboard shakeup between Public (30%) and Private (70%) test sets, 5 independent copula instances are sampled with varying seeds and aggregated into the final 70,000-row corpus.

#### B. NSL-KDD: Controlled Partial Balancing v2 (`stratified_cpb_v2.py`)
1. **Dirichlet/Uniform Prior Smoothing:**
   Category probabilities $p_k$ across attack classes and protocol types are smoothed toward a uniform prior $u_k = 1/K$:
   $$q_k = (1 - \lambda) p_k + \lambda u_k, \quad \text{with } \lambda = 0.30$$
2. **Exact Zero-Dominance & Integer Snapping:**
   Host-based security counters (`num_compromised`, `su_attempted`, `root_shell`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`) are strictly snapped to non-negative integers. Records with negligible values ($< 10^{-4}$) are forced to absolute zero, preserving the steep empirical spike at zero that ECOD requires for near-perfect anomaly detection.
3. **5-Seed Bootstrap Aggregation:**
   Ensembled across 5 bootstrap resamples to produce exactly 40,000 rows.

---

## 3. Comprehensive Experimental Comparison & Leaderboard Progression

| Experiment / Approach | Architecture / Strategy | NSL-KDD Proxy $S$ | NSL-KDD Kaggle LB | UNSW Proxy $S$ | UNSW Kaggle LB | Key Analytical Lesson |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Raw Normal)** | Direct resampling of normal training rows | 0.8912 | ~0.8950 | 0.4776 | 0.42100 | Severe volume false alarms on UNSW; discrete collapse. |
| **Vadiraj Solution** | Controlled Partial Balancing (CPB alone) | **0.9310** | **0.92993** | 0.4420 | 0.43872 | Excellent on discrete counters; fails on continuous UNSW volume tails. |
| **Aayush Phase 7** | Contrastive DVS-Copula (Pure) | 0.8940 | ~0.8990 | **0.5870** | **0.53307** | CLR + DVS $\alpha=35.0$ successfully solved UNSW volume false alarms. |
| **Experiment 2** | Naive 50/50 Concatenation (Proposal Eq. 5.2) | 0.9260 | 0.92492 | 0.5140 | 0.51789 | Blending real rows re-introduced empirical volume ceilings on UNSW. |
| **Experiment 3** | Heuristic Pruning + Clamping | **0.9290** | **0.92885** | 0.4980 | 0.49966 | Hard dropping `state=='INT'` caused WAN extinction (regression). |
| **Interim Track 2 Dip** | DVS-Copula without CLR Purification | 0.9280 | — | 0.4919 | — | Omitting CLR collapsed IForest AUPRC to 0.4784. |
| **Experiment 4 (Track 2 Restored)** | **Restored Contrastive DVS-Copula v2 + CPB v2** | **0.9290** | **Pending** | **0.5888** | **Pending** | **Restores CLR + 5-seed bagging. Surpasses Phase 7 benchmark.** |

### Detailed Breakdown of Experiment 4 Local Proxy Scores
- **NSL-KDD:**
  - $\text{AUPRC}_{\text{ECOD}} = 0.8997$
  - $\text{AUPRC}_{\text{IForest}} = 0.9583$
  - **Composite $S = \frac{1}{2}(0.8997 + 0.9583) = \mathbf{0.9290}$**
- **UNSW-NB15:**
  - $\text{AUPRC}_{\text{ECOD}} = 0.5384$
  - $\text{AUPRC}_{\text{IForest}} = 0.6392$
  - **Composite $S = \frac{1}{2}(0.5384 + 0.6392) = \mathbf{0.5888}$**

---

## 4. Track 1 & Slurm One-Shot Pipeline (NUS SoC GPU Cluster)

To ensure that running on the cluster does not require tedious trial-and-error, `run_cluster.sh` is architected as an **All-In-One, Single-Pass Pipeline**. In one execution, it runs both tracks, computes all local holdout proxy benchmarks, generates and verifies all submission CSVs, saves model weights, and outputs a master comparison table.

### 4.1 What `run_cluster.sh` Does Step-by-Step
1. **Resource Allocation, Email Notification & Bootstrapping (Stage 0):**
   - Configured via Slurm directives (`#SBATCH`) for 1x GPU (A100 / V100 / RTX 3090 / L40S), 8 CPU cores, 32GB RAM, 4-hour max walltime.
   - Configured automated email notifications to `aayush_majumdar@u.nus.edu` on job start, completion, and failure (`--mail-type=BEGIN,END,FAIL`).
   - Detects the Python environment (Conda `cs5344` or active venv) and installs dependencies from `requirements_gpu.txt` quietly.
2. **Fast Track 2 Execution (Stage 1 — ~3 mins on CPU):**
   - Invokes `benchmark_expt4.py` using the 8 allocated CPU cores.
   - Evaluates holdout proxy scores for Controlled Partial Balancing v2 (NSL-KDD) and Contrastive DVS-Copula v2 (UNSW-NB15).
   - Generates the primary submission files: `submission_nslkdd.csv` (40k rows) and `submission_unsw.csv` (70k rows).
3. **Deep Generative Track 1 Execution (Stage 2 — ~1.5 hours on GPU):**
   - Invokes `train_tabsyn_diffusion.py` on the GPU.
   - Trains Tabular VAE (Embeddings + Continuous Normalization with Sinkhorn/ELBO Loss) and Latent DDPM (Score-based reverse diffusion over 1000 timesteps) on both datasets.
   - Evaluates holdout proxy scores (ECOD + 10-seed Isolation Forest) and logs $\text{AUPRC}_{\text{ECOD}}$, $\text{AUPRC}_{\text{IForest}}$, and Composite $S$.
   - Saves serialized PyTorch checkpoints (`checkpoints/tabsyn_nslkdd.pt` and `checkpoints/tabsyn_unsw.pt`) so models can be re-sampled without re-training.
   - Generates diffusion submissions: `submission_nslkdd_tabsyn.csv` (40k rows) and `submission_unsw_tabsyn.csv` (70k rows).
4. **Master Benchmark Compilation & Automated Integrity Audit (Stage 3):**
   - Automatically merges both benchmark runs into `master_benchmark_summary.csv` and prints a formatted side-by-side table in the Slurm log.
   - Asserts integrity for all 4 submission CSVs: exact row counts (40,000 / 70,000), exact column counts (42 / 43), zero NaNs, zero Infs, `id` column at index 0, and absence of `is_anomaly`.
   - Synchronizes verified submission files to the root `submissions/` directory, ready for immediate Kaggle submission.

### 4.2 How to Submit the Job
On the NUS SoC cluster:
```bash
cd expt-4/gpu_cluster
sbatch run_cluster.sh
```
To monitor the job:
```bash
squeue -u $USER
tail -f slurm_*.out
```

### 4.3 Slurm Environment Hardening & Forensic Fixes (Job 917948 Audit)
1. **Working Directory Resolution (`$SLURM_SUBMIT_DIR`)**:
   Slurm copies batch scripts to a local spool directory (`/var/lib/slurm/slurmd/job<ID>/slurm_script`). Relying on `BASH_SOURCE[0]` redirected the working directory away from the project repository. `run_cluster.sh` now dynamically binds to `$SLURM_SUBMIT_DIR` and traverses into `expt-4/gpu_cluster`.
2. **Defensive `nvidia-smi` & Module Loading**:
   On GPU nodes where device access is subject to environment modules or driver configuration (e.g. `xgpe2`), raw `nvidia-smi` returned exit code 6 ("No devices were found"), which triggered `set -e` failure. `run_cluster.sh` now loads `cuda` and `python` modules, wraps `nvidia-smi` with non-fatal fallbacks, inspects `CUDA_VISIBLE_DEVICES`, and probes PyTorch device availability directly.
3. **Offline Node Safety**:
   `pip install` operations are guarded to prevent failure if compute nodes lack outbound internet connectivity.

---

### 4.4 Resolution of ModuleNotFoundError & SoC GPU Topology (Job 918010 Audit)
1. **Root Cause of `ModuleNotFoundError: No module named 'sklearn'`**:
   - NUS SoC HPC compute nodes (`xgph*`, `xgpe*`, `xgpd*`) are completely offline (firewalled from outbound internet access) for cluster security.
   - When jobs are scheduled, any `pip install` executed on a compute node fails silently.
   - Without an activated virtual environment / conda environment created beforehand on login node `xlogin2`, the script falls back to `/usr/bin/python3`, which lacks third-party packages (`sklearn`, `pyod`, `torch`).
2. **Turnkey Resolution (`setup_env.sh`)**:
   - Because user home directories (`/home/a/aayusm`) reside on an NFS network-mounted filesystem shared across all login and compute nodes, installing dependencies once on `xlogin2` makes them immediately accessible to all compute jobs.
   - Created `setup_env.sh` to automate environment creation (`conda` or `.venv`), PyTorch CUDA 12.1 installation, and dependency verification.
3. **GPU Gres Resource Targeting**:
   - Querying `sinfo -p gpu -O "NodeList:12,Partition:8,Gres:20,StateCompact:8"` revealed:
     - `xgph[0,7]` with `gpu:a100-80:1` (NVIDIA A100 80GB VRAM) are **idle**.
     - `xgpd[0-1,3-7]` with `gpu:nv:2` are **idle**.
     - Generic `--gres=gpu:1` previously routed jobs to `xgpe2` (an unconfigured node with exit code 6).
   - Configured `#SBATCH --gres=gpu:a100-80:1` to directly allocate the idle 80GB A100 GPU nodes with pre-configured CUDA drivers.
   - Added pre-flight environment checks in `run_cluster.sh` to fail fast with actionable guidance if the environment has not yet been provisioned.

---

## 5. Submission Artifact Verification

All generated submission CSV files have been rigorously validated against competition integrity requirements:

| Submission File | Target Dataset | Row Count | Column Count | Primary Key | Nulls / NaNs | Infs | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `expt-4/submissions/submission_nslkdd.csv` | NSL-KDD | 40,000 | 42 | `id` (0..39999) | 0 | 0 | **Verified Valid** |
| `expt-4/submissions/submission_unsw.csv` | UNSW-NB15 | 70,000 | 43 | `id` (0..69999) | 0 | 0 | **Verified Valid** |
| `submissions/submission_nslkdd.csv` | NSL-KDD (Root) | 40,000 | 42 | `id` (0..39999) | 0 | 0 | **Verified Valid** |
| `submissions/submission_unsw.csv` | UNSW-NB15 (Root) | 70,000 | 43 | `id` (0..69999) | 0 | 0 | **Verified Valid** |

Both submission files are ready for immediate upload to the Kaggle In-Class competition.


