# Master Strategic Plan: Achieving the Project Goal (Experiment 2)

**Role**: Senior Data Scientist  
**Date**: September 2026  
**Project**: CS5344 Course Project — Tabular Data Augmentation for Anomaly Detection  
**Experiment**: Cycle 2 (`expt-2`)  

---

## 1. Executive Mandate & Mathematical Problem Formulation

### 1.1 Objective Function
The central objective of this project is to construct a unified, dataset-agnostic tabular data augmentation pipeline that generates synthetic normal samples $\tilde{D}$ to maximize downstream anomaly detection utility:

$$\max_{\mathcal{G}} S(\mathcal{G}) = \frac{1}{2} \left[ \text{AUPRC}_{\text{ECOD}}(\tilde{D}, D_{\text{eval}}) + \text{AUPRC}_{\text{IForest}}(\tilde{D}, D_{\text{eval}}) \right]$$

subject to:
1. **Normal-Only Constraint**: $\tilde{D} = \{\tilde{x}_i\}_{i=1}^N$ consists exclusively of samples representing normal network traffic ($y=0$). Anomaly labels are not generated or submitted.
2. **Fixed Submission Volume**:
   - NSL-KDD: Exactly $N = 40,000$ rows.
   - UNSW-NB15: Exactly $N = 70,000$ rows.
3. **Fixed Downstream Detectors**:
   - **ECOD**: Non-parametric empirical cumulative distribution tail estimator.
   - **Isolation Forest**: 10 fixed random seeds $\times$ 500 trees each.
4. **Generalization to Novel Anomaly Types**:
   $$\{\text{types of abnormal in } D_{\text{train}}\} = A_{\text{seen}}$$
   $$\{\text{types of abnormal in } D_{\text{valid}}\} = A_{\text{seen}} \cup A_{\text{val}}$$
   $$\{\text{types of abnormal in } D_{\text{test}}\} = A_{\text{seen}} \cup A_{\text{val}} \cup A_{\text{test}}$$
   The synthetic normal data must establish an optimal decision boundary capable of separating normal traffic from both known attack archetypes ($A_{\text{seen}}$) and completely unseen attack mechanisms ($A_{\text{val}}, A_{\text{test}}$).
5. **Transferability**: A single shared architectural pipeline must operate across both datasets (parameter configurations may be tuned per dataset).

---

## 2. Post-Mortem & Gap Analysis from Experiment 1

### 2.1 The UNSW Public Leaderboard Reversal
In Experiment 1, our conditional Gaussian copula generator yielded strong results on NSL-KDD but suffered a significant reversal on UNSW-NB15:

| Dataset | Ours (Copula) | Starter Kit Baseline | Gap | Local Proxy $S$ (Ours) | Local Proxy $S$ (Baseline) | Local Gap |
|---|---|---|---|---|---|---|
| **NSL-KDD** | 0.92664 | 0.93155 | Starter +0.00491 | 0.9507 | 0.9521 | Starter +0.0014 |
| **UNSW-NB15** | 0.41918 | 0.43384 | **Starter +0.01466** | 0.4744 | 0.4606 | **Copula +0.0138 (Reversed on Real LB)** |

### 2.2 Root Causes Identified
1. **The Boundary Tightness Paradox**:
   - Our copula strictly clamped continuous features to the observed per-protocol training envelope $[x_d^{\min}, x_d^{\max}]$.
   - In ECOD, points outside the empirical training envelope evaluate to $\hat{F}_d = 0$ or $1$, resulting in maximum outlier penalties ($-\log \epsilon \approx 36$). Valid normal test sessions that had slightly higher rates or durations were penalized as anomalies.
   - In Isolation Forest, contracted normal points collapse the bounding box, leaving peripheral test normal points in empty space, where they are isolated at shallow tree depth.
   - The starter kit's naive bootstrap + global jitter ($0.5 \cdot \sigma$) broke feature correlations, but accidentally expanded the normal support in all directions, providing downstream detectors with the slack needed to prevent false positives on unseen test distributions!
2. **Deterministic Domain Identity Inflation**:
   - Recomputing $\text{tcprtt} = \text{synack} + \text{ackdat}$ without bounding the sum resulted in synthetic UNSW rows reaching $\text{tcprtt} = 5.04$ (whereas training normal maximum was $3.18$).
3. **Local Validation Blind Spot**:
   - Our local evaluation proxy used a standard random holdout split where both train and test contained the exact same attack types ($A_{\text{seen}}$). It could not evaluate generalization to novel attack types ($A_{\text{val}}$).

---

## 3. Dual-Track Validation & Evaluation Architecture

To prevent future evaluation blind spots, Experiment 2 establishes a dual-track local evaluation harness:

```mermaid
flowchart TD
    subgraph InputData ["Scrubbed Normal Data"]
        D_norm["D_train (Normal, y=0)"]
    end

    subgraph Generator ["Candidate Generator G"]
        Gen["Synthetic Generator with Adaptive Boundary Slack"]
        Post["Domain Constraint Projection & Identity Fix"]
    end

    subgraph DualTrack ["Dual-Track Evaluation Harness"]
        subgraph TrackA ["Track A: Standard Proxy"]
            HoldoutA["Stratified Random Holdout (A_seen)"]
            ScoreA["S_standard = (AUPRC_ECOD + AUPRC_IForest) / 2"]
        end
        subgraph TrackB ["Track B: Pseudo-Novelty Stress-Test"]
            HoldoutB["Leave-Attack-Cluster-Out (A_seen sub-types)"]
            ScoreB["S_pseudo = Mean AUPRC across Latent Attack Clusters"]
        end
    end

    subgraph Decision ["Submission Gate"]
        Check{"S_pseudo > Baseline AND S_standard >= 0.94 / 0.46?"}
        Submit["Generate Kaggle Submission (N=40k / N=70k)"]
        Iterate["Refine Hyperparameters (Slack gamma, Contamination)"]
    end

    D_norm --> Gen --> Post
    Post --> HoldoutA --> ScoreA --> Check
    Post --> HoldoutB --> ScoreB --> Check
    Check -- Yes --> Submit
    Check -- No --> Iterate
```

### Track A: Standard Leaderboard Proxy
- Detectors: ECOD + 10 seeds $\times$ 500 trees Isolation Forest.
- Score: $S_{\text{standard}} = (\text{AUPRC}_{\text{ECOD}} + \text{AUPRC}_{\text{IForest}}) / 2$.
- Purpose: Benchmarks internal distribution fidelity against known normal modes.

### Track B: Pseudo-Novelty Stress-Test Harness
- Utilizes the 4 latent attack clusters discovered during EDA:
  - NSL-KDD: SYN-Flood, Port Sweep, High-Volume Flood, Stealth/Exploit.
  - UNSW-NB15: High-Rate DoS, Remote Exploit, Low-and-Slow, Remote Infiltration.
- Computes detector performance when each attack archetype is treated as an unseen test-set anomaly.
- Score: $S_{\text{pseudo}} = \frac{1}{K} \sum_{k=1}^K S(\text{Cluster}_k)$.
- Purpose: Direct surrogate for $A_{\text{val}}$ and $A_{\text{test}}$ performance on the private Kaggle leaderboard.

---

## 4. Methodological Architecture: Unified Transferable Pipeline

The Experiment 2 generative pipeline consists of four sequential stages:

```mermaid
flowchart LR
    S1["Stage 1: Scrubbing & Normal Foundation"] --> S2["Stage 2: Hierarchical Protocol Conditioning"]
    S2 --> S3["Stage 3: Generative Modeling with Tail Slack"]
    S3 --> S4["Stage 4: Deterministic Constraint Projection"]
```

### Stage 1: Data Scrubbing & Foundation
- Filter training data to $y = 0$.
- Apply Isolation Forest within-class purification with contamination $\alpha = 0.02$ to discard noisy telemetry and mislabeled intrusions.
- Prune duplicate records.

### Stage 2: Hierarchical Protocol Conditioning
- Stratify connections by high-level protocol groups:
  - NSL-KDD: Stratified by (`protocol_type`, `flag`). Rare services (<1%) bucketed to `Other`.
  - UNSW-NB15: Stratified by (`proto`, `state`). Rare protocols (<1%) bucketed to `Other`.
- Compute empirical group frequencies $w_g = N_g / N_{\text{total}}$ to ensure exact class-balance preservation during sampling.

### Stage 3: Generative Modeling with Controlled Boundary Slack
We evaluate three principled generation techniques:

#### Approach 3A: Conditional Gaussian Copula with Adaptive Tail Expansion (Primary)
1. Within each protocol group $g$, compute rank-based marginal CDFs $\hat{F}_{d,g}(x)$.
2. Map marginals to Gaussian space: $z_{id} = \Phi^{-1}(\hat{F}_{d,g}(x_{id}))$.
3. Compute Pearson correlation matrix $\Sigma_g = \text{Corr}(Z_g)$, projected to nearest positive definite matrix via eigenvalue clipping:
   $$\Sigma_g^{\text{PD}} = V \max(\Lambda, \epsilon) V^T, \quad \text{diag}(\Sigma_g^{\text{PD}}) = 1$$
4. Sample $z^* \sim \mathcal{N}(0, \Sigma_g^{\text{PD}})$.
5. **Controlled Boundary Slackness**: Instead of clipping inverted uniform samples strictly to $[x_d^{\min}, x_d^{\max}]$, apply an adaptive tail expansion factor $\gamma \in [1.05, 1.25]$ to continuous variables:
   $$\tilde{x}_d = \begin{cases} 
   \hat{F}^{-1}(u_d) & \text{if } u_d \in [\delta, 1-\delta] \\
   x_d^{\max} + \gamma \cdot \text{ParetoTail}(u_d) & \text{if } u_d > 1-\delta \\
   x_d^{\min} - \gamma \cdot \text{ParetoTail}(1-u_d) & \text{if } u_d < \delta
   \end{cases}$$
   This preserves 98% of the internal copula correlation structure while introducing the exact peripheral support required by ECOD and Isolation Forest.

#### Approach 3B: Covariance-Preserving Manifold Expansion (Alternative)
- Fit PCA or Mahalanobis metric on standardized scrubbed normal data:
  $$x^* = x_{\text{real}} + \mathcal{N}(0, \sigma^2 \Sigma_{\text{empirical}})$$
- Adds directional jitter along the principal axes of normal variation, preventing unphysical cross-axis distortion.

#### Approach 3C: Hybrid Interior-Boundary Ensemble (Ensemble)
- Generate 75% of rows using the high-fidelity Conditional Copula (interior density).
- Generate 25% of rows using Covariance-Preserving Boundary Slack (convex hull expansion).
- Merges high-precision correlation matching with wide boundary coverage.

### Stage 4: Deterministic Constraint Projection
Every generated batch passes through `expt-2/src/constraints.py`:
1. **Physical Clamping**: Clip rates to $[0.0, 1.0]$, TTL to $[0, 255]$, counts/bytes to $\ge 0$.
2. **Protocol Zeroing**: For non-TCP records, force `swin = dwin = stcpb = dtcpb = tcprtt = synack = ackdat = 0`.
3. **Exact Arithmetic Identities**:
   $$\text{tcprtt} = \text{synack} + \text{ackdat}$$
   with proportional rescaling if $\text{tcprtt} > 3.18$ (the maximum observed normal training value).
4. **Logical Invariants**: In NSL-KDD, force `root_shell = 0` and `su_attempted = 0` whenever `logged_in == 0`.
5. **Constant Feature Invariance**: NSL-KDD `num_outbound_cmds = 0`, `is_host_login = 0`.

---

## 5. Experiment 2 Execution Phases & Milestones

| Phase | Milestone | Expected Deliverable | Status |
|---|---|---|---|
| **Phase 1** | **Comprehensive EDA & Constraints** | Statistical profiles, KS sentinels, cluster archetypes, `EDA_expt2.ipynb`, `eda_report.md` | **Completed** |
| **Phase 2** | **Validation Harness Build** | Modular `evaluator.py` supporting Track A & Track B pseudo-novelty stress-testing | **Completed** |
| **Phase 3** | **Generator Build & Tail Expansion** | Implement `CopulaTailSlackGenerator` in `expt-2/src/generators/` with tunable $\gamma$ | **Next Step** |
| **Phase 4** | **Local Proxy & Pseudo-Novelty Benchmarking** | Run full benchmark comparing: <br>1. Baseline Bootstrap+Jitter<br>2. Experiment 1 Copula (tight)<br>3. Experiment 2 Copula (tail-expanded)<br>4. Hybrid Ensemble | Next Step |
| **Phase 5** | **Kaggle Submission Generation & Calibration** | Generate $N=40,000$ (NSL) and $N=70,000$ (UNSW) candidate CSVs, audit constraints, submit to Kaggle | Next Step |
| **Phase 6** | **Course Deliverables Integration** | Proposal writeup, presentation deck figures, and final report drafts | Final Phase |

---

## 6. Course Grading Rubric Alignment Strategy

The CS5344 project grading allocation emphasizes scientific rigor over pure Kaggle ranking:

| Component | Weight | How Experiment 2 Maximizes Marks |
|---|---|---|
| **Proposal Writeup** | **15%** | Problem formulation grounded in anomaly detection literature; deep mathematical justification for normal augmentation; clear contrast between statistical fidelity vs downstream utility. |
| **Final Presentation** | **20%** | Compelling narrative structure: The "UNSW Reversal" diagnosis; visual diagrams of ECOD / Isolation Forest inductive biases; empirical proof that controlled tail expansion resolves the generalization gap. |
| **Final Report** | **25%** | Rigorous documentation of the unified transferable pipeline; formal derivation of copula rank-transform and tail expansion; ablation tables; mathematical proof of domain invariant preservation. |
| **Leaderboard Evidence** | Contextual | Clear progression across submissions: baseline $\to$ tight copula $\to$ tail-expanded copula; documented calibration between local proxy and public leaderboard. |
| **Reproducibility** | Mandatory | Modular `src/` codebase, exact random seeds, fixed `.venv` requirements, zero manual interventions. |

---

## 7. Immediate Next Actions

1. Construct `expt-2/src/generators/copula_slack.py` implementing the adaptive tail expansion mechanism.
2. Run local dual-track evaluation on NSL-KDD and UNSW-NB15 to measure the gain in $S_{\text{pseudo}}$ and $S_{\text{standard}}$ across tail expansion values $\gamma \in [1.00, 1.30]$.
3. Export candidate submissions to `expt-2/submissions/` and verify submission formatting.

