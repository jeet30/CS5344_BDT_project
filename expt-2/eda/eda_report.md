# Comprehensive Exploratory Data Analysis Report: Tabular Data Augmentation for Anomaly Detection (NSL-KDD & UNSW-NB15)

**Author**: Senior Data Scientist  
**Date**: September 2026  
**Project**: CS5344 Course Project — Tabular Data Augmentation for Anomaly Detection  
**Experiment**: Cycle 2 (`expt-2`)  

---

## Executive Summary

This report establishes the empirical and theoretical foundations for **Experiment 2 (`expt-2`)**. The competition task challenges participants to generate normal-only synthetic tabular datasets $\tilde{D}$ ($N = 40,000$ for NSL-KDD; $N = 70,000$ for UNSW-NB15) from labeled training sets $(D_{\text{train}}, y_{\text{train}})$ to maximize downstream anomaly detection utility on private validation and test splits:

$$S = \frac{\text{AUPRC}_{\text{ECOD}} + \text{AUPRC}_{\text{IForest}}}{2}$$

Crucially, downstream evaluation is governed by two fixed detectors—**ECOD** (an empirical cumulative distribution tail estimator) and **Isolation Forest** (an ensemble of 10 seeds $\times$ 500 trees)—evaluated on holdout sets featuring **unseen attack types**:

$$\{\text{types of abnormal in } D_{\text{train}}\} = A_{\text{seen}}$$
$$\{\text{types of abnormal in } D_{\text{valid}}\} = A_{\text{seen}} \cup A_{\text{val}}$$
$$\{\text{types of abnormal in } D_{\text{test}}\} = A_{\text{seen}} \cup A_{\text{val}} \cup A_{\text{test}}$$

### Headline Diagnostics from Experiment 1
1. **The UNSW Reversal**: On NSL-KDD, our conditional Gaussian copula generator yielded $S = 0.9266$ on the public leaderboard (closely tracking our local proxy of $0.9507$). On UNSW-NB15, however, our generator suffered a severe performance collapse: it achieved $S = 0.4744$ locally (beating the starter kit's $0.4606$), but scored only $0.4192$ on the Kaggle public leaderboard (losing to the starter kit's $0.4338$).
2. **The Mechanism**: Our copula generator strictly bounded all continuous values to the *observed, per-protocol-group* training envelope without extrapolation. As demonstrated below, ECOD and Isolation Forest penalize contracted normal boundaries severely when test-set normal flows exhibit subtle peripheral deviations or when unseen attack distributions appear. The starter kit's naive bootstrap + global jitter ($0.5 \cdot \sigma$) broke joint feature correlations but accidentally widened normal support, giving downstream detectors the slack needed to generalize.
3. **Domain Identity Inflation**: Post-generation recomputation of $\text{tcprtt} = \text{synack} + \text{ackdat}$ in Experiment 1 allowed unconstrained sums to reach $5.04$ (whereas the real training maximum was $3.18$).

This report provides complete statistical profiling, class divergence analysis, latent attack cluster modeling, and detector mechanics to guide the Experiment 2 generator architecture.

---

## 1. Dataset Schema, Quality & Data Sanitization Audit

### 1.1 Structural Comparison

| Property | Dataset 1: NSL-KDD | Dataset 2: UNSW-NB15 |
|---|---|---|
| **Raw Training Rows** | 30,000 | 50,000 |
| **Normal ($y=0$) Count** | 20,000 (66.67%) | 35,000 (70.00%) |
| **Abnormal ($y=1$) Count** | 10,000 (33.33%) | 15,000 (30.00%) |
| **Total Features ($m$)** | 41 features + label | 42 features + label |
| **Continuous Features** | 22 | 22 |
| **Integer Features** | 8 | 16 |
| **Binary Features** | 6 | 1 |
| **Categorical Features** | 3 (`protocol_type`, `service`, `flag`) | 3 (`proto`, `service`, `state`) |
| **Constant / Zero-Variance** | 2 (`num_outbound_cmds`, `is_host_login`) | 0 |
| **Missing Values (NaN)** | 0 (0.0%) | 0 (0.0%) |
| **Target Submission Size ($N$)** | **40,000** normal-only rows | **70,000** normal-only rows |

### 1.2 Data Sanitization & Normal Outlier Scrubbing
Real-world network connection logs inherently contain mislabeled connections, telemetry glitches, and edge-case anomalies within the nominal "normal" class. Training a generative model on unscrubbed normal records forces it to learn and over-generate anomalous noise modes.

We deploy an **Isolation Forest within-class purification filter**:
- Continuous features are standardized: $Z = \text{StandardScaler}(X_{\text{cont}})$.
- Isolation Forest with contamination $\alpha = 0.02$ ($2.0\%$) and 100 trees identifies peripheral anomalies.
- Exact duplicate records are subsequently pruned.

**Purification Audit Results:**
- **NSL-KDD**: Raw normal rows = $20,000$. Outliers removed = $400$ ($2.0\%$). Duplicates = $0$. Final scrubbed normal set = **$19,600$** rows.
- **UNSW-NB15**: Raw normal rows = $35,000$. Outliers removed = $700$ ($2.0\%$). Duplicates = $0$. Final scrubbed normal set = **$34,300$** rows.

Both scrubbed datasets have been exported to `expt-2/eda/outputs/` as the canonical training base for Experiment 2.

---

## 2. Statistical Profiling & Dynamic Range Nuances

### 2.1 Heavy Skewness and Kurtosis
Tabular network telemetry features do not follow Gaussian distributions. Instead, counters and byte volumes exhibit extreme right-skewness:

| Dataset | Feature | Normal Mean | Normal Median | Normal Max | Skewness | Kurtosis | Heavy-Tailed? |
|---|---|---|---|---|---|---|---|
| **NSL-KDD** | `src_bytes` | 3,115.8 | 234.0 | 89,584.0 | 18.42 | 412.8 | **Yes** |
| **NSL-KDD** | `dst_bytes` | 3,612.4 | 388.0 | 62,382.0 | 14.18 | 265.4 | **Yes** |
| **NSL-KDD** | `duration` | 21.6 | 0.0 | 4,210.0 | 16.85 | 320.1 | **Yes** |
| **UNSW-NB15** | `sbytes` | 8,924.3 | 534.0 | 14,355,774.0 | 48.91 | 3,112.5 | **Yes** |
| **UNSW-NB15** | `dbytes` | 14,892.1 | 178.0 | 14,657,531.0 | 42.15 | 2,450.8 | **Yes** |
| **UNSW-NB15** | `rate` | 68,912.4 | 2,857.1 | 1,000,000.0 | 6.84 | 58.2 | **Yes** |
| **UNSW-NB15** | `dur` | 1.15 | 0.014 | 59.99 | 5.21 | 31.8 | **Yes** |
| **UNSW-NB15** | `sload` | 3.5e+06 | 1.2e+04 | 5.2e+09 | 24.30 | 780.4 | **Yes** |

**Mathematical Implication**: Direct linear modeling or isotropic Gaussian sampling on raw byte and rate features produces catastrophic distortion (e.g., negative bytes or massive density over-allocation in the right tail). All continuous heavy-tailed variables must be modeled in logarithmic space:
$$x_{\text{transformed}} = \log(1 + x), \quad x_{\text{generated}} = \max(0, \exp(\tilde{x}) - 1)$$

### 2.2 Zero-Inflation Profiling
A significant subset of features represent conditional event flags or error counters that are **identically 0** for the vast majority of normal sessions:
- NSL-KDD: `serror_rate` (98.2% zeros in normal), `rerror_rate` (97.4% zeros), `num_failed_logins` (99.98% zeros), `urgent` (100.0% zeros), `wrong_fragment` (100.0% zeros).
- UNSW-NB15: `sloss` (74.2% zeros), `dloss` (78.5% zeros), `sinpkt` (62.1% zeros), `trans_depth` (91.4% zeros), `ct_ftp_cmd` (98.9% zeros).

**Mathematical Implication**: Continuous generative densities cannot naturally place point-mass probability at exactly $0.0$. Attempting to fit a continuous density to a zero-inflated feature causes synthetic values to disperse into small positive fractions (e.g., $0.0031$), which detectors flag as abnormal. Zero-inflation must be explicitly preserved via two-stage hurdle modeling or empirical thresholding:
$$\tilde{x} = \begin{cases} 0 & \text{with probability } P(x = 0) \\ \text{sample from } f(x \mid x > 0) & \text{otherwise} \end{cases}$$

---

## 3. Anomaly Boundary & Class Divergence Analysis

To determine how normal traffic is differentiated from attacks, we conducted two-sample Kolmogorov-Smirnov (KS) tests between the normal ($y=0$) and abnormal ($y=1$) subsets across all numerical features.

### 3.1 Top Sentinel Features (Maximum Separation)

#### NSL-KDD Sentinels
1. `src_bytes` ($\text{KS} = 0.8574, p < 10^{-300}$): Normal median = $234.0$, Abnormal median = $0.0$.
2. `dst_bytes` ($\text{KS} = 0.8153, p < 10^{-300}$): Normal median = $388.0$, Abnormal median = $0.0$.
3. `same_srv_rate` ($\text{KS} = 0.7343, p < 10^{-300}$): Normal median = $1.00$, Abnormal median = $0.08$.
4. `diff_srv_rate` ($\text{KS} = 0.7324, p < 10^{-300}$): Normal median = $0.00$, Abnormal median = $0.06$.
5. `dst_host_same_srv_rate` ($\text{KS} = 0.7161, p < 10^{-300}$): Normal median = $1.00$, Abnormal median = $0.05$.

*Domain Insight*: In NSL-KDD, attacks are predominantly SYN-floods and port scans where the 3-way handshake is never completed. Consequently, `src_bytes` and `dst_bytes` are exactly $0$, while normal traffic consists of established connections with bidirectional payload exchange.

#### UNSW-NB15 Sentinels
1. `sttl` ($\text{KS} = 0.6581, p < 10^{-300}$): Normal median = $31.0$, Abnormal median = $254.0$.
2. `ct_state_ttl` ($\text{KS} = 0.6471, p < 10^{-300}$): Normal median = $0.0$, Abnormal median = $1.0$.
3. `dttl` ($\text{KS} = 0.5087, p < 10^{-300}$): Normal median = $29.0$, Abnormal median = $252.0$.
4. `ackdat` ($\text{KS} = 0.4985, p < 10^{-300}$): Normal median = $0.0$, Abnormal median = $0.054$.
5. `tcprtt` ($\text{KS} = 0.4972, p < 10^{-300}$): Normal median = $0.0$, Abnormal median = $0.102$.

*Domain Insight*: In UNSW-NB15, Time-To-Live (`sttl`, `dttl`) is the single strongest discriminator. Normal internal university network traffic originates with TTL=32 or TTL=64, whereas external attacks injected via IXIA PerfectStorm enter with default TTL=255.

---

## 4. Latent Attack Archetypes (The Pseudo-Novelty Foundation)

A critical requirement of the competition design is that the private validation and test sets contain attack types never seen during training ($A_{\text{val}}$ and $A_{\text{test}}$). Evaluating candidates purely on a random 80/20 train/validation split creates a fatal blind spot because both splits share the same attack distribution $A_{\text{seen}}$.

To overcome this, we clustered the $y=1$ abnormal training records into $K=4$ distinct clusters in standardized PCA space, identifying natural attack archetypes within $A_{\text{seen}}$:

### 4.1 NSL-KDD Attack Clusters

| Cluster | Archetype | Count | % Abnormal | Key Signatures | Top Service / Flag |
|---|---|---|---|---|---|
| **0** | SYN-Flood / DoS | 6,009 | 60.09% | `serror_rate` = 0.99, `src_bytes` = 0, `count` = 125 | `private` / `S0` |
| **1** | Rejection Scan / Port Sweep | 2,230 | 22.30% | `rerror_rate` = 0.94, `diff_srv_rate` = 0.42 | `private` / `REJ` |
| **2** | High-Volume Flood | 1,011 | 10.11% | `count` = 312, `srv_count` = 305 | `ecr_i` / `SF` |
| **3** | Stealth / Exploit / Content | 750 | 7.50% | `duration` = 124s, `src_bytes` = 4,210, `hot` > 0 | `smtp` / `SF` |

### 4.2 UNSW-NB15 Attack Clusters

| Cluster | Archetype | Count | % Abnormal | Key Signatures | Top Proto / State |
|---|---|---|---|---|---|
| **0** | High-Rate DoS / Volumetric | 3,592 | 23.95% | `rate` = 94,510 pkts/s, `dur` = 0.008s | `udp` / `INT` |
| **1** | Remote Exploit (Interactive) | 31 | 0.21% | `sbytes` = 1.2 MB, `dur` = 38.4s | `tcp` / `FIN` |
| **2** | Low-and-Slow Attack | 12 | 0.08% | `dur` = 58.2s, `rate` = 0.12 pkts/s | `tcp` / `FIN` |
| **3** | Remote Infiltration / Probe | 11,365 | 75.77% | `sttl` = 254, `ackdat` = 0.054s | `tcp` / `FIN` |

### 4.3 The Pseudo-Novelty Validation Harness
With these archetypes isolated, our **Track B Evaluator** can perform Leave-One-Attack-Cluster-Out validation:
1. Train candidate generative model $\mathcal{G}$ on scrubbed normal data.
2. Evaluate downstream ECOD and Isolation Forest on an evaluation set where one complete attack archetype (e.g. Cluster 0 or Cluster 3) is withheld during parameter selection and tested as a "pseudo-novel" attack.
3. This directly measures whether our synthetic normal distribution generalizes to unseen attack manifolds!

---

## 5. Mechanical Analysis of Downstream Detectors & The UNSW Reversal

### 5.1 ECOD Mechanics
ECOD (Empirical Cumulative Distribution Functions for Outlier Detection) estimates tail probabilities non-parametrically across each dimension independently:
$$O_i = \sum_{d=1}^m -\log \left( \min \left( \hat{F}_d(x_{id}), 1 - \hat{F}_d(x_{id}) \right) \right)$$

- If the synthetic normal dataset $\tilde{D}$ strictly binds to the empirical training minimum and maximum $[x_d^{\min}, x_d^{\max}]$, $\hat{F}_d$ evaluates to extreme bounds ($0$ or $1$) for any test normal flow whose duration, rate, or byte count exceeds the training maximum even slightly.
- ECOD assigns maximum outlier scores ($-\log \epsilon \approx 36$) to these valid normal test instances, generating massive false positives and destroying AUPRC.

### 5.2 Isolation Forest Mechanics
Isolation Forest isolates observations by randomly selecting a feature and randomly selecting a split value between the minimum and maximum of that feature:
$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$

- When synthetic data is tightly contracted around training modes, the peripheral bounding box collapses inward.
- Real test normal points falling into the periphery require very few random cuts to be isolated, resulting in very short path lengths $h(x)$ and high anomaly scores.

### 5.3 Why the Starter Kit's Jitter Won on UNSW-NB15
The starter kit applied isotropic Gaussian noise:
$$\tilde{x}_d = x_d + \mathcal{N}(0, 0.5 \cdot \sigma_d)$$

While theoretically flawed (it distorted protocol correlations and breached physical identities), the $0.5 \cdot \sigma$ global noise **expanded the convex hull** of the normal data in all directions. When the real UNSW validation set was evaluated on Kaggle, the starter kit's expanded support prevented ECOD and Isolation Forest from misclassifying peripheral normal points!

**The Senior Data Scientist Solution for Experiment 2**:
We do not adopt crude isotropic jitter. Instead, we introduce **controlled boundary slackness and tail expansion** into our conditional copula pipeline:
1. Fit the Gaussian copula on rank-transformed uniform marginals to strictly preserve non-linear joint dependency structures.
2. During inverse transform, apply an adaptive tail expansion factor $\gamma \in [1.05, 1.20]$ to continuous unbounded features (or fit Generalized Pareto tails).
3. This provides downstream detectors with the required support slackness while maintaining high fidelity and strict protocol invariants.

---

## 6. Physical & Network Protocol Domain Constraints

All generated samples must satisfy strict physical and protocol invariants:

### 6.1 NSL-KDD Invariants
1. **Rate Bounds**: All 14 rate features (`serror_rate`, `srv_serror_rate`, `same_srv_rate`, etc.) must be strictly in $[0.0, 1.0]$.
2. **Non-Negativity**: `duration`, `src_bytes`, `dst_bytes`, `count`, `srv_count`, etc. $\ge 0$.
3. **Binary Indicators**: `land`, `logged_in`, `root_shell`, `su_attempted`, `is_guest_login` $\in \{0, 1\}$.
4. **Logical Consistency**: If `logged_in == 0`, then `root_shell = 0` and `su_attempted = 0`.
5. **Constant Columns**: `num_outbound_cmds = 0` and `is_host_login = 0`.

### 6.2 UNSW-NB15 Invariants
1. **Physical Non-Negativity**: `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sload`, `dload`, `sinpkt`, `dinpkt`, `tcprtt`, `synack`, `ackdat` $\ge 0.0$.
2. **TTL Bounding**: `sttl`, `dttl` $\in [0, 255]$ (integer).
3. **Protocol Zeroing**: For any connection where `proto != 'tcp'`, the following TCP handshake fields must be strictly zero:
   $$\text{swin} = 0, \quad \text{dwin} = 0, \quad \text{stcpb} = 0, \quad \text{dtcpb} = 0, \quad \text{tcprtt} = 0, \quad \text{synack} = 0, \quad \text{ackdat} = 0$$
4. **The `tcprtt` Exact Identity & Bounding**:
   $$\text{tcprtt} = \text{synack} + \text{ackdat}$$
   Furthermore, to fix the Experiment 1 inflation bug, if $\text{synack} + \text{ackdat} > 3.18$ (the maximum observed normal training value), both components are scaled down proportionately before summation:
   $$\text{synack} \leftarrow \text{synack} \cdot \frac{3.18}{\text{synack} + \text{ackdat}}, \quad \text{ackdat} \leftarrow \text{ackdat} \cdot \frac{3.18}{\text{synack} + \text{ackdat}}$$

---

## 7. Actionable Architectural Directives for Experiment 2

Based on this comprehensive analysis, the Experiment 2 pipeline will execute according to the following directives:

1. **Scrubbed Normal Foundation**: Train all generators strictly on `outputs/nslkdd_scrubbed_normal.csv` ($N=19,600$) and `outputs/unsw_scrubbed_normal.csv` ($N=34,300$).
2. **Dual-Track Evaluation**: Score every generator on both the Track A standard proxy ($S$) and the Track B pseudo-novelty stress-test ($S_{\text{pseudo}}$) before making any Kaggle submission.
3. **Controlled Boundary Slackness**: Implement parametric tail expansion on the Gaussian copula to systematically eliminate the UNSW generalization gap.
4. **Deterministic Invariant Enforcement**: Post-process all synthetic batches through `expt-2/src/constraints.py` to ensure $0$ identity or protocol violations.

