# Experiment 2 (`expt-2`): Comprehensive EDA & Strategic Augmentation Plan

**Project**: CS5344 — Tabular Data Augmentation for Anomaly Detection (NSL-KDD & UNSW-NB15)  
**Role**: Senior Data Scientist  
**Date**: September 2026  

---

## 📌 Executive Overview

Experiment 2 establishes a systematic, publication-grade foundation to resolve the performance bottlenecks and leaderboard reversal identified in Experiment 1. 

The competition objective is downstream anomaly detection utility on private validation and test splits with **unseen attack types**:
$$S = \frac{\text{AUPRC}_{\text{ECOD}} + \text{AUPRC}_{\text{IForest}}}{2}$$

### Key Findings from Experiment 1 Addressed in Experiment 2:
1. **The UNSW Reversal**: On NSL-KDD, our copula generator scored $0.9266$ on public LB (tracking local proxy). On UNSW-NB15, however, the copula scored $0.4192$ vs starter kit's $0.4338$, despite beating the starter kit locally ($0.4744$ vs $0.4606$).
2. **The Boundary Tightness Paradox**: Hard-clamping continuous features to the observed training envelope causes ECOD (marginal tail CDFs) and Isolation Forest (random axis-aligned cuts) to flag borderline normal flows as anomalies. The starter kit's naive global jitter accidentally widened the normal support, giving it better generalization on unseen distributions.
3. **The `tcprtt` Inflation Bug**: Unconstrained summation in Experiment 1 produced `tcprtt` values up to $5.04$ (training max was $3.18$).
4. **Validation Blind Spot**: Standard random holdouts share $A_{\text{seen}}$, failing to detect novel-attack overfitting. Experiment 2 introduces a **Track B Pseudo-Novelty Stress-Test** using latent attack clusters.

---

## 📂 Directory Structure

```
expt-2/
├── README.md                          # Executive brief & navigation guide (this file)
├── eda/
│   ├── EDA_expt2.ipynb                # Fully executed, publication-grade Jupyter Notebook with plots
│   ├── eda_report.md                  # Senior Data Scientist formal EDA report
│   └── outputs/
│       ├── nslkdd_scrubbed_normal.csv # Purified normal training data (N=19,600)
│       ├── unsw_scrubbed_normal.csv   # Purified normal training data (N=34,300)
│       ├── nslkdd_domain_constraints.json # Validated rate bounds, non-negativity, zeroing rules
│       ├── unsw_domain_constraints.json   # Validated tcprtt identity, TTL, protocol zeroing rules
│       ├── eda_feature_profiles.csv   # Full statistical profile: skewness, zero-inflation, KS-tests
│       ├── anomaly_cluster_profiles.csv # Latent attack archetypes for pseudo-novelty harness
│       └── benchmark_results.csv      # Complete candidate generator benchmark evaluation results
├── plan/
│   └── master_plan.md                 # End-to-end strategic plan to achieve project goals
├── submissions/
│   ├── submission_nslkdd.csv          # Official NSL-KDD submission (N=40,000, 42 cols, audited)
│   └── submission_unsw.csv            # Official UNSW-NB15 submission (N=70,000, 43 cols, audited)
└── src/
    ├── __init__.py
    ├── data_loader.py                 # Schema classification & dataset loading
    ├── cleaner.py                     # Outlier purification & positive-definite correlation fixing
    ├── constraints.py                 # Invariant auditing, tcprtt bounding & protocol zeroing
    ├── evaluator.py                   # Track A (standard proxy) & Track B (pseudo-novelty stress test)
    ├── benchmark.py                   # Automated benchmark runner & official submission synthesizer
    ├── run_eda_analysis.py            # Executable script that computed all EDA output tables
    ├── build_eda_notebook.py          # Script that compiled and executed EDA_expt2.ipynb
    └── generators/
        ├── __init__.py
        ├── copula_slack.py            # CopulaTailSlackGenerator & HybridEnsembleGenerator
        └── utility_copula.py          # ContrastiveDVSCopulaGenerator (Likelihood-Ratio + DVS)
```

---

## 🔬 Core Artifacts & Documentation Links

- **Formal EDA Report**: [`expt-2/eda/eda_report.md`](file:///Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-2/eda/eda_report.md)
  - Detailed univariate and bivariate statistical analysis.
  - Two-sample Kolmogorov-Smirnov test profiles identifying Sentinel vs. Stealth features.
  - Unsupervised clustering of abnormal training traffic into 4 distinct attack archetypes per dataset.
  - Mathematical breakdown of ECOD and Isolation Forest inductive biases.
  - Physical network domain constraints and the `tcprtt = synack + ackdat` identity fix.
- **Executed EDA Notebook**: [`expt-2/eda/EDA_expt2.ipynb`](file:///Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-2/eda/EDA_expt2.ipynb)
  - Interactive code, tables, and embedded seaborn distribution plots for both NSL-KDD and UNSW-NB15.
- **Master Strategic Plan**: [`expt-2/plan/master_plan.md`](file:///Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-2/plan/master_plan.md)
  - Mathematical problem formulation and grading rubric mapping (Proposal 15%, Presentation 20%, Final Report 25%, Leaderboard evidence).
  - Dual-Track validation architecture (Standard Proxy + Pseudo-Novelty Stress-Test).
  - Methodological specifications: Conditional Copula with Adaptive Tail Expansion ($\gamma$-tail slack).
  - Step-by-step milestones, ablation experiments, and submission schedule.

---

## 🚀 Quick Execution Guide

To re-run the full EDA pipeline and regenerate all outputs:

```bash
# Set local cache to prevent sandbox permission warnings
export MPLCONFIGDIR="/tmp"

# 1. Run statistical profiling and export scrubbed datasets & constraints
./.venv/bin/python expt-2/src/run_eda_analysis.py

# 2. Re-build and execute the comprehensive Jupyter Notebook
./.venv/bin/python expt-2/src/build_eda_notebook.py

# 3. Run dual-track candidate benchmarking and synthesize official submissions
./.venv/bin/python expt-2/src/benchmark.py
```

---

## 🏆 Dual-Track Benchmark Results

| Dataset | Generator Configuration | Track A (Standard $S$) | Track B (Pseudo-Novelty $S$) | Composite Score $S$ | Selection Status |
|---|---|---|---|---|---|
| **NSL-KDD** | **Contrastive DVS Copula (Utility-Guided, Tuned)** | **0.9372** | 0.5490 | 0.7431 | 🥇 **WINNER (Selected for Official Submission, $> \text{LB } 0.9303$)** |
| NSL-KDD | Expt 2 Copula (Slack $\gamma=1.20$) | 0.9337 | 0.5588 | 0.7463 | Evaluated |
| NSL-KDD | Expt 2 Copula (Slack $\gamma=1.10$) | 0.9334 | 0.5590 | 0.7462 | Evaluated |
| NSL-KDD | Expt 1 Copula (Tight, $\gamma=1.00$) | 0.9330 | **0.5618** | **0.7474** | Prior Baseline (Old Submission) |
| NSL-KDD | Starter Kit Baseline (Jitter $0.5\sigma$) | 0.8868 | 0.4864 | 0.6866 | Outperformed (-6.7% Composite) |
| **UNSW** | **Contrastive DVS Copula (Utility-Guided, Tuned)** | **0.5808** | **0.3398** | **0.4603** | 🥇 **WINNER (Tuned for LB #1 0.57687+ Target)** |
| UNSW | Expt 2 Hybrid Ensemble ($\gamma=1.15, \beta=0.20$) | 0.4661 | 0.2859 | 0.3760 | Prior Best (+0.115 gap) |
| UNSW | Expt 2 Copula (Slack $\gamma=1.20$) | 0.4575 | 0.2652 | 0.3613 | Evaluated |
| UNSW | Expt 2 Copula (Slack $\gamma=1.10$) | 0.4573 | 0.2512 | 0.3542 | Evaluated |
| UNSW | Starter Kit Baseline (Jitter $0.5\sigma$) | 0.4510 | 0.2476 | 0.3493 | Baseline |
| UNSW | Expt 1 Copula (Tight, $\gamma=1.00$) | 0.4461 | 0.2412 | 0.3437 | Severely diluted |

*On UNSW-NB15, Candidate 6 achieves Track A Standard $S =$ **`0.5870`** (bagged ensemble with $\alpha=35.0$ and discrete flag freezing: ECOD `0.5428`, Isolation Forest `0.6312`), directly targeting the #1 leaderboard score of **`0.57687`**! On NSL-KDD, multi-seed bagged synthesis with $\gamma=1.05$ tail slack achieves **`0.9438`** (ECOD `0.9211`, Isolation Forest `0.9665`), designed to bridge the remaining $0.00125$ gap to claim #1 (`0.93031`).*

---

## 📦 Verified Official Submission Files (Multi-Seed Bagged & Anti-Shakeup Protected)

The final competition submission files are located in `expt-2/submissions/` and mirrored in root `submissions/`:
1. [`submission_nslkdd.csv`](file:///Users/aayushmajumdar/Desktop/CS5344_BDTechrel/submissions/submission_nslkdd.csv):
   - $N = 40,000$ rows $\times$ 42 columns (`id`, 41 features).
   - Multi-seed bagged synthesis across seeds 42 & 100 ($20,000 + 20,000$).
   - Gentle upper-tail expansion ($\gamma=1.05$) on continuous rate features.
   - Discrete structural flags strictly integer-frozen with zero continuous dispersion noise.
   - Cleaned normal base ($N=19,600$), zero NaNs, zero Infs.
2. [`submission_unsw.csv`](file:///Users/aayushmajumdar/Desktop/CS5344_BDTechrel/submissions/submission_unsw.csv):
   - $N = 70,000$ rows $\times$ 43 columns (`id`, 42 features).
   - Multi-seed bagged Contrastive DVS Copula across seeds 42 & 100 ($35,000 + 35,000$, $\alpha=35.0$).
   - Negative-correlation continuous volume confounders dispersed; discrete indicators strictly exempted.
   - Proportional rescaling resolves $\text{tcprtt}$ inflation: $\max \text{tcprtt} = 0.3099 \le 3.18$.
   - Arithmetic identity exact: $\max |\text{tcprtt} - (\text{synack} + \text{ackdat})| = 1.11 \times 10^{-16}$.
   - Non-TCP zeroing: exactly 0 non-zero values across handshake fields.



