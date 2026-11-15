# Group 42: Detector-Aware Data Augmentation (CS5344)

This code reproduces our two selected submissions: normal-only training tables for fixed ECOD and Isolation Forest detectors.

| Dataset | Output | Public leaderboard | Final hidden |
|---|---|---|---|
| UNSW-NB15 | `out/final_unsw.csv`, 70,000 rows | 0.61452 | 0.68508 |
| NSL-KDD | `out/final_nslkdd.csv`, 40,000 rows | 0.93734 | 0.95247 |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # tested with Python 3.14
pip install -r requirements.txt

python pipeline.py unsw        # -> out/final_unsw.csv
python pipeline.py nslkdd      # -> out/final_nslkdd.csv
python verify.py               # regenerates both and checks them against reference/ (about 25 s)
```

`verify.py` should print `OK` for both datasets. It compares numeric cells with a 1e-9 relative tolerance, because the copula's
linear algebra differs by about 1e-23 between runs. All seeds are fixed at 42. A NumPy `RuntimeWarning: invalid value encountered in divide`
(from correlation estimates on columns with zero variance) appears during UNSW generation and does not affect the output.

## Layout

```
Group42Codebase/
├── pipeline.py                 common pipeline: one code path, stages 1-6
├── config.py                   the two dataset configurations (all parameters)
├── verify.py                   reproduction check against reference/
├── requirements.txt
├── modules/
│   ├── dvs_copula_v2.py        UNSW generator: contrastive reweighting + conditional Gaussian copula + DVS
│   ├── cleaner.py              nearest positive-definite correlation repair
│   ├── constraints.py          domain rules for both datasets
│   ├── data_loader.py          feature-type classification used by the NSL generator
│   └── generators/
│       ├── silk_kfreq.py               SILK-style MinHash/LSH seeding + k-FreqItems clustering
│       ├── itemset_copula_hybrid.py    real normals + per-cluster copula rows, proportional allocation
│       ├── itemset_clustered.py        itemset tokenisation (also the default clusterer of the hybrid)
│       └── copula_slack.py             protocol-conditioned copula with tail slack
├── artifacts/                  precomputed inputs (see below)
├── data/                       supplied training sets (train_unsw.csv, train_nslkdd.csv)
└── reference/                  the scored submission files that verify.py must reproduce
```

## Pipeline stages

A stage is switched off by a neutral parameter, never by separate code.

| # | Stage | UNSW-NB15 | NSL-KDD |
|---|---|---|---|
| 1 | generate | `dvs_copula`: one copula model on joint `proto\|service\|state`, contrastive reweighting, DVS (α=35, τ_u=0.15, τ_low=0.05), 5 bags | `silk_copula_hybrid`: 19,600 scrubbed real normals + 20,400 copula rows over SILK k-FreqItems clusters (k=20, ρ=0.35, min cluster 500), tail slack 1.05 |
| 2 | category weights | 14-cell θ, `exp(3θ)`, sampled with replacement | off (scale 0) |
| 2b | jitter repeated rows | 0.10 × std on continuous columns | off |
| 3 | gated tail stretch | 10× on the opposite normal tail, gated by agreement across 4 attack clusters (AUC ≥ 0.58, ≥ 75% of clusters) | off (factor 1) |
| 4 | domain constraints | non-negativity, integer counts, TTL bounds, protocol zeroing, `tcprtt = synack + ackdat` | rates in [0,1], integer counts, valid indicators |
| 5 | neutralisation (U[q01,q99] fill) | `sinpkt`, `sjit`, `ct_dst_src_ltm` (stability-gated leave-one-fold-out table) + `ct_dst_ltm` | `srv_count`, `srv_diff_host_rate`, `duration`, `dst_host_count`, `dst_host_same_src_port_rate`, `dst_host_rerror_rate` (single-feature ECOD lift ≤ 0.10, sparse-column guard) |
| 6 | hard bounds | `ct_dst_ltm`, `ct_src_ltm` ≤ 100; `dmean` ≤ 1500 | none |

## Precomputed artifacts

These were produced during experimentation and are shipped as fixed inputs, so a run reproduces the submitted files exactly.
The pipeline does not recompute them.

| File | Used by | Contents |
|---|---|---|
| `artifacts/weights_theta_unsw.json` | UNSW stage 2 | 14 protocol/service/state cells and their log-weights θ, fitted by an evolution strategy on the local shift harness |
| `artifacts/loo_uniform_unsw.csv` | UNSW stage 5 | leave-one-fold-out harness gain of uniform-filling each column (14 folds); the selector applies the stability gate to it |
| `artifacts/nslkdd_scrubbed_normal.csv` | NSL generator | the 20,000 NSL-KDD normals after removing the 2% most isolated rows with Isolation Forest (19,600 rows) |

## Notes

- **Different selectors per dataset.** Stage 5 is shared, but UNSW selects columns from the harness table and NSL-KDD uses a label-based single-feature rule, because the local harness did not track the NSL-KDD leaderboard.
- **Leaderboard tuning.** Several column and weight decisions were tuned against the public leaderboard. The final hidden scores come from a separate, larger test set that includes unseen attack types.
- **Deliberately unrealistic fills.** Uniform fill and DVS break some dependence between columns on purpose; the constraints and hard bounds keep values physically valid.
