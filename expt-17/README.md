# Experiment 17 — final successful preparation pipeline

Experiment 17 contains the latest leaderboard-confirmed preparation strategy for
both mixed-attribute intrusion datasets. It uses one runner and the same final
validation/output path, with a generator suited to each dataset.

| Dataset | Rows | Confirmed score | Winning configuration |
|---|---:|---:|---|
| NSL-KDD | 40,000 | **0.93734** | SILK-seeded k-FreqItems, `k=20`, center frequency `0.35` |
| UNSW-NB15 | 70,000 | **0.61316** | 50% protocol-conditioned DVS + 50% state-conditioned DVS |

The retained scored files are:

```text
submissions/submission_nslkdd_v3_k20_cf35.csv
submissions/submission_unsw_v3_state50.csv
```

The full submission files are local generated artifacts and are ignored by Git.
Only the small score summary in `submissions/leaderboard_results.csv` is tracked,
following the project convention used for proxy and result tables.

## Inputs and dependencies

The runner expects `train_nslkdd.csv` and `train_unsw.csv` in the repository
root. Both contain `is_anomaly`, where `0` is normal and `1` is anomalous.
NSL-KDD also uses the EDA-produced foundation:

```text
final_pipeline/artifacts/nslkdd_scrubbed_normal.csv
```

The experiment imports the established copulas, selectors, learned weights,
constraints, and shared stages from `final_pipeline/`. That directory is part of
the final implementation and must be retained.

Use the repository virtual environment. It requires NumPy, pandas, SciPy,
scikit-learn, and PyOD.

## Shared end-to-end flow

```text
Training data
→ model normal behavior using labeled training evidence
→ generate the required normal sample pool
→ apply configured downstream-utility transformations
→ enforce dataset domain constraints
→ neutralize empirically weak features
→ restore required feature order
→ add unique IDs
→ validate and write submission CSV
```

After initial generation, both datasets use the same pipeline functions. A stage
is disabled through a neutral parameter when experiments showed that it hurt a
dataset—for example, weight scale `0` or stretch factor `1`.

## NSL-KDD preparation

### 1. Scrubbed normal foundation

NSL-KDD starts from 19,600 normal rows in the EDA artifact. These are normal
training rows after a 2% Isolation Forest scrub. The foundation has no anomaly
label and all rows are preserved in the final pool.

### 2. Mixed-attribute tokenization

`src/silk_kfreq.py` converts every row to an item set:

* categorical attributes remain exact tokens;
* low-cardinality integer attributes remain exact;
* continuous attributes use five quantile bins fitted only on the foundation;
* constant columns receive a constant token.

### 3. SILK-style seed discovery

Each token set receives 48 deterministic MinHash values divided into 12 LSH
bands. Rows colliding within a band form similar-row candidate buckets. A sparse
center contains tokens occurring in at least 35% of a bucket. Weighted Jaccard
k-means++ chooses 20 separated initial centers, with bucket size used as weight.

### 4. Iterative k-FreqItems clustering

Rows are assigned to the nearest center by Jaccard distance. Centers are updated
to their within-cluster frequent tokens. Assignment and updating repeat until
stable or 25 iterations. Clusters below 500 rows are merged into their nearest
viable cluster. The winning run produced 16 clusters, the smallest with 541 rows,
and assigned every row exactly once.

### 5. Cluster-conditioned copulas

The existing `ItemsetCopulaHybridSampler` fits two
`CopulaTailSlackGenerator` seeds per cluster. A local copula requires at least
500 rows and uses tail slack `1.05`. Additional rows are allocated proportionally
to original cluster size until the preserved foundation plus generated rows total
40,000. The result is deterministically shuffled.

### 6. Constraints and weak-feature neutralization

NSL-KDD rate, count, indicator, and categorical constraints are enforced. Then
single-feature ECOD utility identifies non-sparse features with anomaly-detection
lift no greater than `0.10`. The following six confirmed weak columns are replaced
by independent uniform draws over their normal-class 1st–99th percentile range:

```text
srv_count
srv_diff_host_rate
duration
dst_host_count
dst_host_same_src_port_rate
dst_host_rerror_rate
```

`dst_host_srv_diff_host_rate` is deliberately excluded because neutralizing it
reduced the leaderboard score. Category weighting and tail stretching are off for
NSL-KDD because they did not improve downstream utility.

## UNSW-NB15 preparation

### 1. Contrastive DVS fitting

DVS measures single-feature ECOD utility using normal and anomalous training rows,
uses high-utility attributes to emphasize canonical normal modes, and fits
empirical Gaussian-copula marginals and correlations. Its confirmed parameters are:

```text
alpha_dispersion = 35.0
tau_utility      = 0.15
tau_low          = 0.05
bags             = 5
seed             = 42
```

Low-utility continuous columns receive controlled dispersion. Structural,
integer, and high-utility columns remain inside observed normal bounds.

### 2. Two conditional DVS models

Two models are fitted to the same labeled training data:

* a protocol-conditioned DVS produces 35,000 rows;
* a state-conditioned DVS produces 35,000 rows.

Local correlation matrices are shrunk toward the global matrix to stabilize small
groups. The pools are combined and shuffled. The 50% state share scored better
than the tested 25% and 35% shares.

### 3. Category-cell weighting

Rows are resampled with replacement using learned categorical-cell values from
`final_pipeline/artifacts/weights_theta_unsw.json`:

```text
weight(cell) = exp(3 × theta[cell])
```

### 4. Duplicate-only jitter

Only repeated copies created by weighted resampling receive Gaussian jitter on
eligible continuous features:

```text
noise standard deviation = 0.10 × normal column standard deviation
```

The first occurrence remains unchanged and jitter cannot go below the observed
normal minimum.

### 5. Cross-attack-cluster tail stretching

Anomalous rows are divided into four standardized log-space KMeans clusters. A
feature is modified only when attack-versus-normal AUC reaches `0.58` and its
direction agrees in at least 75% of attack clusters. The normal tail opposite to
attacks is stretched 10×. The winning run changes:

```text
dload
dmean
ct_dst_ltm
ct_src_ltm
```

### 6. Constraints, neutralization, and bounds

Network relationships are restored, including TCP timing, protocol-dependent
zeroing, and TTL rules. A stability-gated leave-one-fold-out selector identified
four weak columns, which are uniformly filled over their normal 1st–99th
percentile ranges:

```text
sinpkt
sjit
ct_dst_src_ltm
ct_dst_ltm
```

Finally, semantic caps are enforced:

```text
ct_dst_ltm ≤ 100
ct_src_ltm ≤ 100
dmean      ≤ 1500
```

## Output and validation

For each dataset, the runner removes `is_anomaly`, restores training-column
order, inserts sequential unique IDs from zero, verifies the exact row count and
absence of missing values, and writes an independent CSV. The retained winning
files were also checked for finite numerics, exact duplicate rows, and domain
constraint violations; all checks passed.

## Reproduce the final strategy

The runner defaults now equal the winning parameters:

```bash
.venv/bin/python expt-17/src/run_expt17.py --dataset both --tag final
```

This creates:

```text
expt-17/submissions/submission_nslkdd_final.csv
expt-17/submissions/submission_unsw_final.csv
```

Equivalent explicit commands are:

```bash
.venv/bin/python expt-17/src/run_expt17.py \
  --dataset nslkdd \
  --k 20 \
  --center-frequency 0.35 \
  --min-cluster-size 500 \
  --tag final

.venv/bin/python expt-17/src/run_expt17.py \
  --dataset unsw \
  --state-fraction 0.50 \
  --tag final
```

Seeds are fixed. Identical inputs and dependency versions are deterministic apart
from negligible platform-level floating-point differences in copula linear algebra.
