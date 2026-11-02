# Common pipeline for NSL-KDD and UNSW-NB15

One code path (`pipeline.py`), one parameter block per dataset (`config.py`). A stage is switched off by its parameter
(weights `scale=0`, stretch `factor=1.0`, `constraints=None`, `jitter` absent), never by separate code.

```
python final_pipeline/pipeline.py unsw      # -> final_pipeline/out/final_unsw.csv   (70,000 rows)  public score 0.61452
python final_pipeline/pipeline.py nslkdd    # -> final_pipeline/out/final_nslkdd.csv (40,000 rows)  public score 0.93734
python final_pipeline/verify.py             # regenerates both and checks them against the scored files in scored/
```
`verify.py` compares numeric cells with a 1e-9 relative tolerance: the copula's linear algebra is not bit-reproducible (differences of ~1e-23 in a few cells of the NSL-KDD hybrid).
Needs `train_unsw.csv` / `train_nslkdd.csv` in the repo root, plus pandas, numpy, scipy, scikit-learn, pyod. Seeds are fixed (42).

## Stages

| # | Stage | UNSW-NB15 | NSL-KDD |
|---|---|---|---|
| 1 | generate | `dvs_copula`: contrastive reweighting of normals, ONE Gaussian copula conditioned on the joint `proto\|service\|state` (`conditions`), dispersion on low-utility columns, 5 bags | `silk_copula_hybrid` (teammate Vadiraj Rao, vendored in `vendor/generators/`): 19,600 scrubbed real normals + copula rows per SILK-seeded k-FreqItems cluster (k=20, centre frequency 0.35, min cluster 500), tail slack 1.05 |
| 2 | category weights | cell weights from `artifacts/weights_theta_unsw.json`, scale 3, sampled with replacement | off |
| 2b | jitter repeated rows | 0.10 x std on 11 continuous columns, only on repeated rows | off (the hybrid has no repeated rows) |
| 3 | gated tail stretch | 10x on the opposite tail of columns whose attack direction agrees across attack clusters (never below the observed minimum) | off |
| 4 | physical constraints | `vendor/constraints.py` (`tcprtt = synack + ackdat`, TTL, protocol zeroing) | NSL-KDD rules (`vendor/constraints.py`) |
| 5 | neutralise columns by **uniform fill** (independent uniform draw on the normals' [q01, q99]) | stability-gated leave-one-fold-out tables (`artifacts/loo_*_unsw.csv`) -> `sinpkt`, `sjit`, `ct_dst_src_ltm`, plus `ct_dst_ltm` from a second selection round | single-feature ECOD lift <= 0.10 on non-sparse columns, minus `dst_host_srv_diff_host_rate` (hurt in a leaderboard test) -> `srv_count`, `srv_diff_host_rate`, `duration`, `dst_host_count`, `dst_host_same_src_port_rate`, `dst_host_rerror_rate` |
| 6 | hard bounds | `ct_dst_ltm`, `ct_src_ltm` <= 100 (they count the last 100 connections), `dmean` <= 1500 (MTU) | none |

## Evidence (public leaderboard = validation split)

UNSW-NB15, each row one change (the first rows are from the earlier experiments):

| Step | Score |
|---|---|
| DVS copula as first submitted (expt-4) | 0.53263 |
| + stability-gated constants (`sinpkt`, `sjit`) | 0.54803 |
| + category weights x1 / x2 / **x3** / x4 (6x stretch) | 0.55468 / 0.56254 / **0.56940** / 0.56106 |
| + stretch 10x with physical bounds | 0.57067 |
| + uniform instead of constant fill | 0.57698 |
| + `ct_dst_src_ltm` (selection re-run with uniform fill) | 0.59066 |
| jitter of repeated rows instead (alone, vs 0.57698) | 0.58806 |
| selection round 1 + jitter | 0.60782 |
| + `ct_dst_ltm` (selection round 2) = previous default | 0.60905 |
| + 50% protocol-conditioned + 50% state-conditioned DVS (teammate, expt-17) | 0.61316 |
| one DVS model on the joint `proto\|state` (expt-19) | 0.61392 |
| **one DVS model on the joint `proto\|service\|state` = default** | **0.61452** |
| mixes: proto 50% + SILK-conditioned 50% / proto, state, SILK one third each | 0.61138 / 0.60781 (three partitions are worse than one or two) |
| no duplicates (16x pool, no replacement) | 0.56630 (worse; sampling with replacement kept) |

NSL-KDD:

| File | Score |
|---|---|
| bootstrap of real normals | 0.92919 |
| bootstrap + `srv_count`, `srv_diff_host_rate` constant / + `duration`, `dst_host_count` constant | 0.93006 / 0.92895 |
| bootstrap + those 4 columns uniform | 0.93094 |
| teammate's hybrid alone | 0.93092 |
| hybrid + 4 uniform columns | 0.93331 |
| + `dst_host_srv_diff_host_rate` (5 columns) | 0.93192 (that column hurts) |
| + `dst_host_same_src_port_rate`, `dst_host_rerror_rate` (7 columns) | 0.93512 |
| 7 columns minus the harmful one (6 columns) = previous default | 0.93674 |
| **same, with SILK k-FreqItems clusters instead of itemset clusters = default** | **0.93734** (teammate; +0.0006, inside the noise) |
| SILK 50% + (protocol_type, flag)-partition 50% / partition only | 0.93642 / 0.93668 (a flag partition adds nothing) |
| 6 columns with log1p-uniform fill | 0.93101 (worse) |
| 6 columns + `dst_bytes` | 0.93643 (no gain) |
| category weights kappa -0.5 / +0.5 | 0.92901 / 0.92197 (off) |
| 11 constant columns incl. `service`, `logged_in` | 0.91799 (the sparse-column guard prevents this) |

## Other configs

| Config | Change | Status |
|---|---|---|
| `unsw_sel5_dupjit` | default + `trans_depth` filled (third selection round) | harness held-out +0.0064, 12/14 folds; leaderboard result being clarified |
| `nslkdd_unif6_db` | default + `dst_bytes` | scored 0.93643 (no gain) |
| `unsw_prev_0.60905`, `nslkdd_prev_0.93674` | the previous defaults (still reproduced by `verify.py`) | scored |
| `unsw_blend_proto_state`, `unsw_product_proto_state`, `unsw_product3` | generator conditioning variants (`unsw` = `unsw_product3`) | scored 0.61316 / 0.61392 / 0.61452 |
| `nslkdd_silk` | SILK clusters (`nslkdd` = `nslkdd_silk`) | scored 0.93734 |
| `unsw_prev_0.57698`, `nslkdd_prev_0.93331` | older defaults | scored |

| `nslkdd_loguniform`, `nslkdd_partial50` | default with log1p-uniform / 50% partial fill | not submitted; the UNSW harness found both clearly worse (log1p-uniform -0.0149, partial 50% -0.0137) |
| `nslkdd_slack12` | hybrid tail slack 1.2 instead of 1.05 | untested (generator-side) |

The fill options `shape` (uniform, uniform_wide, uniform_narrow, loguniform, quantile) and `frac` live in `fill_columns`; on UNSW the default (U[q01,q99], all rows) beat every alternative (expt-17).

Run any config with `python final_pipeline/pipeline.py <config>`. Promote one only after a leaderboard confirmation, and re-point `verify.py` and `scored/`.

## Honest limitations

* **Different selectors per dataset.** UNSW picks columns with a leave-one-fold-out harness (reliable on UNSW); NSL-KDD cannot use that harness
  (it disagreed with the leaderboard on every neutralisation test), so it uses a label-based single-feature rule and leaderboard feedback.
  The single-feature rule hurts on UNSW. We describe this as one neutralisation stage with a dataset-calibrated criterion; whether that satisfies
  "one transferable pipeline" is a question for the course staff.
* **Public-leaderboard tuning.** Many column decisions (especially the NSL-KDD ones: adding / dropping columns by public score) were tuned against the
  validation split. The private test adds attack types never seen in training, so the ranking can change.
* **Pre-computed artifacts.** `weights_theta_unsw.json` (evolution search, `expt-8/src/weights.py`) and the leave-one-fold-out tables
  (`expt-6/src/pipeline.py`, `expt-13/src/reselect_uniform.py`) are shipped, not recomputed here. The extra UNSW column from the second selection round
  (`expt-16/src/round2.py`) is listed in the config.
* **Mechanisms are not fully understood.** The fitted category weights do not follow the attack/normal category ratio (rank correlation 0.03). Uniform fill beats
  constant fill on both datasets, and the column effects are heterogeneous (one NSL-KDD column hurt); a random-forest predictability probe (expt-14) did not explain which columns help.
* **Uniform fill is deliberately unrealistic.** An uninformative column becomes flat noise between its 1st and 99th percentile, so its ECDF is flat for ECOD and IForest
  wastes only random splits on it. The jitter stage and the tail stretch likewise break exact physical relations between columns (e.g. `dload` vs `dbytes`/`dur`).
* **The stretch needs bounds.** Without the stage-6 caps it produces impossible values (`ct_*` up to 151, `dmean` up to 6,529); the capped file scored higher.
* **Generator gains are small and the harness cannot resolve them.** The UNSW harness scores the proto-only and the blend generators identically (0.5704), so generator choices (and SILK vs itemset clusters, +0.0006) rest on single leaderboard scores. The more partitions the better only up to two; SILK was never compared with a random partition of the same size.
* The NSL-KDD generator is the teammate's hybrid, vendored unmodified from branch `vadi/nslkdd` (clusterer `silk_kfreq.py` from expt-17, commit b82c394); its scrubbed-normal foundation is
  `artifacts/nslkdd_scrubbed_normal.csv` (2% isolation-forest scrub from the expt-2 EDA). `scored/` holds the files behind the defaults' scores. The NSL-KDD 0.93734 reference there is regenerated: it matches the output of the teammate's own runner (`expt-17/src/run_expt17.py`) to 1e-9, but his submitted CSV is not in the repo.
