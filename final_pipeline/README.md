# Common pipeline for NSL-KDD and UNSW-NB15

One code path (`pipeline.py`), one parameter block per dataset (`config.py`). A stage is switched off by its parameter
(weights `scale=0`, stretch `factor=1.0`, `constraints=None`), never by separate code.

```
python final_pipeline/pipeline.py unsw      # -> final_pipeline/out/final_unsw.csv   (70,000 rows)
python final_pipeline/pipeline.py nslkdd    # -> final_pipeline/out/final_nslkdd.csv (40,000 rows)
python final_pipeline/verify.py             # regenerates both and checks them against the scored files
```
`verify.py` compares numeric cells with a 1e-9 relative tolerance: the copula's linear algebra is not bit-reproducible (differences of ~1e-23 in a few cells of the NSL-KDD hybrid).
Needs `train_unsw.csv` / `train_nslkdd.csv` in the repo root, plus pandas, numpy, scipy, scikit-learn, pyod. Seeds are fixed (42).

## Stages

| # | Stage | UNSW-NB15 | NSL-KDD |
|---|---|---|---|
| 1 | generate | `dvs_copula`: contrastive reweighting of normals, protocol-conditioned Gaussian copula, dispersion on low-utility columns, 5 bags | `itemset_copula_hybrid` (teammate Vadiraj Rao, vendored): all 19,600 scrubbed real normals + itemset-cluster copula rows with tail slack 1.05 |
| 2 | category weights | cell weights from `artifacts/weights_theta_unsw.json`, scale 3, sampled with replacement | off |
| 3 | gated tail stretch | 10x on the opposite tail of columns whose attack direction agrees across attack clusters (never below the observed minimum) | off |
| 4 | physical constraints | `vendor/constraints.py` (`tcprtt = synack + ackdat`, TTL, protocol zeroing) | NSL-KDD rules (`vendor/constraints.py`) |
| 5 | neutralise columns (`fill`: constant or uniform) | stability-gated leave-one-fold-out table (`artifacts/loo_const_unsw.csv`) -> `sinpkt`, `sjit`, filled with uniform draws | single-feature ECOD lift <= 0.01 on non-sparse columns -> `srv_count`, `srv_diff_host_rate`, `duration`, `dst_host_count`, each filled with an independent uniform draw on the normals' [q01, q99] |
| 6 | hard bounds | `ct_dst_ltm`, `ct_src_ltm` <= 100 (they count the last 100 connections), `dmean` <= 1500 (MTU) | none |

## Evidence for each setting (public leaderboard)

UNSW-NB15, each row one change on the previous best:

| Step | Score |
|---|---|
| DVS copula, as first submitted (expt-4) | 0.53263 |
| + stability-gated constants (`sinpkt`, `sjit`) | 0.54803 |
| + gated tail stretch 1.5x / 3x / 6x | 0.54861 / 0.54991 / 0.55048 |
| + category weights x1 / x2 / x3 / x4 (6x stretch) | 0.55468 / 0.56254 / **0.56940** / 0.56106 |
| stretch 8x (uncapped) | 0.57037 |
| stretch 10x with physical bounds, `sinpkt`/`sjit` constant | 0.57067 |
| **same, `sinpkt`/`sjit` filled uniform instead of constant (this pipeline)** | **0.57698** |
| same without duplicates (16x pool, no replacement) | 0.56630 (worse; kept with replacement) |

NSL-KDD:

| File | Score |
|---|---|
| bootstrap of real normals | 0.92919 |
| bootstrap + `srv_count`, `srv_diff_host_rate` constant | 0.93006 |
| bootstrap + those 2 plus `duration`, `dst_host_count` constant | 0.92895 |
| bootstrap + the same 4 columns uniform | 0.93094 |
| **teammate's hybrid + the same 4 columns uniform (this pipeline)** | **0.93331** |
| category weights kappa -0.5 / +0.5 | 0.92901 / 0.92197 (weights off) |
| 11 constant columns incl. `service`, `logged_in` | 0.91799 (the guard now prevents this) |
| teammate's itemset-copula hybrid alone | 0.93092 |

## Honest limitations

* **Different selectors per dataset.** UNSW picks constants with the leave-one-fold-out harness; NSL-KDD cannot use that harness (it
  disagreed with the leaderboard on every neutralisation test), so it uses a label-based single-feature rule instead. The same rule hurts on UNSW
  (10 columns, harness mean -0.007). One neutralisation stage with a dataset-calibrated criterion is how we describe this; whether that counts as one
  "transferable pipeline" is a question for the course staff.
* **Pre-computed artifacts.** `weights_theta_unsw.json` (evolution search, `expt-8/src/weights.py`) and `loo_const_unsw.csv`
  (leave-one-fold-out table, `expt-6/src/pipeline.py`) are shipped, not recomputed here.
* **Mechanism of the weights is not understood.** The fitted weights do not follow the attack/normal category ratio (rank correlation 0.03).
  The evidence is the leaderboard curve (peak near scale 3.05 by a quadratic fit) and agreement of the validation harness on its direction.
* **Uniform fill is deliberately unrealistic.** An uninformative column is replaced by flat noise between its 1st and 99th percentile (e.g. `sinpkt` up to ~10,000 ms), so its ECDF is flat for ECOD and IForest wastes only random splits on it.
  The `unsw_uniform` config (harness: +0.0185 vs +0.0162 for constants, worse worst-fold) is not yet leaderboard-tested.
* **Stretch needs bounds.** Without the stage-6 caps the stretch produces impossible values (`ct_*` up to 151, `dmean` up to 6,529); the capped
  file scored higher, so the caps are kept.
* **The NSL-KDD generator is the teammate's hybrid, vendored** (`vendor/generators/`, from branch `vadi/nslkdd`, unmodified); its scrubbed-normal
  foundation is shipped as `artifacts/nslkdd_scrubbed_normal.csv` (2% isolation-forest scrub from the expt-2 EDA). `scored/` holds the exact files that reached 0.57698 and 0.93331.
* The public leaderboard covers only the validation split; the private test adds attack types never seen in training.
