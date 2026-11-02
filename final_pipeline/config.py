"""Per-dataset parameters for the single common pipeline (pipeline.py).

Every stage runs for both datasets; a stage is "off" through its parameter (scale 0, widen 1.0, selector None), never through separate code.
Leaderboard evidence for each setting is in README.md.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = Path(__file__).resolve().parent / "artifacts"

CONFIG = {
    "unsw": dict(
        train_csv=ROOT / "train_unsw.csv", n=70000, seed=42,
        # stage 1: generator ------------------------------------------------------------------
        generator="dvs_copula",                                   # contrastive-reweighted, protocol-conditioned Gaussian copula + dispersion
        dvs=dict(alpha_dispersion=35.0, tau_utility=0.15, tau_low=0.05, n_bags=5),
        # stage 2: category-cell weights (resample the generated pool with replacement) -----------
        weights=dict(theta=ART / "weights_theta_unsw.json", scale=3.0),
        # stage 3: cross-cluster-gated opposite-tail stretch -------------------------------------
        stretch=dict(factor=10.0, K=4, min_auc=0.58, min_agree=0.75),
        # stage 4: physical-law constraints (dataset rules, vendor/constraints.py) ---------------
        constraints="unsw",
        # stage 5: columns held constant ------------------------------------------------------
        neutralize=dict(selector="loo_table", table=ART / "loo_const_unsw.csv", thr=0.003, vote=0.7, freq=0.8, cvar_tol=0.004, guard=0.9, fill="uniform"),   # uniform 0.57698 vs constant 0.57067
        # stage 6: hard physical bounds applied last ----------------------------------------------
        bounds={"ct_dst_ltm": 100, "ct_src_ltm": 100, "dmean": 1500},   # ct_* count the last 100 connections; dmean <= MTU
    ),
    "nslkdd": dict(
        train_csv=ROOT / "train_nslkdd.csv", n=40000, seed=42,
        generator="itemset_copula_hybrid",                         # 0.93092 alone (teammate); "real_resample" (plain bootstrap) scored 0.92919
        hybrid=dict(min_support=0.01, max_support=0.35, min_copula_rows=500, tail_slack_gamma=1.05, n_copula_seeds=2),
        foundation_csv=ART / "nslkdd_scrubbed_normal.csv",         # normals after the 2% isolation-forest scrub (expt-2 EDA output)
        dvs=None,
        weights=dict(theta=None, scale=0.0),                       # weights OFF (kappa +-0.5 tests: 0.92197 / 0.92901 vs 0.92919)
        stretch=dict(factor=1.0, K=4, min_auc=0.58, min_agree=0.75),   # stretch OFF
        constraints="nslkdd",
        neutralize=dict(selector="single_feature_ecod", lift_max=0.01, guard=0.9, fill="uniform"),   # bootstrap base: 0.93094 (constant fill of 2 cols: 0.93006, of 4 cols: 0.92895); hybrid base: 0.93331
        bounds={},
    ),
}

# EXPERIMENT (not default): relax the single-feature lift cutoff to 0.04 -> also fills dst_host_srv_diff_host_rate (lift 0.036). Leaderboard-untested.
CONFIG["nslkdd_lift04"] = {**CONFIG["nslkdd"], "neutralize": {**CONFIG["nslkdd"]["neutralize"], "lift_max": 0.04}}

# EXPERIMENT (not default): UNSW column selection re-run with UNIFORM fill (expt-13) -> sinpkt, sjit, ct_dst_src_ltm. Harness held-out gain +0.0279 vs +0.0183; leaderboard-untested.
CONFIG["unsw_sel3"] = {**CONFIG["unsw"], "neutralize": {**CONFIG["unsw"]["neutralize"], "table": ART / "loo_uniform_unsw.csv"}}

# EXPERIMENT (not default): jitter the repeated rows (sigma = 0.10 x column std, 11 continuous columns). Harness: +0.0057 mean, 9/14 folds improved; leaderboard-untested.
CONFIG["unsw_dupjit"] = {**CONFIG["unsw"], "jitter": dict(sigma=0.10, duplicates_only=True)}

# EXPERIMENT (not default): both leaderboard-confirmed improvements together (unsw_sel3 0.59066, unsw_dupjit 0.58806, each vs scored 0.57698). Combined: untested.
CONFIG["unsw_sel3_dupjit"] = {**CONFIG["unsw_sel3"], "jitter": dict(sigma=0.10, duplicates_only=True)}

# EXPERIMENT (not default): lift cutoff 0.10 -> 7 uniform-filled columns (adds dst_host_same_src_port_rate, dst_host_rerror_rate to the 0.04 set). Leaderboard-untested.
CONFIG["nslkdd_lift10"] = {**CONFIG["nslkdd"], "neutralize": {**CONFIG["nslkdd"]["neutralize"], "lift_max": 0.10}}

# EXPERIMENT (not default): sel3 + dup-jitter + a SECOND selection round (expt-16, uniform fill on top of the sel3 set) which adds ct_dst_ltm. Harness held-out +0.0037; untested.
CONFIG["unsw_sel4_dupjit"] = {**CONFIG["unsw_sel3_dupjit"], "neutralize": {**CONFIG["unsw_sel3_dupjit"]["neutralize"], "extra_columns": ["ct_dst_ltm"]}}

# EXPERIMENT (not default): unif7 (LB 0.93512) minus the column that hurt in unif5 (dst_host_srv_diff_host_rate; unif5 0.93192 vs unif4 0.93331). Draws of all other columns identical to unif7. Projection ~0.9365, untested.
CONFIG["nslkdd_unif6"] = {**CONFIG["nslkdd_lift10"], "neutralize": {**CONFIG["nslkdd_lift10"]["neutralize"], "skip_columns": ["dst_host_srv_diff_host_rate"]}}

# EXPERIMENT (not default): unif6 + dst_bytes (lift 0.128, the next-lowest non-sparse column). Earlier columns' draws unchanged. Untested.
CONFIG["nslkdd_unif6_db"] = {**CONFIG["nslkdd_unif6"], "neutralize": {**CONFIG["nslkdd_unif6"]["neutralize"], "lift_max": 0.13}}

# EXPERIMENT (not default): sel4_dupjit (LB 0.60905) + a THIRD selection round (expt-16/round3.py) which adds trans_depth. Harness held-out +0.0064 (12/14 folds). Untested.
CONFIG["unsw_sel5_dupjit"] = {**CONFIG["unsw_sel4_dupjit"], "neutralize": {**CONFIG["unsw_sel4_dupjit"]["neutralize"], "extra_columns": ["ct_dst_ltm", "trans_depth"]}}

# ---- DEFAULTS PROMOTED (2026-10-20), each confirmed on the public leaderboard --------------------------------------------------------------
CONFIG["unsw_prev_0.57698"] = CONFIG["unsw"]          # previous default (uniform fill of sinpkt, sjit only)
CONFIG["nslkdd_prev_0.93331"] = CONFIG["nslkdd"]      # previous default (4 uniform-filled columns)
CONFIG["unsw"] = CONFIG["unsw_sel4_dupjit"]           # 0.60905: selection round 1+2 with uniform fill (sinpkt, sjit, ct_dst_src_ltm, ct_dst_ltm) + jitter of repeated rows
CONFIG["nslkdd"] = CONFIG["nslkdd_unif6"]             # 0.93674: hybrid + uniform fill of 6 columns (lift <= 0.10, minus dst_host_srv_diff_host_rate which hurt)

# ---- EXPERIMENTS on the promoted defaults (expt-17); none leaderboard-tested yet --------------------------------------------------------------
_nsl = CONFIG["nslkdd"]
CONFIG["nslkdd_loguniform"] = {**_nsl, "neutralize": {**_nsl["neutralize"], "shape": "loguniform"}}          # same 6 columns, log1p-uniform fill
CONFIG["nslkdd_partial50"] = {**_nsl, "neutralize": {**_nsl["neutralize"], "frac": 0.5}}                    # same 6 columns, uniform fill on a random 50% of rows
CONFIG["nslkdd_slack12"] = {**_nsl, "hybrid": {**_nsl["hybrid"], "tail_slack_gamma": 1.2}}                  # hybrid tail slack 1.05 -> 1.2

# ---- NEW GENERATORS (teammate Vadiraj Rao's expt-17 + our expt-19), every one leaderboard-confirmed; stages 2-6 unchanged ---------------------
_u, _n = CONFIG["unsw"], CONFIG["nslkdd"]                                    # still the 0.60905 / 0.93674 defaults at this point
CONFIG["unsw_prev_0.60905"], CONFIG["nslkdd_prev_0.93674"] = _u, _n
_dvs = lambda *conds: {**_u, "conditions": [dict(cols=c, frac=f) for c, f in conds]}
CONFIG["unsw_blend_proto_state"] = _dvs((["proto"], 0.5), (["state"], 0.5))                       # 0.61316 (teammate): 50% protocol-conditioned + 50% state-conditioned DVS
CONFIG["unsw_product_proto_state"] = _dvs((["proto", "state"], 1.0))                              # 0.61392: ONE model on the joint proto|state
CONFIG["unsw_product3"] = _dvs((["proto", "service", "state"], 1.0))                              # 0.61452: ONE model on the joint proto|service|state
CONFIG["nslkdd_silk"] = {**_n, "generator": "silk_copula_hybrid", "silk": dict(k=20, center_frequency=0.35, min_cluster_size=500)}   # 0.93734 (teammate)
CONFIG["unsw"] = CONFIG["unsw_product3"]                                                          # default: 0.61452
CONFIG["nslkdd"] = CONFIG["nslkdd_silk"]                                                          # default: 0.93734

# EXPERIMENT (not default): unsw_product3 (0.61452) with the 40-cell category weights fitted on the DVS-blend base (expt-19/src/big.py; harness held-out +0.0069 / +0.0127 on the two family splits).
# Scale 2 chosen in the harness (x1 +0.0028, x2 +0.0071, x3 -0.0041 vs the 14-cell x3 recipe; in-sample). LEADERBOARD: 0.59394 (-0.0206 vs unsw_product3 0.61452) -> does not transfer.
CONFIG["unsw_product3_w40"] = {**CONFIG["unsw_product3"], "weights": dict(theta=ART / "weights_theta_unsw_40cell.json", scale=2.0)}

# EXPERIMENT (not default): unsw_product3 with an extra x0.5 weight on the four attack-overlapping tcp cells (tcp|-|FIN, tcp|http|FIN, tcp|smtp|FIN, tcp|ftp|FIN) on top of the 14-cell theta x3.
# Motivated by the leaderboard drop of the 40-cell weights (0.59394), which UPweighted those cells. LEADERBOARD: 0.59746 (-0.0171 vs unsw_product3) -> also worse: the old 14-cell theta x3 is a sharp optimum in both directions.
CONFIG["unsw_product3_shared05"] = {**CONFIG["unsw_product3"], "weights": dict(theta=ART / "weights_theta_unsw_shared05.json", scale=3.0)}
