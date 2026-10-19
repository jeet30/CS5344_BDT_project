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
