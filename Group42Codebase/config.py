"""Per-dataset parameters for the common pipeline (pipeline.py).

Every stage runs for both datasets; a stage is switched off through its parameter (weights scale 0, stretch factor 1.0,
no jitter entry, empty bounds), never through separate code. These are the two configurations behind the selected
submissions (public leaderboard 0.61452 / 0.93734; final hidden 0.68508 / 0.95247).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ART = HERE / "artifacts"

CONFIG = {
    # ---------------------------------------------------------------------------------------------------- UNSW-NB15
    "unsw": dict(
        train_csv=DATA / "train_unsw.csv", n=70000, seed=42,
        # 1 generator: contrastive-reweighted Gaussian copula conditioned on the joint proto|service|state, plus DVS dispersion
        generator="dvs_copula",
        dvs=dict(alpha_dispersion=35.0, tau_utility=0.15, tau_low=0.05, n_bags=5),
        conditions=[dict(cols=["proto", "service", "state"], frac=1.0)],
        # 2 category-cell weights: 14 fitted cells (artifact), scale 3, sampled with replacement
        weights=dict(theta=ART / "weights_theta_unsw.json", scale=3.0),
        # 2b jitter of repeated rows: 0.10 x std on continuous columns
        jitter=dict(sigma=0.10, duplicates_only=True),
        # 3 cross-cluster-gated opposite-tail stretch, 10x
        stretch=dict(factor=10.0, K=4, min_auc=0.58, min_agree=0.75),
        # 4 domain rules (modules/constraints.py)
        constraints="unsw",
        # 5 uniform fill; stability-gated leave-one-fold-out table selects sinpkt, sjit, ct_dst_src_ltm; a second round adds ct_dst_ltm
        neutralize=dict(selector="loo_table", table=ART / "loo_uniform_unsw.csv", thr=0.003, vote=0.7, freq=0.8, cvar_tol=0.004,
                        guard=0.9, fill="uniform", extra_columns=["ct_dst_ltm"]),
        # 6 hard bounds: ct_* count the last 100 connections; dmean <= MTU
        bounds={"ct_dst_ltm": 100, "ct_src_ltm": 100, "dmean": 1500},
    ),
    # ---------------------------------------------------------------------------------------------------- NSL-KDD
    "nslkdd": dict(
        train_csv=DATA / "train_nslkdd.csv", n=40000, seed=42,
        # 1 generator: 19,600 scrubbed real normals (artifact) + 20,400 copula rows allocated over SILK-seeded k-FreqItems clusters
        generator="silk_copula_hybrid",
        foundation_csv=ART / "nslkdd_scrubbed_normal.csv",
        hybrid=dict(min_support=0.01, max_support=0.35, min_copula_rows=500, tail_slack_gamma=1.05, n_copula_seeds=2),
        silk=dict(k=20, center_frequency=0.35, min_cluster_size=500),
        dvs=None,
        # 2, 3 off
        weights=dict(theta=None, scale=0.0),
        stretch=dict(factor=1.0, K=4, min_auc=0.58, min_agree=0.75),
        # 4 domain rules
        constraints="nslkdd",
        # 5 uniform fill of non-sparse columns with single-feature ECOD AP lift <= 0.10, minus dst_host_srv_diff_host_rate (hurt on the leaderboard)
        neutralize=dict(selector="single_feature_ecod", lift_max=0.10, guard=0.9, fill="uniform",
                        skip_columns=["dst_host_srv_diff_host_rate"]),
        # 6 no hard bounds
        bounds={},
    ),
}
