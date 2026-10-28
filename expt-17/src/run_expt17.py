"""Experiment 17: SILK+k-FreqItems for NSL and multi-condition DVS for UNSW."""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINAL = ROOT / "final_pipeline"
sys.path[:0] = [str(FINAL), str(FINAL / "vendor"), str(HERE)]

import pipeline as base
from config import CONFIG
from generators.itemset_copula_hybrid import ItemsetCopulaHybridSampler
from silk_kfreq import SILKKFreqClusterer


class SILKCopulaHybrid(ItemsetCopulaHybridSampler):
    def __init__(self, k=20, center_frequency=0.35, min_cluster_size=500, **kwargs):
        super().__init__(**kwargs)
        self.cluster_sampler = SILKKFreqClusterer(
            k=k, center_frequency=center_frequency,
            min_cluster_size=max(min_cluster_size, self.min_copula_rows),
            random_state=self.random_state,
        )


def generate_nsl(df, cfg, k, center_frequency, min_cluster_size):
    normal = pd.read_csv(cfg["foundation_csv"])
    model = SILKCopulaHybrid(k=k, center_frequency=center_frequency,
                             min_cluster_size=min_cluster_size, **cfg["hybrid"]).fit(normal)
    print("[nslkdd] SILK diagnostics:", model.cluster_sampler.diagnostics_)
    return model.sample(cfg["n"])


def generate_unsw(df, cfg, state_fraction):
    """Blend proven protocol-conditioned DVS with state-conditioned DVS."""
    from dvs_copula_v2 import ContrastiveDVSCopulaGenerator
    n_state = int(round(cfg["n"] * state_fraction)); n_proto = cfg["n"] - n_state
    frames = []
    for condition, n, seed_delta in (("proto", n_proto, 0), ("state", n_state, 1000)):
        if not n: continue
        params = dict(cfg["dvs"]); seed = cfg["seed"] + seed_delta
        with contextlib.redirect_stdout(io.StringIO()):
            model = ContrastiveDVSCopulaGenerator(random_state=seed, **params).fit(
                df, primary_cat_col=condition)
        frames.append(model.sample(n))
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)


def run(name, out, k=20, state_fraction=0.50, center_frequency=0.35,
        min_cluster_size=500):
    cfg = CONFIG[name]; df, cats, nums = base.load(cfg)
    s = (generate_nsl(df, cfg, k, center_frequency, min_cluster_size)
         if name == "nslkdd" else generate_unsw(df, cfg, state_fraction))
    s = base.apply_weights(s, cats, cfg)
    s = base.jitter_duplicates(s, df, cfg)
    s, stretched = base.stretch_tails(s, df, nums, cfg)
    s = base.apply_constraints(s, cfg)
    s, neutralized = base.neutralize(s, df, cats, nums, cfg)
    for c, cap in cfg["bounds"].items(): s[c] = np.minimum(s[c], cap)
    s = s[[c for c in df if c != base.LABEL]].copy(); s.insert(0, "id", np.arange(len(s)))
    assert len(s) == cfg["n"] and s.id.is_unique and not s.isna().any().any()
    out.parent.mkdir(parents=True, exist_ok=True); s.to_csv(out, index=False)
    print(f"[{name}] stretched={stretched} neutralized={neutralized} -> {out} {s.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("nslkdd", "unsw", "both"), default="both")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--center-frequency", type=float, default=0.35)
    p.add_argument("--min-cluster-size", type=int, default=500)
    p.add_argument("--state-fraction", type=float, default=0.50)
    p.add_argument("--tag", default="final")
    a = p.parse_args()
    if not 0 <= a.state_fraction <= 1: p.error("--state-fraction must be in [0, 1]")
    names = ("nslkdd", "unsw") if a.dataset == "both" else (a.dataset,)
    for name in names:
        run(name, ROOT / "expt-17" / "submissions" / f"submission_{name}_{a.tag}.csv",
            a.k, a.state_fraction, a.center_frequency, a.min_cluster_size)
