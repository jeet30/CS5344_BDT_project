"""Expt 18: do the teammate's two generator ideas transfer across datasets?

UNSW:  protocol DVS 50% + SILK-cluster-conditioned DVS 50%  (replaces his state-conditioned half), and a 3-way mix.
NSL:   itemset/SILK hybrid mixed with a hybrid whose partition is (protocol_type, flag) instead of SILK clusters.
Everything after stage 1 is final_pipeline's own stages, unchanged.
"""
from __future__ import annotations
import argparse, contextlib, io, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINAL = ROOT / "final_pipeline"
sys.path[:0] = [str(FINAL), str(FINAL / "vendor"), str(ROOT / "expt-17" / "src")]
import pipeline as base
from config import CONFIG
from generators.itemset_copula_hybrid import ItemsetCopulaHybridSampler
from silk_kfreq import SILKKFreqClusterer


class PartitionClusterer:
    """Drop-in for the hybrid's cluster_sampler: partition normals by categorical columns, merging rare cells."""
    def __init__(self, cols, min_cluster_size=500):
        self.cols, self.min_cluster_size = list(cols), min_cluster_size
    def fit(self, df, label_col="is_anomaly"):
        normal = df.loc[df[label_col] == 0].drop(columns=label_col) if label_col in df else df
        self.normal_ = normal.reset_index(drop=True).copy()
        key = self.normal_[self.cols].astype(str).agg("|".join, axis=1)
        small = key.map(key.value_counts()) < self.min_cluster_size
        key = key.where(~small, "rare")
        self.cluster_rows_ = {f"part_{k}": np.flatnonzero(key == k) for k in sorted(key.unique())}
        self.cluster_counts_ = {k: len(v) for k, v in self.cluster_rows_.items()}
        return self


class Hybrid(ItemsetCopulaHybridSampler):
    def __init__(self, clusterer, **kw):
        super().__init__(**kw); self.cluster_sampler = clusterer

    def sample_synthetic(self, n):
        """Only the copula rows (no foundation copy), allocated like ItemsetCopulaHybridSampler.sample."""
        labels = sorted(self.cluster_counts_); counts = np.array([self.cluster_counts_[l] for l in labels], float)
        exact = counts / counts.sum() * n; alloc = np.floor(exact).astype(int)
        for i in sorted(range(len(labels)), key=lambda i: (-(exact[i] - alloc[i]), labels[i]))[: n - alloc.sum()]: alloc[i] += 1
        out = []
        for label, k in zip(labels, alloc):
            if not k: continue
            models = self.models_.get(label, self.models_["__global__"])
            per = [k // len(models)] * len(models)
            for j in range(k % len(models)): per[j] += 1
            out += [m.sample(int(p)) for m, p in zip(models, per) if p]
        return pd.concat(out, ignore_index=True)


def gen_nsl(cfg, mix):
    """mix = share of the copula rows drawn from the (protocol, flag) partition; the rest from SILK k=20 (expt-17 winner)."""
    normal = pd.read_csv(cfg["foundation_csv"]); extra = cfg["n"] - len(normal); kw = dict(cfg["hybrid"])
    silk = Hybrid(SILKKFreqClusterer(k=20, center_frequency=0.35, min_cluster_size=500, random_state=cfg["seed"]), random_state=cfg["seed"], **kw).fit(normal)
    parts = [normal]
    n_part = int(round(extra * mix))
    if extra - n_part: parts.append(silk.sample_synthetic(extra - n_part))
    if n_part:
        pm = Hybrid(PartitionClusterer(["protocol_type", "flag"]), random_state=cfg["seed"], **kw).fit(normal)
        print("[nsl] partition cells:", pm.cluster_counts_)
        parts.append(pm.sample_synthetic(n_part))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)


def gen_unsw(df, cfg, fractions):
    """fractions = {condition: share}; 'proto' and 'state' are columns, 'silk' is a SILK k-FreqItems cluster id of the normals."""
    from dvs_copula_v2 import ContrastiveDVSCopulaGenerator
    d = df.copy()
    if "silk" in fractions:
        feats = d.drop(columns=base.LABEL); nm = d[base.LABEL] == 0
        sc = SILKKFreqClusterer(k=20, center_frequency=0.35, min_cluster_size=500, random_state=cfg["seed"]).fit(d.loc[nm].reset_index(drop=True))
        print("[unsw] SILK clusters:", sc.cluster_counts_)
        lab = np.full(len(d), "none", dtype=object); idx = np.flatnonzero(nm.to_numpy())
        for name, rows in sc.cluster_rows_.items(): lab[idx[rows]] = name
        d["silk"] = lab
    names = list(fractions); ns = [int(round(cfg["n"] * fractions[c])) for c in names]; ns[0] += cfg["n"] - sum(ns)
    frames = []
    for c, n, delta in zip(names, ns, (0, 1000, 2000)):
        with contextlib.redirect_stdout(io.StringIO()):
            m = ContrastiveDVSCopulaGenerator(random_state=cfg["seed"] + delta, **cfg["dvs"]).fit(d, primary_cat_col=c)
        frames.append(m.sample(n).drop(columns=["silk"], errors="ignore"))
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)


def finish(name, s, df, cats, nums, cfg, out):
    s = base.apply_weights(s, cats, cfg); s = base.jitter_duplicates(s, df, cfg)
    s, stretched = base.stretch_tails(s, df, nums, cfg); s = base.apply_constraints(s, cfg)
    s, neutralized = base.neutralize(s, df, cats, nums, cfg); s = s.copy()
    for c, cap in cfg["bounds"].items(): s[c] = np.minimum(s[c], cap)
    s = s[[c for c in df if c != base.LABEL]].copy(); s.insert(0, "id", np.arange(len(s)))
    assert len(s) == cfg["n"] and s.id.is_unique and not s.isna().any().any()
    s.to_csv(out, index=False); print(f"[{name}] stretched={stretched} neutralized={neutralized} -> {out} {s.shape}")


UNSW = {"unsw_proto50_silk50": {"proto": .5, "silk": .5},
        "unsw_proto34_state33_silk33": {"proto": 1/3, "state": 1/3, "silk": 1/3}}
NSL = {"nsl_silk50_partflag50": 0.5, "nsl_partflag100": 1.0}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("variants", nargs="*", default=list(UNSW) + list(NSL)); a = ap.parse_args()
    for v in a.variants:
        out = HERE.parent / "submissions" / f"{v}.csv"
        if v in UNSW:
            cfg = CONFIG["unsw"]; df, cats, nums = base.load(cfg); finish("unsw", gen_unsw(df, cfg, UNSW[v]), df, cats, nums, cfg, out)
        else:
            cfg = CONFIG["nslkdd"]; df, cats, nums = base.load(cfg); finish("nslkdd", gen_nsl(cfg, NSL[v]), df, cats, nums, cfg, out)
