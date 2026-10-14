"""Expt 9 pipeline = pipeline8 + (a) weighted sampling WITHOUT duplicates (pool of `pool`x rows, draw n by cell weights, no replacement)
+ (b) configurable cross-cluster gating thresholds. Same seed/stages as pipeline8 so every file is a one-change A/B."""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-8/src")
import pipeline8 as p8
from e8 import feature_direction, h
p7, p6 = p8.p7, p8.p6

def widen_gated2(s, df, nums, widen, min_auc, min_agree, seed=42):
    dirs = feature_direction(df, nums, seed=seed, min_auc=min_auc, min_agree=min_agree); real = df[df.is_anomaly == 0]; s = s.copy(); touched = []
    for c, d in dirs.items():
        v = s[c].to_numpy(float).copy()
        if d > 0: u = np.quantile(v, 0.10); m = v < u; v[m] = u + (v[m] - u) * widen
        else: u = np.quantile(v, 0.90); m = v > u; v[m] = u + (v[m] - u) * widen
        v = np.maximum(v, real[c].min())
        if np.allclose(real[c], np.rint(real[c])): v = np.rint(v)
        if (v != s[c].to_numpy(float)).any(): touched.append(c)
        s[c] = v
    return s, touched

def run(name, widen, scale=1.0, nodup=True, pool=2, min_auc=0.58, min_agree=0.75, tag=""):
    cfg = p7.GEN[name]; df, cats, nums = h.load(name); normals = df[df.is_anomaly == 0]
    T = p6.loo_table(name, cfg, p7.R7 + f"loo_const_{name}_pipeline.csv"); M = T[T.feat != "_base"].pivot(index="feat", columns="fold", values="d")
    sel = p7.select_v2(M, nums + cats, normals, **p7.SHARED); n = cfg["n"]
    s = p6.generate(df, n * (pool if nodup else 1), 42, cfg, cats, nums)
    if scale is not None:
        W = json.load(open(h.ROOT + f"expt-8/out/weights_theta_{name}.json")); idx = {c: i for i, c in enumerate(W["cells"])}; th = np.array(W["theta"] + [0.0])
        k = s[cats].astype(str).agg("|".join, axis=1).map(idx).fillna(len(W["theta"])).astype(int).to_numpy(); w = np.exp(scale * th[k]); w /= w.sum()
        s = s.iloc[np.random.default_rng(42).choice(len(s), size=n, replace=not nodup, p=w)].reset_index(drop=True)
    s, touched = widen_gated2(s, df, nums, widen, min_auc, min_agree)
    if cfg["constraints"] == "unsw":
        from constraints import enforce_unsw_constraints; s = enforce_unsw_constraints(s)
    for c in sel: s[c] = p6.neutral_value(s, c, cats)
    s = s[[c for c in df.columns if c != h.LABEL]]; s.insert(0, "id", np.arange(len(s))); assert len(s) == n and s.isna().sum().sum() == 0
    out = h.ROOT + f"expt-9/submissions/pipeline9_gate{int(widen*10)}_w{int(round((scale or 0)*10))}{'_nodup' if nodup else ''}{tag}_{name}.csv"
    s.to_csv(out, index=False); print("wrote", out, s.shape, "| dup share %.3f" % s.drop(columns="id").duplicated().mean(), "| stretched:", touched); return s

if __name__ == "__main__":
    run("unsw", 6.0, 1.0, nodup=True)
