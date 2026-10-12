"""Expt 8 pipeline: expt-7 pipeline + cross-cluster-gated opposite-tail widening (item 5). Everything else identical to pipeline7 (same seed) => clean A/B."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-7/src")
import pipeline7 as p7
from e8 import feature_direction, h, R8
p6 = p7.p6

def widen_gated(s, df, nums, widen, seed=42):
    dirs = feature_direction(df, nums, seed=seed); real = df[df.is_anomaly == 0]; s = s.copy(); touched = []
    for c, d in dirs.items():
        v = s[c].to_numpy(float).copy()
        if d > 0: u = np.quantile(v, 0.10); m = v < u; v[m] = u + (v[m] - u) * widen          # attacks consistently higher -> stretch lower tail
        else: u = np.quantile(v, 0.90); m = v > u; v[m] = u + (v[m] - u) * widen              # attacks consistently lower  -> stretch upper tail
        v = np.maximum(v, real[c].min())                                                      # never below the observed lower bound
        if np.allclose(real[c], np.rint(real[c])): v = np.rint(v)
        if (v != s[c].to_numpy(float)).any(): touched.append(c)
        s[c] = v
    return s, dirs, touched

def run(name, widen, theta=None, scale=1.0):
    cfg = p7.GEN[name]; df, cats, nums = h.load(name); normals = df[df.is_anomaly == 0]
    T = p6.loo_table(name, cfg, p7.R7 + f"loo_const_{name}_pipeline.csv"); M = T[T.feat != "_base"].pivot(index="feat", columns="fold", values="d")
    sel = p7.select_v2(M, nums + cats, normals, **p7.SHARED)
    s = p6.generate(df, cfg["n"], 42, cfg, cats, nums)
    if theta:   # item 6: category-cell row weights (fitted by evolution strategy on all folds); resample the generated pool
        import json; T = json.load(open(h.ROOT + f"expt-8/out/weights_theta_{name}.json")); idx = {c: i for i, c in enumerate(T["cells"])}
        k = s[cats].astype(str).agg("|".join, axis=1).map(idx); w = np.exp(scale * np.where(k.isna(), 0.0, np.array(T["theta"] + [0.0])[k.fillna(len(T["theta"])).astype(int)])); w /= w.sum()
        s = s.iloc[np.random.default_rng(42).choice(len(s), size=len(s), p=w)].reset_index(drop=True)
    s, dirs, touched = widen_gated(s, df, nums, widen)
    if cfg["constraints"] == "unsw":
        from constraints import enforce_unsw_constraints; s = enforce_unsw_constraints(s)
    for c in sel: s[c] = p6.neutral_value(s, c, cats)
    s = s[[c for c in df.columns if c != h.LABEL]]; s.insert(0, "id", np.arange(len(s))); assert len(s) == cfg["n"] and s.isna().sum().sum() == 0
    out = h.ROOT + f"expt-8/submissions/pipeline8_gate{int(widen*10)}{('_w%d' % round(scale * 10)) if theta else ''}_{name}.csv"; s.to_csv(out, index=False); print("wrote", out, s.shape, "| widened columns:", touched); return s

if __name__ == "__main__":
    for a in sys.argv[2:]:
        wv, _, t = a.partition(":"); run(sys.argv[1], float(wv), theta=bool(t), scale=float(t) if t and t != "w" else 1.0)
