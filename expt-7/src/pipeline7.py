"""Expt 7 unified pipeline: IDENTICAL parameters for both datasets except the generator switch (copula on/off).
Stage 1 generator (expt-6 pipeline.generate) -> Stage 2 stability-gated neutralisation (select_v2, shared thr/vote/freq/guard) -> Stage 3 constraints."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-6/src")
import pipeline as p6
from e7 import select_v2, smooth_cats, h, R7
SHARED = dict(thr=0.003, vote=0.7, freq=0.8, cvar_tol=0.004, guard=0.9)          # one setting for both datasets
GEN = {"unsw": p6.CFG["unsw"], "nslkdd": {**p6.CFG["nslkdd"], "thr": SHARED["thr"]}}   # only difference: copula on (UNSW) / off (NSL-KDD)

def run(name, lam=0.0):
    cfg = GEN[name]; df, cats, nums = h.load(name); normals = df[df.is_anomaly == 0]
    T = p6.loo_table(name, cfg, R7 + f"loo_const_{name}_pipeline.csv"); M = T[T.feat != "_base"].pivot(index="feat", columns="fold", values="d")
    sel = select_v2(M, nums + cats, normals, **SHARED); print(name, "neutralised:", sel)
    s = p6.generate(df, cfg["n"], 42, cfg, cats, nums)
    if lam > 0:  # stage 1b: shrink categorical joint toward conditional independence given the protocol column (numerics untouched => clean A/B vs lam=0)
        cond = "proto" if "proto" in cats else ("protocol_type" if "protocol_type" in cats else cats[0]); s = smooth_cats(s, cond, cats, lam, 42)
    if cfg["constraints"] == "unsw":
        from constraints import enforce_unsw_constraints; s = enforce_unsw_constraints(s)
    for c in sel: s[c] = p6.neutral_value(s, c, cats)
    s = s[[c for c in df.columns if c != h.LABEL]]; s.insert(0, "id", np.arange(len(s))); assert len(s) == cfg["n"] and s.isna().sum().sum() == 0
    out = h.ROOT + f"expt-7/submissions/pipeline7{'_eb%d' % round(lam*100) if lam > 0 else ''}_{name}.csv"; s.to_csv(out, index=False); print("wrote", out, s.shape); return s

if __name__ == "__main__":
    for a in sys.argv[1:]:
        n, _, l = a.partition(":"); run(n, float(l) if l else 0.0)
