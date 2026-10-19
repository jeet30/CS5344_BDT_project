"""Expt 15: candidate extra post-processing for UNSW, tested on the reliable harness (weighted + 6x-stretched base, uniform fill of sinpkt/sjit, ct/dmean caps).
 dupjit   : tiny Gaussian jitter on the REPEATED rows only (duplicates come from the with-replacement weighting)
 jitall   : tiny jitter on every row
 blendNN  : NN% of rows replaced by real contrastive-purified normals (no copula)"""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
for p in ("expt-7/src", "expt-8/src", "expt-12/src", "expt-13/src"): sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/" + p)
from reselect_uniform import *       # Weighter, h, R8, uni, const, OLD, SCALE
R15 = h.ROOT + "expt-15/out/"; CAPS = {"ct_dst_ltm": 100, "ct_src_ltm": 100, "dmean": 1500}
W = Weighter("unsw", base="gatev_dvs_w60"); th = np.array(json.load(open(R8 + "weights_theta_unsw.json"))["theta"]) * SCALE
nm = W.df[W.df.is_anomaly == 0]; cont = [c for c in W.nums if nm[c].nunique() > 100 and not np.allclose(nm[c], np.rint(nm[c]))]; std = nm[cont].std()
print("continuous columns jittered:", len(cont), flush=True)
def finish(s):
    s = uni(s, OLD, nm, W.cats)
    for c, m in CAPS.items(): s[c] = np.minimum(s[c], m)
    return s
def jitter(s, sigma, mask, seed=7):
    s = s.copy(); rng = np.random.default_rng(seed); m = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
    for c in cont: s.loc[m, c] = np.maximum(s.loc[m, c].to_numpy() + rng.normal(0, sigma * std[c], int(m.sum())), nm[c].min())
    return s
rows = []
for fn in W.F:
    tr = W.df.loc[W.F[fn][0]]; w = np.exp(np.r_[th, 0.0][W.cellid[fn]]); w /= w.sum()
    s0 = W.S0[fn].iloc[np.random.default_rng(0).choice(len(w), size=len(w), p=w)].reset_index(drop=True); dup = s0.duplicated(keep="first")
    V = {"base": finish(s0), "dupjit_002": finish(jitter(s0, 0.02, dup)), "dupjit_010": finish(jitter(s0, 0.10, dup)), "jitall_005": finish(jitter(s0, 0.05, pd.Series(True, index=s0.index)))}
    for pct in (10, 25):
        pool = h.g_clr(tr, len(s0), 0, W.cats, W.nums)[s0.columns]; idx = np.random.default_rng(1).choice(len(s0), size=int(len(s0) * pct / 100), replace=False)
        b = s0.copy(); b.iloc[idx] = pool.iloc[: len(idx)].to_numpy(); V[f"blend{pct}"] = finish(b)
    sc = {k: h.score(v, W.te[fn], W.cats, W.nums, seeds=2)[2] for k, v in V.items()}; rows += [(fn, W.fam[fn], k, v) for k, v in sc.items()]
    print(fn, "dup share %.2f |" % dup.mean(), {k: round(v - sc["base"], 4) for k, v in sc.items() if k != "base"}, flush=True)
R = pd.DataFrame(rows, columns=["fold", "family", "variant", "S"]); R.to_csv(R15 + "postproc.csv", index=False)
P = R.pivot(index="fold", columns="variant", values="S"); d = P.sub(P["base"], axis=0).drop(columns="base")
print("\nchange in S vs current post-processing (mean / worst fold / folds improved of %d):" % len(P)); print(pd.DataFrame({"mean": d.mean(), "worst": d.min(), "improved": (d > 0).sum()}).round(4).to_string())
print("\nby family:\n", d.join(R.drop_duplicates("fold").set_index("fold").family).groupby("family").mean().round(4).to_string())
