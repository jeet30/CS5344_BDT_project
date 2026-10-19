import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
for p in ("expt-7/src", "expt-8/src", "expt-12/src", "expt-13/src"): sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/" + p)
from reselect_uniform import *       # Weighter, h, R8, uni, const, SCALE
R16 = h.ROOT + "expt-16/out/"; CAPS = {"ct_dst_ltm": 100, "ct_src_ltm": 100, "dmean": 1500}; SEL3 = ["sinpkt", "sjit", "ct_dst_src_ltm"]
W = Weighter("unsw", base="gatev_dvs_w60"); th = np.array(json.load(open(R8 + "weights_theta_unsw.json"))["theta"]) * SCALE
nm = W.df[W.df.is_anomaly == 0]; cont = [c for c in W.nums if nm[c].nunique() > 100 and not np.allclose(nm[c], np.rint(nm[c]))]; std = nm[cont].std()
def jitter(s, sigma, mask, seed=7):
    s = s.copy(); rng = np.random.default_rng(seed)
    for c in cont: s.loc[mask, c] = np.maximum(s.loc[mask, c].to_numpy() + rng.normal(0, sigma * std[c], int(mask.sum())), nm[c].min())
    return s
def fill(s, cols):
    s = uni(s, [c for c in cols if c not in W.cats], nm, W.cats); return const(s, [c for c in cols if c in W.cats], W.cats)
def caps(s):
    s = s.copy()
    for c, m in CAPS.items(): s[c] = np.minimum(s[c], m)
    return s
def resampled(fn):
    w = np.exp(np.r_[th, 0.0][W.cellid[fn]]); w /= w.sum(); return W.S0[fn].iloc[np.random.default_rng(0).choice(len(w), size=len(w), p=w)].reset_index(drop=True)
def sc(fn, s): return h.score(s, W.te[fn], W.cats, W.nums, seeds=2)[2]
