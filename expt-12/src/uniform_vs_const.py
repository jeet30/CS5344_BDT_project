"""UNSW (reliable harness): hold sinpkt/sjit CONSTANT vs fill them with independent UNIFORM draws on the normals' [q01,q99], on the weighted + stretched base."""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from reselect import *
W = Weighter("unsw", base="gatev_dvs_w60"); th = np.array(json.load(open(R8 + "weights_theta_unsw.json"))["theta"]) * SCALE; rows = []
nm = W.df[W.df.is_anomaly == 0]
def uni(s, cols, seed=42):
    s = s.copy(); rng = np.random.default_rng(seed)
    for c in cols:
        lo, hi = nm[c].quantile(.01), nm[c].quantile(.99); x = rng.uniform(lo, hi, len(s)); s[c] = np.rint(x) if np.allclose(nm[c], np.rint(nm[c])) else x
    return s
for fn in W.F:
    w = np.exp(np.r_[th, 0.0][W.cellid[fn]]); w /= w.sum(); s = W.S0[fn].iloc[np.random.default_rng(0).choice(len(w), size=len(w), p=w)].reset_index(drop=True)
    sc = lambda d: h.score(d, W.te[fn], W.cats, W.nums, seeds=2)[2]
    b = sc(s); rows.append((fn, W.fam[fn], sc(const(s, OLD, W.cats)) - b, sc(uni(s, OLD)) - b)); print(fn, np.round(rows[-1][2:], 4), flush=True)
R = pd.DataFrame(rows, columns=["fold", "family", "constant", "uniform"])
print("\nmean change in S vs untouched columns:", R[["constant", "uniform"]].mean().round(4).to_dict(), "| worst:", R[["constant", "uniform"]].min().round(4).to_dict(), "| folds uniform > constant: %d/%d" % ((R.uniform > R.constant).sum(), len(R)))
print(R.groupby("family")[["constant", "uniform"]].mean().round(4).to_string())
