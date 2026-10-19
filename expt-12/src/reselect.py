"""Expt 12: re-run the stability-gated constant-column selection on the WEIGHTED base (6x gated stretch + x3 cell weights), the distribution the detectors now see."""
import sys, json, time
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
for p in ("expt-7/src", "expt-8/src"): sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/" + p)
from weights import *            # Weighter, h
from e7 import select_v2, select_v1
R12 = h.ROOT + "expt-12/out/"
SCALE = 3.0; OLD = ["sinpkt", "sjit"]

def const(s, cols, cats):
    s = s.copy()
    for c in cols: s[c] = s[c].mode().iloc[0] if c in cats else s[c].median()
    return s

if __name__ == "__main__":
    W = Weighter("unsw", base="gatev_dvs_w60"); th = np.array(json.load(open(R8 + "weights_theta_unsw.json"))["theta"]) * SCALE; cols = W.nums + W.cats
    base_syn = {}
    for fn in W.F:
        w = np.exp(np.r_[th, 0.0][W.cellid[fn]]); w /= w.sum(); rng = np.random.default_rng(0)
        base_syn[fn] = W.S0[fn].iloc[rng.choice(len(w), size=len(w), p=w)].reset_index(drop=True)
    rows = []; t = time.time()
    for fn in W.F:
        b = h.score(base_syn[fn], W.te[fn], W.cats, W.nums, seeds=2)[2]; rows.append((fn, "_base", b)); o = h.score(const(base_syn[fn], OLD, W.cats), W.te[fn], W.cats, W.nums, seeds=2)[2]; rows.append((fn, "_old_const2", o - b))
        for c in cols: rows.append((fn, c, h.score(const(base_syn[fn], [c], W.cats), W.te[fn], W.cats, W.nums, seeds=2)[2] - b))
        print(fn, "base %.4f | old 2 constants %+.4f  [%.0fs]" % (b, o - b, time.time() - t), flush=True)
    L = pd.DataFrame(rows, columns=["fold", "feat", "d"]); L.to_csv(R12 + "loo_weighted.csv", index=False)
    M = L[~L.feat.str.startswith("_")].pivot(index="feat", columns="fold", values="d"); normals = W.df[W.df.is_anomaly == 0]
    SH = dict(thr=0.003, vote=0.7, freq=0.8, cvar_tol=0.004, guard=0.9)
    print("\nselection on the weighted base (same rule/params as pipeline7):", select_v2(M, cols, normals, **SH))
    print("selection with thr=0 (looser):", select_v2(M, cols, normals, **{**SH, "thr": 0.0}))
    print("old selection (unweighted base):", OLD)
    top = M.mean(axis=1).sort_values(ascending=False).head(10).round(4); print("\ncolumns with the largest mean gain from being constant (weighted base):\n", top.to_string())
    # held-out check: choose from OTHER folds, apply to the held-out fold; compare with fixed old pair and with none
    res = []
    for fn in W.F:
        oth = [f for f in W.F if f != fn]; sel = select_v2(M[oth], cols, normals, **SH); b = L[(L.fold == fn) & (L.feat == "_base")].d.iloc[0]
        new = h.score(const(base_syn[fn], sel, W.cats), W.te[fn], W.cats, W.nums, seeds=2)[2] - b if sel else 0.0
        old = L[(L.fold == fn) & (L.feat == "_old_const2")].d.iloc[0]; res.append((fn, W.fam[fn], len(sel), old, new))
    Rr = pd.DataFrame(res, columns=["fold", "family", "n_new", "old_pair", "reselected"]); Rr.to_csv(R12 + "heldout_reselect.csv", index=False)
    print("\nheld-out effect (change in S vs no constants), mean over folds:  old pair %+.4f | reselected %+.4f ; worst: old %+.4f reselected %+.4f ; folds improved: old %d reselected %d / %d" % (Rr.old_pair.mean(), Rr.reselected.mean(), Rr.old_pair.min(), Rr.reselected.min(), (Rr.old_pair > 0).sum(), (Rr.reselected > 0).sum(), len(Rr)))
    print(Rr.groupby("family")[["old_pair", "reselected"]].mean().round(4).to_string())
