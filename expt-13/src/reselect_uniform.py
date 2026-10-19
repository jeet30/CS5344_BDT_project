"""Expt 13: re-run the stability-gated column selection on UNSW with UNIFORM fill (the 0.57698 recipe) instead of constants.
Base = 6x gated stretch + x3 cell weights (as in expt-12); the effect of each column is measured by filling it with an independent uniform draw on the normals' [q01,q99]."""
import sys, json, time
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
for p in ("expt-7/src", "expt-8/src", "expt-12/src"): sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/" + p)
from reselect import *             # Weighter, h, R8, const, SCALE, OLD ...
from e7 import select_v2
R13 = h.ROOT + "expt-13/out/"

def uni(s, cols, nm, cats, seed=42):
    s = s.copy()
    for c in cols:
        rng = np.random.default_rng(seed + hash(c) % 10007)
        if c in cats: continue
        lo, hi = nm[c].quantile(.01), nm[c].quantile(.99); x = rng.uniform(lo, hi, len(s)); s[c] = np.rint(x) if np.allclose(nm[c], np.rint(nm[c])) else x
    return s

if __name__ == "__main__":
    W = Weighter("unsw", base="gatev_dvs_w60"); th = np.array(json.load(open(R8 + "weights_theta_unsw.json"))["theta"]) * SCALE; cols = W.nums + W.cats; nm = W.df[W.df.is_anomaly == 0]
    base_syn = {}
    for fn in W.F:
        w = np.exp(np.r_[th, 0.0][W.cellid[fn]]); w /= w.sum(); base_syn[fn] = W.S0[fn].iloc[np.random.default_rng(0).choice(len(w), size=len(w), p=w)].reset_index(drop=True)
    def fill(s, cs):   # numeric -> uniform draw, categorical -> mode (uniform over categories is not defined)
        s = uni(s, [c for c in cs if c not in W.cats], nm, W.cats); return const(s, [c for c in cs if c in W.cats], W.cats)
    rows = []; t = time.time()
    for fn in W.F:
        b = h.score(base_syn[fn], W.te[fn], W.cats, W.nums, seeds=2)[2]; rows.append((fn, "_base", b)); rows.append((fn, "_old_pair_uniform", h.score(fill(base_syn[fn], OLD), W.te[fn], W.cats, W.nums, seeds=2)[2] - b))
        for c in cols: rows.append((fn, c, h.score(fill(base_syn[fn], [c]), W.te[fn], W.cats, W.nums, seeds=2)[2] - b))
        print(fn, "base %.4f | old pair (uniform) %+.4f [%.0fs]" % (b, rows[-1 - len(cols)][2], time.time() - t), flush=True)
    L = pd.DataFrame(rows, columns=["fold", "feat", "d"]); L.to_csv(R13 + "loo_uniform.csv", index=False)
    M = L[~L.feat.str.startswith("_")].pivot(index="feat", columns="fold", values="d"); normals = nm.assign(is_anomaly=0)
    SH = dict(thr=0.003, vote=0.7, freq=0.8, cvar_tol=0.004, guard=0.9)
    print("\nselection with uniform fill (pipeline rule):", select_v2(M, cols, normals, **SH)); print("looser thr=0:", select_v2(M, cols, normals, **{**SH, "thr": 0.0}))
    print("columns with the largest mean gain from uniform fill:\n", M.mean(axis=1).sort_values(ascending=False).head(12).round(4).to_string())
    res = []
    for fn in W.F:
        oth = [f for f in W.F if f != fn]; sel = select_v2(M[oth], cols, normals, **SH); b = L[(L.fold == fn) & (L.feat == "_base")].d.iloc[0]
        new = h.score(fill(base_syn[fn], sel), W.te[fn], W.cats, W.nums, seeds=2)[2] - b if sel else 0.0; old = L[(L.fold == fn) & (L.feat == "_old_pair_uniform")].d.iloc[0]; res.append((fn, W.fam[fn], len(sel), old, new))
    Rr = pd.DataFrame(res, columns=["fold", "family", "n_new", "old_pair", "reselected"]); Rr.to_csv(R13 + "heldout_uniform.csv", index=False)
    print("\nheld-out (select from other folds): mean change vs untouched: old pair %+.4f | reselected %+.4f ; worst %+.4f | %+.4f ; improved %d | %d of %d" % (Rr.old_pair.mean(), Rr.reselected.mean(), Rr.old_pair.min(), Rr.reselected.min(), (Rr.old_pair > 0).sum(), (Rr.reselected > 0).sum(), len(Rr)))
    print(Rr.groupby("family")[["old_pair", "reselected"]].mean().round(4).to_string())
