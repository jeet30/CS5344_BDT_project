"""Expt 8: (5) cross-cluster consistency gating of tail edits, (6) low-dim weight optimisation, (7) shift diagnostics.
Builds on expt-6 harness + expt-7 (combo folds, smoothing)."""
import sys, io, contextlib
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-7/src")
from eb import *            # also registers combo folds + eb generators
R8 = h.ROOT + "expt-8/out/"


# ---------- item 5: cross-cluster consistency ----------
def feature_direction(tr, nums, K=4, min_auc=0.58, min_agree=0.75, seed=0, max_norm=6000):
    """Return {col: +1 (attacks consistently higher), -1 (consistently lower)} for columns whose direction agrees across attack clusters (computed on tr only)."""
    nm = tr[tr[h.LABEL] == 0]; ab = tr[tr[h.LABEL] == 1]
    X = h.StandardScaler().fit_transform(np.log1p(np.abs(ab[nums].to_numpy(float))))
    cl = pd.Series(h.KMeans(K, n_init=5, random_state=seed).fit_predict(X), index=ab.index)
    nms = nm.sample(min(max_norm, len(nm)), random_state=seed); out = {}
    for c in nums:
        if nm[c].nunique() <= 10: continue
        a = []
        for k in range(K):
            v = ab.loc[cl.index[cl == k], c].to_numpy(float)
            if len(v) < 30: continue
            y = np.r_[np.ones(len(v)), np.zeros(len(nms))]; a.append(roc_auc_score(y, np.r_[v, nms[c].to_numpy(float)]))
        if not a: continue
        a = np.array(a); up = ((a >= min_auc).mean() >= min_agree); dn = ((a <= 1 - min_auc).mean() >= min_agree)
        if up and not dn: out[c] = 1
        elif dn and not up: out[c] = -1
    return out


def g_gate(base, q=0.97, widen=1.0, min_auc=0.58, min_agree=0.75, valid=True):
    """On consistent columns: clip the attack-side tail of the synthetic data at its q-quantile; stretch the opposite tail beyond its (1-q_o) quantile by `widen`."""
    def f(tr, n, seed, cats, nums):
        syn = h.GENS[base](tr, n, seed, cats, nums).copy(); dirs = feature_direction(tr, nums, min_auc=min_auc, min_agree=min_agree, seed=seed); isint = {c: np.allclose(tr[c], np.rint(tr[c])) for c in dirs}
        for c, d in dirs.items():
            v = syn[c].to_numpy(float).copy()
            if q < 1.0:
                if d > 0: v = np.minimum(v, np.quantile(v, q))
                else: v = np.maximum(v, np.quantile(v, 1 - q))
            if widen != 1.0:                       # opposite tail = side attacks do NOT live on
                if d > 0: u = np.quantile(v, 0.10); m = v < u; v[m] = u + (v[m] - u) * widen
                else: u = np.quantile(v, 0.90); m = v > u; v[m] = u + (v[m] - u) * widen
            if valid: v = np.maximum(v, tr.loc[tr[h.LABEL] == 0, c].min())     # never leave the observed lower bound (e.g. negative durations)
            syn[c] = np.rint(v) if isint[c] else v
        return syn
    return f


for b in ("dvs", "bootstrap"):
    for q in (0.99, 0.95, 0.90): h.GENS[f"gate_{b}_q{int(q*100)}"] = g_gate(b, q)
    for w in (1.5, 3.0): h.GENS[f"gate_{b}_w{int(w*10)}"] = g_gate(b, 1.0, w)
    h.GENS[f"gate_{b}_q95w15"] = g_gate(b, 0.95, 1.5)
    for w in (1.5, 3.0, 6.0): h.GENS[f"gatev_{b}_w{int(w*10)}"] = g_gate(b, 1.0, w, valid=True)


def robust_table(name, gens, tag):
    h.run(name, gens, R8 + tag + "_"); F = pd.read_csv(R8 + f"{tag}_folds_{name}.csv")
    P = F.pivot_table(index="gen", columns="family", values="S"); g = F.groupby("gen").S.agg(mean="mean", worst="min", cvar3=lambda x: np.sort(x.to_numpy())[:3].mean())
    T = P.join(g).sort_values("mean", ascending=False); print("\nby family + robust\n", T.round(4).to_string()); T.to_csv(R8 + f"{tag}_robust_{name}.csv"); return T


if __name__ == "__main__":
    robust_table(sys.argv[1], sys.argv[2].split(","), sys.argv[3])
