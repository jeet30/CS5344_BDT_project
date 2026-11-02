"""Expt 19f (UNSW harness, blend base WITHOUT the built-in stretch): which attack-direction gate to stretch with?
V0 current: direction agrees in >=75% of the K=4 attack clusters (unweighted) | V1 agreement weighted by cluster size >=0.75 | V2 weighted >=0.5 | V3 direction of the LARGEST cluster only.
Recipe after the gate: weights x3 (14-cell theta), jitter, stretch 10x (pipeline semantics), uniform fill SEL4, caps."""
import sys, time
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-19/src")
from blend_harness import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def cluster_auc(tr, nums, K=4, seed=0, max_norm=6000):
    nm = tr[tr.is_anomaly == 0]; ab = tr[tr.is_anomaly == 1]
    X = StandardScaler().fit_transform(np.log1p(np.abs(ab[nums].to_numpy(float)))); cl = pd.Series(KMeans(K, n_init=5, random_state=seed).fit_predict(X), index=ab.index)
    nms = nm.sample(min(max_norm, len(nm)), random_state=seed); sizes = cl.value_counts().sort_index(); A = {}
    for c in nums:
        if nm[c].nunique() <= 10: continue
        a = []
        for k in range(K):
            v = ab.loc[cl.index[cl == k], c].to_numpy(float)
            a.append(roc_auc_score(np.r_[np.ones(len(v)), np.zeros(len(nms))], np.r_[v, nms[c].to_numpy(float)]) if len(v) >= 30 else np.nan)
        A[c] = np.array(a)
    return A, sizes.to_numpy(float)

def dirs(A, sizes, variant, min_auc=0.58):
    out = {}
    for c, a in A.items():
        ok = ~np.isnan(a); w = sizes * ok; w = w / w.sum(); up = a >= min_auc; dn = a <= 1 - min_auc
        if variant == "V0": u, d = (up[ok].mean() >= 0.75), (dn[ok].mean() >= 0.75)
        elif variant == "V1": u, d = (w[up].sum() >= 0.75), (w[dn].sum() >= 0.75)
        elif variant == "V2": u, d = (w[up].sum() >= 0.5), (w[dn].sum() >= 0.5)
        elif variant == "V3": big = int(np.nanargmax(sizes * ok)); u, d = up[big], dn[big]
        if u and not d: out[c] = 1
        elif d and not u: out[c] = -1
    return out

def stretch(x, D, real, factor):
    x = x.copy()
    for c, d in D.items():
        v = x[c].to_numpy(float).copy()
        if d > 0: u = np.quantile(v, .10); m = v < u; v[m] = u + (v[m] - u) * factor
        else: u = np.quantile(v, .90); m = v > u; v[m] = u + (v[m] - u) * factor
        v = np.maximum(v, real[c].min()); x[c] = np.rint(v) if np.allclose(real[c], np.rint(real[c])) else v
    return x

if __name__ == "__main__":
    C = Ctx("blend", theta_old()); W = C.W; t0 = time.time(); rows = []; feats_by = {}
    configs = [("none", None, 1), ("V0", "V0", 10), ("V1", "V1", 10), ("V2", "V2", 10), ("V3", "V3", 10), ("V3 x3", "V3", 3)]
    for fn in W.F:
        tr = W.df.loc[W.F[fn][0]]; A, sz = cluster_auc(tr, W.nums); base = C.jitter(C.resampled(fn))
        for name, v, f in configs:
            D = {} if v is None else {c: d for c, d in dirs(A, sz, v).items() if c not in SEL4}; feats_by.setdefault(name, []).append(set(D))
            rows.append((fn, W.fam[fn], name, len(D), C.sc(fn, C.caps(C.fill(stretch(base, D, C.nm, f) if D else base, SEL4)))))
        print(fn, "[%.0fs]" % (time.time() - t0), flush=True)
    R = pd.DataFrame(rows, columns=["fold", "family", "variant", "n_cols", "S"]); R.to_csv(OUT + "gate_targeted.csv", index=False)
    b = R[R.variant == "V0"].set_index("fold").S; R["d_vs_V0"] = R.S - R.fold.map(b)
    print(R.groupby("variant").agg(S=("S", "mean"), d_vs_V0=("d_vs_V0", "mean"), worst=("d_vs_V0", "min"), up=("d_vs_V0", lambda x: int((x > 0).sum())), n_cols=("n_cols", "mean")).round(4).to_string())
    print(R.groupby(["variant", "family"]).d_vs_V0.mean().unstack().round(4).to_string())
    for name in ("V0", "V1", "V2", "V3"):
        cnt = pd.Series([c for s in feats_by[name] for c in s]).value_counts(); print(name, "columns used in >=10/14 folds:", sorted(cnt[cnt >= 10].index))
