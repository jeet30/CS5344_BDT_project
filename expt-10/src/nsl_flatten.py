"""Expt 10 (NSL-KDD): neutralise only columns with NO single-feature ECOD signal (non-sparse, lift over chance <= ~0.01), on the bootstrap base (same rows as the 0.92919 file).
 const  : column held at its median (IForest never splits on it; for ECOD every differing test value gets the same extreme score = constant offset)
 uniform: independent uniform draw on the normals' [q01, q99] (flat ECDF => near-constant ECOD contribution; IForest sees noise)"""
import numpy as np, pandas as pd
R = "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/"; N = 40000; SEED = 42
df = pd.read_csv(R + "train_nslkdd.csv"); nm = df[df.is_anomaly == 0].drop(columns="is_anomaly").reset_index(drop=True)
base = nm.sample(N, replace=True, random_state=SEED).reset_index(drop=True)
assert base.equals(nm.sample(N, replace=True, random_state=SEED).reset_index(drop=True))
def make(cols, how, name):
    s = base.copy(); rng = np.random.default_rng(SEED)
    for c in cols:
        if how == "const": s[c] = nm[c].median()
        else:
            lo, hi = nm[c].quantile(.01), nm[c].quantile(.99); x = rng.uniform(lo, hi, N); s[c] = np.rint(x) if np.allclose(nm[c], np.rint(nm[c])) else x
        if np.allclose(nm[c], np.rint(nm[c])): s[c] = s[c].round().astype(nm[c].dtype)
    o = s.copy(); o.insert(0, "id", np.arange(N)); assert o.isna().sum().sum() == 0 and len(o) == N; o.to_csv(R + f"expt-10/submissions/{name}.csv", index=False)
    print(f"{name:20s} {how:8s} cols {cols} | other columns identical to bootstrap: {all((s[c]==base[c]).all() for c in base.columns if c not in cols)} | constant now: {[c for c in cols if s[c].nunique()==1]}")
two = ["srv_count", "srv_diff_host_rate"]; four = two + ["duration", "dst_host_count"]
make(two, "const", "nsl_const2"); make(four, "const", "nsl_const4"); make(four, "uniform", "nsl_unif4")
print("\nmedians used:", {c: float(nm[c].median()) for c in four})
