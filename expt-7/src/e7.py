"""Expt 7: (1) group-robust selection, (2) stability selection + sparse-support guard, (3) extreme-value (GPD) tails.
Reuses expt-6 harness (folds/scoring/generators)."""
import sys, io, contextlib
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import genpareto
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-6/src")
import harness as h
R7 = h.ROOT + "expt-7/out/"


# ---------- (2) selection rules ----------
def dominant_share(df, c):
    return float(df[c].value_counts(normalize=True).iloc[0])


def select_v1(M, cols, thr, vote=0.6):
    return [c for c in cols if M.loc[c].mean() > thr and (M.loc[c] > 0).mean() >= vote]


def select_v2(M, cols, normals, thr, vote=0.7, freq=0.8, cvar_tol=0.004, guard=0.9, k_worst=3, B=200, seed=0):
    """guard: never touch a column whose dominant value holds >= guard of normals (zero-inflated/sparse-support: tails carry unseen-attack signal).
    stability: feature must clear mean>thr in >= freq of B fold-bootstrap resamples. robust: mean of its k worst folds must be >= -cvar_tol."""
    rng = np.random.default_rng(seed); F = M.shape[1]; idx = rng.integers(0, F, (B, F)); out = []
    for c in cols:
        if dominant_share(normals, c) >= guard: continue
        v = M.loc[c].to_numpy()
        if not (v.mean() > thr and (v > 0).mean() >= vote): continue
        if (v[idx].mean(axis=1) > thr).mean() < freq: continue
        if np.sort(v)[:k_worst].mean() < -cvar_tol: continue
        out.append(c)
    return out


# ---------- (3) extreme-value tail extension ----------
def evt_extend(syn, real_norm, cols, ext=3.0, q_u=0.95, xi_max=0.5):
    """Rank-preserving upper-tail replacement: rows of `syn` above the real q_u quantile are mapped onto a GPD fitted to real exceedances.
    ext = return-period multiplier of the largest quantile (1 ~ observed max; >1 extends beyond it)."""
    s = syn.copy()
    for c in cols:
        x = real_norm[c].to_numpy(float); u = np.quantile(x, q_u); ex = x[x > u] - u
        if len(ex) < 50 or ex.std() == 0: continue
        xi, _, sc = genpareto.fit(ex, floc=0); xi = float(np.clip(xi, -0.3, xi_max)); sc = max(float(sc), 1e-12)
        v = s[c].to_numpy(float).copy(); m = v > u
        if m.sum() < 10: continue
        r = pd.Series(v[m]).rank(method="first").to_numpy(); q = (r - 0.5) / m.sum() * (1 - 1.0 / (ext * len(ex)))
        new = u + genpareto.ppf(q, xi, loc=0, scale=sc)
        if real_norm[c].nunique() <= 50 or np.allclose(x, np.rint(x)) and real_norm[c].nunique() <= 50: continue
        v[m] = new; s[c] = np.rint(v) if np.allclose(x, np.rint(x)) else v
    return s


def g_evt(base, ext, q_u=0.95):
    def f(tr, n, seed, cats, nums):
        real = tr[tr[h.LABEL] == 0]; syn = h.GENS[base](tr, n, seed, cats, nums)
        return evt_extend(syn, real, [c for c in nums if real[c].nunique() > 50], ext, q_u)
    return f


h.GENS.update({f"evt_{b}_{e}": g_evt(b, float(e)) for b in ("dvs", "bootstrap") for e in (1, 3, 10, 30)})
h.GENS["evt_dvs_3_q90"] = g_evt("dvs", 3.0, 0.90)


def smooth_cats(syn, cond, cats, lam, seed):
    """conditional-independence shrinkage: for a share lam of rows, redraw every categorical except `cond` independently from P(col | cond)."""
    rng = np.random.default_rng(seed); s = syn.copy(); m = rng.random(len(s)) < lam
    if m.sum() == 0: return s
    for c in cats:
        if c == cond: continue
        for g, idx in s[m].groupby(cond).groups.items():
            pool = syn.loc[syn[cond] == g, c].to_numpy()
            if len(pool): s.loc[idx, c] = rng.choice(pool, size=len(idx))
    return s


def summarize(name, gens, k_worst=3):
    h.run(name, gens, R7)
    F = pd.read_csv(R7 + f"folds_{name}.csv")
    g = F.groupby("gen").S.agg(mean="mean", worst="min", cvar3=lambda x: np.sort(x.to_numpy())[:k_worst].mean())
    print("\nrobust summary\n", g.round(4).to_string()); g.to_csv(R7 + f"robust_{name}.csv")


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2].split(","))
