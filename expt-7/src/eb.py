"""Expt 7b: empirical-Bayes-style smoothing of categorical combinations + 'unseen combination' folds."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from e7 import *

_orig_make_folds = h.make_folds
def make_folds_combo(df, cats, nums, seed=0, k_atk=3):
    F = _orig_make_folds(df, cats, nums, seed, k_atk)
    rng0 = np.random.default_rng(seed); is_tr = pd.Series(rng0.random(len(df)) < 0.7, index=df.index)
    key = df[cats].astype(str).agg("|".join, axis=1); nm = df[h.LABEL] == 0; cnt = key[nm].value_counts()
    cand = cnt[(cnt >= 3) & (cnt <= 80)].index.to_numpy()
    for s in (1, 2, 3):
        r = np.random.default_rng(100 + s); held_keys = set(r.choice(cand, size=max(1, len(cand) // 2), replace=False))
        held = nm & key.isin(held_keys)
        # keep only held triples whose every single value still appears in the remaining normal training rows
        F[f"combo{s}"] = (df.index[is_tr & ~held].to_numpy(), df.index[(~is_tr & ~held) | held].to_numpy(), "combo")
    return F
h.make_folds = make_folds_combo


def smooth_cats(syn, cond, cats, lam, seed):
    """for a share lam of rows, redraw every categorical except `cond` independently from P(col | cond) (conditional-independence shrinkage of the joint)."""
    rng = np.random.default_rng(seed); s = syn.copy(); m = rng.random(len(s)) < lam
    if m.sum() == 0: return s
    for c in cats:
        if c == cond: continue
        for g, idx in s[m].groupby(cond).groups.items():
            pool = syn.loc[syn[cond] == g, c].to_numpy()
            if len(pool): s.loc[idx, c] = rng.choice(pool, size=len(idx))
    return s


def g_eb(base, lam):
    def f(tr, n, seed, cats, nums):
        cond = "proto" if "proto" in cats else ("protocol_type" if "protocol_type" in cats else cats[0])
        return smooth_cats(h.GENS[base](tr, n, seed, cats, nums), cond, cats, lam, seed)
    return f
for b in ("dvs", "bootstrap"):
    for l in (0.05, 0.1, 0.2, 0.4): h.GENS[f"eb_{b}_{int(l*100)}"] = g_eb(b, l)

if __name__ == "__main__":
    name = sys.argv[1]; gens = sys.argv[2].split(",")
    h.run(name, gens, R7 + "eb_"); F = pd.read_csv(R7 + f"eb_folds_{name}.csv")
    P = F.pivot_table(index="gen", columns="family", values="S"); g = F.groupby("gen").S.agg(mean="mean", worst="min", cvar3=lambda x: np.sort(x.to_numpy())[:3].mean())
    print("\nby family + robust\n", P.join(g).round(4).to_string())
