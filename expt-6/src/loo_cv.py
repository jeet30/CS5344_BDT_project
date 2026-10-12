import numpy as np, pandas as pd, warnings, sys, json; warnings.filterwarnings("ignore")
import harness as h
name = sys.argv[1] if len(sys.argv) > 1 else "unsw"; gen = sys.argv[2] if len(sys.argv) > 2 else "dvs"
df, cats, nums = h.load(name); F = h.make_folds(df, cats, nums)
cols = nums + cats
def const(s, drop):
    s = s.copy()
    for c in drop: s[c] = s[c].mode().iloc[0] if c in cats else s[c].median()
    return s
rows = []; syn = {}
for fn, (tr_i, te_i, fam) in F.items():
    tr, te = df.loc[tr_i], df.loc[te_i]; s = h.GENS[gen](tr, h.N_SYN[name], 0, cats, nums); syn[fn] = (s, te)
    b = h.score(s, te, cats, nums, seeds=2)[2]; rows.append((fn, "_base", b))
    for c in cols: rows.append((fn, c, h.score(const(s, [c]), te, cats, nums, seeds=2)[2] - b))
    print(fn, "base", round(b, 4), flush=True)
L = pd.DataFrame(rows, columns=["fold", "feat", "d"]); L.to_csv(f"../out/loo_const_{name}_{gen}.csv", index=False)
M = L[L.feat != "_base"].pivot(index="feat", columns="fold", values="d")
base = L[L.feat == "_base"].set_index("fold").d
# cross-fold selection: pick features helped on average by the OTHER folds (and not hurt on a majority), set constant, score held-out fold
print("\nheld-out fold: base -> after constant-ing features selected from OTHER folds")
res = []
for fn in F:
    oth = [f for f in F if f != fn]
    for thr in (0.0, 0.003):
        sel = [c for c in cols if M.loc[c, oth].mean() > thr and (M.loc[c, oth] > 0).mean() >= 0.6]
        s, te = syn[fn]; sc = h.score(const(s, sel), te, cats, nums, seeds=2)[2]
        res.append((fn, thr, len(sel), base[fn], sc, sc - base[fn]))
R = pd.DataFrame(res, columns=["fold", "thr", "n_sel", "base", "after", "delta"]); print(R.round(4).to_string(index=False))
print("\nmean delta by thr:\n", R.groupby("thr").agg(n_sel=("n_sel", "mean"), mean_delta=("delta", "mean"), worst_delta=("delta", "min"), folds_improved=("delta", lambda x: (x > 0).sum())).round(4))
R.to_csv(f"../out/loo_cv_{name}_{gen}.csv", index=False)
