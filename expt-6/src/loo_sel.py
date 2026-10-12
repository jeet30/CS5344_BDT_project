import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings("ignore")
import harness as h
name = sys.argv[1]; df, cats, nums = h.load(name); F = h.make_folds(df, cats, nums); cols = nums + cats
L = pd.read_csv(f"../out/loo_const_{name}.csv"); M = L[L.feat != "_base"].pivot(index="feat", columns="fold", values="d"); base = L[L.feat == "_base"].set_index("fold").d
def const(s, drop):
    s = s.copy()
    for c in drop: s[c] = s[c].mode().iloc[0] if c in cats else s[c].median()
    return s
res = []
for fn, (tr_i, te_i, fam) in F.items():
    tr, te = df.loc[tr_i], df.loc[te_i]; s = h.g_dvs(tr, h.N_SYN[name], 0, cats, nums); oth = [f for f in F if f != fn]
    for thr in (0.0, 0.003):
        sel = [c for c in cols if M.loc[c, oth].mean() > thr and (M.loc[c, oth] > 0).mean() >= 0.6]
        sc = h.score(const(s, sel), te, cats, nums, seeds=2)[2]; res.append((fn, thr, len(sel), base[fn], sc, sc - base[fn]))
R = pd.DataFrame(res, columns=["fold", "thr", "n_sel", "base", "after", "delta"]); print(R.round(4).to_string(index=False))
print(R.groupby("thr").agg(n_sel=("n_sel", "mean"), mean_delta=("delta", "mean"), worst_delta=("delta", "min"), folds_improved=("delta", lambda x: (x > 0).sum())).round(4))
R.to_csv(f"../out/loo_cv_{name}.csv", index=False)
allsel = [c for c in cols if (M.loc[c] > 0).mean() >= 0.6 and M.loc[c].mean() > 0.003]; print("features consistently helped by constant (all folds):", allsel)
