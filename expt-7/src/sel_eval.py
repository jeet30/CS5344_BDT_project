import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from e7 import *
name = sys.argv[1]; gen = sys.argv[2]; tab = sys.argv[3]
df, cats, nums = h.load(name); F = h.make_folds(df, cats, nums); cols = nums + cats; normals = df[df.is_anomaly == 0]
L = pd.read_csv(tab); M = L[L.feat != "_base"].pivot(index="feat", columns="fold", values="d"); base = L[L.feat == "_base"].set_index("fold").d
def const(s, drop):
    s = s.copy()
    for c in drop: s[c] = s[c].mode().iloc[0] if c in cats else s[c].median()
    return s
rules = {"v1_thr.003": lambda M_, thr=.003: select_v1(M_, cols, thr), "v2_thr.003": lambda M_, thr=.003: select_v2(M_, cols, normals, thr),
         "v1_thr0": lambda M_: select_v1(M_, cols, 0.0), "v2_thr0": lambda M_: select_v2(M_, cols, normals, 0.0)}
rows = []
for fn, (tr_i, te_i, fam) in F.items():
    tr, te = df.loc[tr_i], df.loc[te_i]; s = h.GENS[gen](tr, h.N_SYN[name], 0, cats, nums); oth = [f for f in F if f != fn]
    for rn, r in rules.items():
        sel = r(M[oth]); sc = h.score(const(s, sel), te, cats, nums, seeds=2)[2] if sel else base[fn]; rows.append((fn, rn, len(sel), sc - base[fn]))
R = pd.DataFrame(rows, columns=["fold", "rule", "n_sel", "delta"])
g = R.groupby("rule").agg(n_sel=("n_sel", "mean"), mean=("delta", "mean"), worst=("delta", "min"), cvar3=("delta", lambda x: np.sort(x.to_numpy())[:3].mean()), improved=("delta", lambda x: int((x > 0).sum())))
print(f"\n[{name}] held-out effect of neutralising features chosen from OTHER folds ({len(F)} folds)"); print(g.round(4).to_string())
print("\nfull-data selections:"); 
for rn, r in rules.items(): print(f"  {rn:11s}", r(M))
guard = [c for c in cols if dominant_share(normals, c) >= 0.9]; print("\nprotected by sparse-support guard (dominant value >= 90% of normals):", guard)
R.to_csv(R7 + f"sel_eval_{name}.csv", index=False)
