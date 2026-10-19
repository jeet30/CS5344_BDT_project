"""Second round of uniform-fill selection ON TOP of the current set (sinpkt, sjit, ct_dst_src_ltm) with dup-jitter 0.10 in the base: marginal gain of filling each further column."""
from common16 import *
SEL4 = SEL3 + ["ct_dst_ltm"]
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-7/src")
from e7 import select_v2
cols = [c for c in W.nums + W.cats if c not in SEL4]; rows = []; base = {}
for fn in W.F:
    s0 = resampled(fn); base[fn] = jitter(s0, 0.10, s0.duplicated(keep="first").to_numpy()); b = sc(fn, caps(fill(base[fn], SEL4))); rows.append((fn, "_base", b))
    for c in cols: rows.append((fn, c, sc(fn, caps(fill(base[fn], SEL4 + [c]))) - b))
    print(fn, "base %.4f" % b, flush=True)
L = pd.DataFrame(rows, columns=["fold", "feat", "d"]); L.to_csv(R16 + "round3.csv", index=False); M = L[~L.feat.str.startswith("_")].pivot(index="feat", columns="fold", values="d")
SH = dict(thr=0.003, vote=0.7, freq=0.8, cvar_tol=0.004, guard=0.9); print("\nround-2 selection (pipeline rule):", select_v2(M, cols, nm.assign(is_anomaly=0), **SH)); print("looser thr=0:", select_v2(M, cols, nm.assign(is_anomaly=0), **{**SH, "thr": 0.0}))
print("largest mean gains:\n", M.mean(axis=1).sort_values(ascending=False).head(8).round(4).to_string())
res = []
for fn in W.F:
    oth = [f for f in W.F if f != fn]; sel = select_v2(M[oth], cols, nm.assign(is_anomaly=0), **SH); b = L[(L.fold == fn) & (L.feat == "_base")].d.iloc[0]
    res.append((fn, W.fam[fn], len(sel), sc(fn, caps(fill(base[fn], SEL4 + sel))) - b if sel else 0.0))
Rr = pd.DataFrame(res, columns=["fold", "family", "n_new", "gain"]); print("\nheld-out gain of extra columns chosen from other folds: mean %+.4f worst %+.4f improved %d/%d | avg extra columns %.1f" % (Rr.gain.mean(), Rr.gain.min(), (Rr.gain > 0).sum(), len(Rr), Rr.n_new.mean())); print(Rr.groupby("family").gain.mean().round(4).to_string())
