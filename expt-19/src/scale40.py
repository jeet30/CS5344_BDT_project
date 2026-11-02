"""Harness: recipe score with the 40-cell theta at several scales (old 14-cell theta x3 = the current recipe, for reference)."""
import sys, json
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-19/src")
from blend_harness import *
import blend_harness as bh
C = Ctx("gatev_blend_w60", theta_old()); W = C.W; rows = []
old_cellid = {fn: W.cellid[fn].copy() for fn in W.F}
T = json.load(open(OUT + "weights_theta_unsw_blend40.json")); idx = {c: i for i, c in enumerate(T["cells"])}; m = len(T["cells"])
cid40 = {fn: W.S0[fn][W.cats].astype(str).agg("|".join, axis=1).map(idx).fillna(m).astype(int).to_numpy() for fn in W.F}
for name, theta, cid, scales in (("old14", np.array(theta_old()), old_cellid, (0, 3)), ("new40", np.array(T["theta"]), cid40, (1, 2, 3))):
    for sc_ in scales:
        W.cellid = cid; C.th = theta * sc_
        for fn in W.F: rows.append((name, sc_, fn, W.fam[fn], C.sc(fn, C.caps(C.fill(C.jitter(C.resampled(fn)), SEL4)))))
        print(name, sc_, "done", flush=True)
R = pd.DataFrame(rows, columns=["theta", "scale", "fold", "family", "S"]); R.to_csv(OUT + "scale40.csv", index=False)
b = R[(R.theta == "old14") & (R.scale == 3)].set_index("fold").S; R["d"] = R.S - R.fold.map(b)
print(R.groupby(["theta", "scale"]).agg(S=("S", "mean"), d=("d", "mean"), worst=("d", "min"), up=("d", lambda x: int((x > 0).sum()))).round(4).to_string())
print(R.groupby(["theta", "scale", "family"]).d.mean().unstack().round(4).to_string())
