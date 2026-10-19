"""Jitter strength sweep on the current best base (sel3 uniform fill + caps): repeated rows only, sigma in {0.10, 0.25, 0.5, 1.0}; plus all-row jitter at 0.25."""
from common16 import *
rows = []
for fn in W.F:
    s0 = resampled(fn); dup = s0.duplicated(keep="first").to_numpy(); allm = np.ones(len(s0), bool)
    V = {"sel3_nojitter": caps(fill(s0, SEL3))}
    for sg in (0.10, 0.25, 0.5, 1.0): V[f"dup_{sg}"] = caps(fill(jitter(s0, sg, dup), SEL3))
    V["all_0.25"] = caps(fill(jitter(s0, 0.25, allm), SEL3))
    r = {k: sc(fn, v) for k, v in V.items()}; rows += [(fn, W.fam[fn], k, v) for k, v in r.items()]; print(fn, {k: round(v - r["sel3_nojitter"], 4) for k, v in r.items() if k != "sel3_nojitter"}, flush=True)
R = pd.DataFrame(rows, columns=["fold", "family", "variant", "S"]); R.to_csv(R16 + "jitter_sigma.csv", index=False)
P = R.pivot(index="fold", columns="variant", values="S"); d = P.sub(P["sel3_nojitter"], axis=0).drop(columns="sel3_nojitter")
print("\nchange in S vs sel3 without jitter:"); print(pd.DataFrame({"mean": d.mean(), "worst": d.min(), "improved": (d > 0).sum()}).round(4).to_string())
