"""Expt 17: fill shape / partial fill / weights-scale re-check on the current UNSW pipeline (weighted base, dup-jitter 0.10, 4 filled columns, caps), UNSW harness.
Uses the REAL pipeline.fill_columns so the tested code is the code that makes the files."""
import sys
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-16/src"); sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/final_pipeline")
from common16 import *               # W, th, jitter, caps, resampled, sc, SEL3, nm
import importlib.util
_spec = importlib.util.spec_from_file_location('final_pipeline_mod', '/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/final_pipeline/pipeline.py'); P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)
SEL4 = SEL3 + ["ct_dst_ltm"]; R17 = h.ROOT + "expt-17/out/"
def fs(s, shape="uniform", frac=1.0): return P.fill_columns(s, W.df, W.cats, SEL4, "uniform", 42, shape=shape, frac=frac)
def resampled_scale(fn, scale):
    w = np.exp(np.r_[th, 0.0][W.cellid[fn]] * (scale / 3.0)); w /= w.sum(); return W.S0[fn].iloc[np.random.default_rng(0).choice(len(w), size=len(w), p=w)].reset_index(drop=True)
rows = []
for fn in W.F:
    s0 = resampled(fn); b0 = jitter(s0, 0.10, s0.duplicated(keep="first").to_numpy())
    V = {"uniform (current)": caps(fs(b0)), "loguniform": caps(fs(b0, "loguniform")), "quantile (real marginal)": caps(fs(b0, "quantile")), "uniform_wide [min,max]": caps(fs(b0, "uniform_wide")),
         "uniform_narrow [q05,q95]": caps(fs(b0, "uniform_narrow")), "partial 50%": caps(fs(b0, frac=0.5)), "partial 75%": caps(fs(b0, frac=0.75)), "loguniform 75%": caps(fs(b0, "loguniform", 0.75))}
    for scale in (2.5, 3.5):
        s1 = resampled_scale(fn, scale); V[f"weights x{scale}"] = caps(fs(jitter(s1, 0.10, s1.duplicated(keep="first").to_numpy())))
    r = {k: sc(fn, v) for k, v in V.items()}; rows += [(fn, W.fam[fn], k, v) for k, v in r.items()]; print(fn, {k: round(v - r["uniform (current)"], 4) for k, v in r.items() if k != "uniform (current)"}, flush=True)
R = pd.DataFrame(rows, columns=["fold", "family", "variant", "S"]); R.to_csv(R17 + "fillshape.csv", index=False)
Pv = R.pivot(index="fold", columns="variant", values="S"); d = Pv.sub(Pv["uniform (current)"], axis=0).drop(columns="uniform (current)")
print("\nchange in S vs the current pipeline (mean / worst fold / folds improved of %d):" % len(Pv)); print(pd.DataFrame({"mean": d.mean(), "worst": d.min(), "improved": (d > 0).sum()}).round(4).sort_values("mean", ascending=False).to_string())
print("\nby family:\n", d.join(R.drop_duplicates("fold").set_index("fold").family).groupby("family").mean().round(4).T.to_string())
