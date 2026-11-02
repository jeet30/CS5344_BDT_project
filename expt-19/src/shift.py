"""Expt 19d (UNSW harness, blend base): shift the BULK of consistent-direction columns away from the attack side by lam * std (stretch only touches the opposite 10% tail)."""
import sys, time
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-19/src")
from blend_harness import *
from e8 import feature_direction

if __name__ == "__main__":
    C = Ctx("gatev_blend_w60", theta_old()); W = C.W; t0 = time.time(); rows = []; lams = (0.1, 0.25, 0.5, 1.0)
    for fn in W.F:
        tr = W.df.loc[W.F[fn][0]]; dirs = feature_direction(tr, W.nums, seed=0); base = C.jitter(C.resampled(fn)); b = C.sc(fn, C.caps(C.fill(base, SEL4)))
        sd = tr[tr.is_anomaly == 0][list(dirs)].std() if dirs else None
        for lam in lams:
            x = base.copy()
            for c, d in dirs.items():
                if c in SEL4: continue
                v = np.maximum(x[c].to_numpy(float) - d * lam * sd[c], C.nm[c].min()); x[c] = np.rint(v) if np.allclose(C.nm[c], np.rint(C.nm[c])) else v
            rows.append((fn, W.fam[fn], lam, C.sc(fn, C.caps(C.fill(x, SEL4))) - b))
        print(fn, "base %.4f gated=%s [%.0fs]" % (b, [c for c in dirs if c not in SEL4], time.time() - t0), flush=True)
    R = pd.DataFrame(rows, columns=["fold", "family", "lam", "d"]); R.to_csv(OUT + "shift.csv", index=False)
    print(R.groupby("lam").d.agg(mean="mean", worst="min", up=lambda x: int((x > 0).sum())).round(4).to_string()); print(R.groupby(["family", "lam"]).d.mean().unstack().round(4).to_string())
