"""Expt 19c (UNSW harness, blend base): per-column action menu on top of the current recipe (weights x3, jitter, uniform fill of SEL4, caps).
For each numeric column not already filled and each action, score the change per fold (leave-one-fold-out table, like the fill selection but with richer actions).
actions: L10 = stretch lower tail (below q10) x10 | R10 = stretch upper tail (above q90) x10 | B5 = both tails x5 | CLIP = clip at q03/q97 | uniform fill is the reference (already in the pipeline)."""
import sys, time
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-19/src")
from blend_harness import *

def op(x, name, real):
    v = x.to_numpy(float).copy()
    if name == "L10": u = np.quantile(v, .10); m = v < u; v[m] = u + (v[m] - u) * 10
    elif name == "R10": u = np.quantile(v, .90); m = v > u; v[m] = u + (v[m] - u) * 10
    elif name == "B5":
        a, b = np.quantile(v, .10), np.quantile(v, .90); v = np.where(v < a, a + (v - a) * 5, np.where(v > b, b + (v - b) * 5, v))
    elif name == "CLIP": v = np.clip(v, *np.quantile(v, [.03, .97]))
    v = np.maximum(v, real.min())
    return np.rint(v) if np.allclose(real, np.rint(real)) else v

if __name__ == "__main__":
    acts = sys.argv[1].split(","); C = Ctx("gatev_blend_w60", theta_old()); W = C.W; t0 = time.time()
    cols = [c for c in W.nums if c not in SEL4 and C.nm[c].nunique() > 10]; base = {fn: C.jitter(C.resampled(fn)) for fn in W.F}; rows = []
    for fn in W.F:
        b = C.sc(fn, C.caps(C.fill(base[fn], SEL4)))
        for a in acts:
            for c in cols:
                x = base[fn].copy(); x[c] = op(x[c], a, C.nm[c]); rows.append((fn, a, c, C.sc(fn, C.caps(C.fill(x, SEL4))) - b))
        print(fn, "base %.4f [%.0fs]" % (b, time.time() - t0), flush=True)
    pd.DataFrame(rows, columns=["fold", "action", "feat", "d"]).to_csv(OUT + "ops_" + "_".join(acts) + ".csv", index=False)
