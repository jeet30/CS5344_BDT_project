"""Expt 19e (UNSW harness): strength of the contrastive purification (lr_power) and the utility threshold tau_utility, proto-conditioned DVS, full recipe (theta x3, jitter, uniform fill SEL4, caps)."""
import sys, time
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-19/src")
import blend_harness as bh
from blend_harness import *

def g_lr(power, tau):
    def f(tr, n, seed, cats, nums):
        from dvs_lr import ContrastiveDVSCopulaGenerator as G
        with contextlib.redirect_stdout(io.StringIO()): return G(random_state=seed, n_bags=2, lr_power=power, tau_utility=tau).fit(tr).sample(n)
    return f

if __name__ == "__main__":
    power, tau = float(sys.argv[1]), float(sys.argv[2]); tag = f"lr{power:g}_t{tau:g}"; t0 = time.time()
    h.GENS[tag] = g_lr(power, tau); h.GENS["gatev_" + tag] = g_gate(tag, 1.0, 6.0, valid=True)
    C = Ctx("gatev_" + tag, theta_old()); rows = []
    for fn in C.W.F:
        rows.append((tag, fn, C.W.fam[fn], C.sc(fn, C.W.S0[fn]), C.sc(fn, C.caps(C.fill(C.jitter(C.resampled(fn)), SEL4)))))
    R = pd.DataFrame(rows, columns=["tag", "fold", "family", "raw", "recipe"]); R.to_csv(OUT + f"purif_{tag}.csv", index=False); print(tag, "done %.0fs" % (time.time() - t0), R[["raw", "recipe"]].mean().round(4).to_dict())
