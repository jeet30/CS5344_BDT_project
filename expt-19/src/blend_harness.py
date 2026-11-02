"""Expt 19 (UNSW harness): do the weights (theta) and the uniform-fill columns still fit once the generator is the proto+state DVS blend?
parts:  build -> cache per-fold blend pools | calib -> proto vs blend under the full recipe | select -> column rounds | theta -> refit weights."""
import sys, io, json, time, pickle, contextlib
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
ROOT = "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/"
for p in ("expt-7/src", "expt-8/src", "expt-12/src", "expt-13/src"): sys.path.insert(0, ROOT + p)
from reselect_uniform import *          # Weighter, h, R8, uni, const, SCALE
from e8 import g_gate
from e7 import select_v2
OUT = ROOT + "expt-19/out/"; CAPS = {"ct_dst_ltm": 100, "ct_src_ltm": 100, "dmean": 1500}
SH = dict(thr=0.003, vote=0.7, freq=0.8, cvar_tol=0.004, guard=0.9)
SEL4 = ["sinpkt", "sjit", "ct_dst_src_ltm", "ct_dst_ltm"]

def g_blend(tr, n, seed, cats, nums):
    from generators.dvs_copula_v2 import ContrastiveDVSCopulaGenerator as G
    fr = []
    for cond, k, d in (("proto", n - n // 2, 0), ("state", n // 2, 1000)):
        with contextlib.redirect_stdout(io.StringIO()): fr.append(G(random_state=seed + d, n_bags=2).fit(tr, primary_cat_col=cond).sample(k))
    return pd.concat(fr, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
h.GENS["blend"] = g_blend; h.GENS["gatev_blend_w60"] = g_gate("blend", 1.0, 6.0, valid=True)

def get_W(base):
    f = OUT + f"pool_{base}.pkl"
    try: d = pickle.load(open(f, "rb")); W = Weighter.__new__(Weighter); W.__dict__.update(d); return W
    except FileNotFoundError: pass
    W = Weighter("unsw", base=base); pickle.dump(W.__dict__, open(f, "wb")); return W

class Ctx:
    def __init__(s, base, theta):
        s.W = get_W(base); s.th = np.array(theta) * SCALE; s.nm = s.W.df[s.W.df.is_anomaly == 0]
        s.cont = [c for c in s.W.nums if s.nm[c].nunique() > 100 and not np.allclose(s.nm[c], np.rint(s.nm[c]))]; s.std = s.nm[s.cont].std()
    def resampled(s, fn):
        w = np.exp(np.r_[s.th, 0.0][s.W.cellid[fn]]); w /= w.sum(); return s.W.S0[fn].iloc[np.random.default_rng(0).choice(len(w), size=len(w), p=w)].reset_index(drop=True)
    def jitter(s, x, sigma=0.10, seed=7):
        x = x.copy(); m = x.duplicated(keep="first").to_numpy(); rng = np.random.default_rng(seed)
        for c in s.cont: x.loc[m, c] = np.maximum(x.loc[m, c].to_numpy() + rng.normal(0, sigma * s.std[c], int(m.sum())), s.nm[c].min())
        return x
    def fill(s, x, cols):
        x = uni(x, [c for c in cols if c not in s.W.cats], s.nm, s.W.cats); return const(x, [c for c in cols if c in s.W.cats], s.W.cats)
    def caps(s, x):
        x = x.copy()
        for c, m in CAPS.items(): x[c] = np.minimum(x[c], m)
        return x
    def sc(s, fn, x): return h.score(x, s.W.te[fn], s.W.cats, s.W.nums, seeds=2)[2]

def theta_old(): return json.load(open(R8 + "weights_theta_unsw.json"))["theta"]

if __name__ == "__main__":
    part = sys.argv[1]; t0 = time.time()
    if part == "build":
        for b in ("gatev_dvs_w60", "gatev_blend_w60"): get_W(b); print("built", b, "%.0fs" % (time.time() - t0), flush=True)
    elif part == "calib":
        rows = []
        for b in ("gatev_dvs_w60", "gatev_blend_w60"):
            C = Ctx(b, theta_old())
            for fn in C.W.F:
                s0 = C.resampled(fn); rows.append((b, fn, C.W.fam[fn], C.sc(fn, C.W.S0[fn]), C.sc(fn, C.caps(C.fill(C.jitter(s0), SEL4)))))
            print(b, "done %.0fs" % (time.time() - t0), flush=True)
        R = pd.DataFrame(rows, columns=["base", "fold", "family", "raw", "recipe"]); R.to_csv(OUT + "calib.csv", index=False)
        print(R.groupby("base")[["raw", "recipe"]].mean().round(4).to_string()); print(R.pivot(index="fold", columns="base", values="recipe").round(4).to_string())
        print(R.groupby(["base", "family"]).recipe.mean().unstack(0).round(4).to_string())
    elif part == "select":
        C = Ctx("gatev_blend_w60", theta_old()); W = C.W; cols = [c for c in W.nums + W.cats]; sel = []; base = {}
        for fn in W.F: s0 = C.resampled(fn); base[fn] = C.jitter(s0)
        for rnd in (1, 2, 3):
            rem = [c for c in cols if c not in sel]; rows = []
            for fn in W.F:
                b = C.sc(fn, C.caps(C.fill(base[fn], sel))); rows.append((fn, "_base", b))
                for c in rem: rows.append((fn, c, C.sc(fn, C.caps(C.fill(base[fn], sel + [c]))) - b))
                print(f"round {rnd} {fn} base {b:.4f} [{time.time()-t0:.0f}s]", flush=True)
            L = pd.DataFrame(rows, columns=["fold", "feat", "d"]); L.to_csv(OUT + f"select_round{rnd}.csv", index=False)
            M = L[~L.feat.str.startswith("_")].pivot(index="feat", columns="fold", values="d"); new = select_v2(M, rem, C.nm.assign(is_anomaly=0), **SH)
            print(f"ROUND {rnd} selected: {new}\n top mean gains:\n", M.mean(axis=1).sort_values(ascending=False).head(6).round(4).to_string(), flush=True)
            if not new: break
            sel += new
        print("FINAL blend selection:", sel, "| proto-base selection was", SEL4, flush=True); json.dump(sel, open(OUT + "blend_selection.json", "w"))
        for name, cs in (("blend-selected", sel), ("SEL4", SEL4), ("SEL4+trans_depth", SEL4 + ["trans_depth"])):
            v = [C.sc(fn, C.caps(C.fill(base[fn], cs))) for fn in W.F]; print(f"{name:18s} mean over 14 folds {np.mean(v):.4f} worst {np.min(v):.4f}", flush=True)
    elif part == "theta":
        W = get_W("gatev_blend_w60"); fams = {}
        for f, fam in W.fam.items(): fams.setdefault(fam, []).append(f)
        res = []
        for hold in [("sig", "combo"), ("atk",)]:
            tr = [f for f in W.F if W.fam[f] not in hold]; te = [f for f in W.F if W.fam[f] in hold]; th = W.es(tr, log=lambda s: None); R = W.report(th, te); res.append(R)
            print(f"held-out {hold}: mean delta {R.delta.mean():+.4f} worst {R.delta.min():+.4f} improved {(R.delta>0).sum()}/{len(R)} [{time.time()-t0:.0f}s]", flush=True)
        th = W.es(list(W.F), log=lambda s: None); json.dump({"cells": W.cells, "theta": th.tolist()}, open(OUT + "weights_theta_unsw_blend.json", "w"), indent=1)
        old = np.array(theta_old()); print("theta new:", np.round(th, 2).tolist(), "\ntheta old:", np.round(old, 2).tolist(), "\ncorr", np.corrcoef(th, old)[0, 1])
        for nm_, t in (("old theta", old), ("new theta", th)):
            v = [W.S(t, f, quick=False) for f in W.F]; print(f"{nm_}: mean {np.mean(v):.4f} (scale 1, no fill)", flush=True)
