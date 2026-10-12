"""Item 6: low-dimensional row-weight optimisation over category cells with a simple evolution strategy, nested by fold family."""
import sys, json, time
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from e8 import *

def cells_from(df, cats, m):
    key = df[df.is_anomaly == 0][cats].astype(str).agg("|".join, axis=1); return list(key.value_counts().head(m).index)

class Weighter:
    def __init__(self, name, m=14, base="dvs"):
        self.name = name; self.df, self.cats, self.nums = h.load(name); self.F = h.make_folds(self.df, self.cats, self.nums); self.cells = cells_from(self.df, self.cats, m); self.m = m
        self.S0, self.cellid, self.te = {}, {}, {}
        for fn, (tr_i, te_i, fam) in self.F.items():
            tr = self.df.loc[tr_i]; s = h.GENS[base](tr, h.N_SYN[name], 0, self.cats, self.nums); k = s[self.cats].astype(str).agg("|".join, axis=1)
            idx = {c: i for i, c in enumerate(self.cells)}; self.S0[fn] = s; self.cellid[fn] = k.map(idx).fillna(m).astype(int).to_numpy(); self.te[fn] = self.df.loc[te_i]
        self.fam = {fn: v[2] for fn, v in self.F.items()}
    def S(self, theta, fn, seed=0, quick=True):
        th = np.r_[theta, 0.0]; w = np.exp(th[self.cellid[fn]]); w /= w.sum(); rng = np.random.default_rng(seed)
        syn = self.S0[fn].iloc[rng.choice(len(w), size=len(w), p=w)].reset_index(drop=True)
        return h.score(syn, self.te[fn], self.cats, self.nums, seeds=1 if quick else 3, trees=60 if quick else 100)[2]
    def obj(self, theta, folds, reg=0.02):
        return float(np.mean([self.S(theta, f) for f in folds]) - reg * np.mean(theta ** 2))
    def es(self, folds, gens=18, lam=8, sigma=0.6, seed=0, log=print):
        rng = np.random.default_rng(seed); mu = np.zeros(self.m); best = (self.obj(mu, folds), mu.copy()); log(f"  init {best[0]:.4f}")
        for g in range(gens):
            pop = mu + sigma * rng.standard_normal((lam, self.m)); fit = np.array([self.obj(p, folds) for p in pop]); top = np.argsort(-fit)[: lam // 2]
            mu = pop[top].mean(axis=0); sigma *= 0.9
            if fit[top[0]] > best[0]: best = (fit[top[0]], pop[top[0]].copy())
            log(f"  gen {g} best {best[0]:.4f}")
        return best[1]
    def report(self, theta, folds):
        base = {f: self.S(np.zeros(self.m), f, quick=False) for f in folds}; new = {f: self.S(theta, f, quick=False) for f in folds}
        return pd.DataFrame({"fold": folds, "family": [self.fam[f] for f in folds], "base": [base[f] for f in folds], "opt": [new[f] for f in folds]}).assign(delta=lambda d: d.opt - d.base)

if __name__ == "__main__":
    name = sys.argv[1]; W = Weighter(name); fams = {}
    for f, fam in W.fam.items(): fams.setdefault(fam, []).append(f)
    print(name, "folds by family:", {k: len(v) for k, v in fams.items()}, "cells:", W.cells[:6], "...", flush=True)
    results = []
    for hold in [("sig", "combo"), ("atk",)]:                      # nested: optimise on the other families, test on the held-out family
        train = [f for f in W.F if W.fam[f] not in hold]; test = [f for f in W.F if W.fam[f] in hold]
        t = time.time(); th = W.es(train, log=lambda s: None); R = W.report(th, test); R["heldout_families"] = "+".join(hold); results.append(R)
        print(f"\nheld-out {hold}: mean delta {R.delta.mean():+.4f}, worst {R.delta.min():+.4f}, improved {int((R.delta>0).sum())}/{len(R)}  [{time.time()-t:.0f}s]\n", R.round(4).to_string(index=False), flush=True)
    pd.concat(results).to_csv(R8 + f"weights_nested_{name}.csv", index=False)
    th = W.es(list(W.F), log=lambda s: None); json.dump({"cells": W.cells, "theta": th.tolist()}, open(R8 + f"weights_theta_{name}.json", "w"), indent=1); print("final theta (all folds):", dict(zip(W.cells, np.round(th, 2))))
