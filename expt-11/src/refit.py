"""Expt 11: refit the UNSW category weights as a main-effects model  w = exp(a[proto]+b[service]+c[state])  (about 17 params), warm-started from the
LB-best direction (x3 of the 14-cell fit), on a harness base that already contains the 6x gated stretch and the sinpkt/sjit constants."""
import sys, json, time
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-8/src")
from weights import *            # Weighter, h, R8 ...
R11 = h.ROOT + "expt-11/out/"
TOPK = {"proto": 4, "service": 8, "state": 5}; CONST = ["sinpkt", "sjit"]

class ME(Weighter):
    def __init__(self, name="unsw", base="gatev_dvs_w60"):
        super().__init__(name, m=14, base=base)          # S0 per fold from gated-DVS (stretch 6x); cellid unused here
        nm = self.df[self.df.is_anomaly == 0]; self.levels = {c: list(nm[c].value_counts().head(k).index) for c, k in TOPK.items()}
        self.offs = {}; o = 0
        for c in TOPK: self.offs[c] = o; o += len(self.levels[c])
        self.P = o; self.idx = {}
        for fn, s in self.S0.items():
            self.idx[fn] = {c: s[c].map({v: i for i, v in enumerate(self.levels[c])}).fillna(-1).astype(int).to_numpy() for c in TOPK}
        self.const_vals = {fn: {c: self.S0[fn][c].median() for c in CONST} for fn in self.S0}
    def logw(self, beta, fn):
        lw = np.zeros(len(self.S0[fn]))
        for c in TOPK:
            i = self.idx[fn][c]; b = np.r_[beta[self.offs[c]: self.offs[c] + len(self.levels[c])], 0.0]; lw += b[np.where(i < 0, len(self.levels[c]), i)]
        return lw
    def S(self, beta, fn, seed=0, quick=True):
        w = np.exp(self.logw(beta, fn)); w /= w.sum(); rng = np.random.default_rng(seed)
        syn = self.S0[fn].iloc[rng.choice(len(w), size=len(w), p=w)].reset_index(drop=True)
        for c in CONST: syn[c] = self.const_vals[fn][c]
        return h.score(syn, self.te[fn], self.cats, self.nums, seeds=1 if quick else 3, trees=60 if quick else 100)[2]
    def obj(self, beta, folds, reg=0.001): return float(np.mean([self.S(beta, f) for f in folds]) - reg * np.mean(beta ** 2))
    def es(self, folds, beta0, gens=18, lam=10, sigma=0.35, seed=0, log=print):
        rng = np.random.default_rng(seed); mu = beta0.copy(); best = (self.obj(mu, folds), mu.copy()); log(f"  start {best[0]:.4f}")
        for g in range(gens):
            pop = np.vstack([mu + sigma * rng.standard_normal(self.P) for _ in range(lam)]); fit = np.array([self.obj(p, folds) for p in pop]); top = np.argsort(-fit)[: lam // 2]
            mu = pop[top].mean(axis=0); sigma *= 0.92
            if fit[top[0]] > best[0]: best = (fit[top[0]], pop[top[0]].copy())
            log(f"  gen {g} best {best[0]:.4f}")
        return best[1]
    def project(self, theta_cells, cells, scale=3.0):
        """ridge-project the 14-cell weights (x scale) onto the main-effects design."""
        rows, y = [], []
        for cell, t in zip(cells, theta_cells):
            p, s, st = cell.split("|"); x = np.zeros(self.P)
            for c, v in (("proto", p), ("service", s), ("state", st)):
                if v in self.levels[c]: x[self.offs[c] + self.levels[c].index(v)] = 1
            rows.append(x); y.append(scale * t)
        X = np.array(rows); return np.linalg.solve(X.T @ X + 0.1 * np.eye(self.P), X.T @ np.array(y))
    def report(self, beta, folds):
        base = {f: self.S(self.beta0, f, quick=False) for f in folds}; new = {f: self.S(beta, f, quick=False) for f in folds}
        return pd.DataFrame({"fold": folds, "family": [self.fam[f] for f in folds], "baseline_x3": [base[f] for f in folds], "refit": [new[f] for f in folds]}).assign(delta=lambda d: d.refit - d.baseline_x3)

if __name__ == "__main__":
    mode = sys.argv[1]; W = ME(); T = json.load(open(R8 + "weights_theta_unsw.json")); W.beta0 = W.project(np.array(T["theta"]), T["cells"], 3.0)
    print("levels:", {c: W.levels[c] for c in TOPK}); print("warm start beta:", {f"{c}={v}": round(W.beta0[W.offs[c] + i], 2) for c in TOPK for i, v in enumerate(W.levels[c])}, flush=True)
    allf = list(W.F); fams = {}
    for f in allf: fams.setdefault(W.fam[f], []).append(f)
    if mode == "baseline":
        r = W.report(W.beta0, allf); z = W.report(np.zeros(W.P), allf).rename(columns={"refit": "no_weights"}); print(r.groupby("family")[["baseline_x3"]].mean().join(z.groupby("family")[["no_weights"]].mean()).round(4).to_string())
        print("mean over folds: x3 projected %.4f | no weights %.4f" % (r.baseline_x3.mean(), z.no_weights.mean()))
    elif mode == "nested":
        hold = sys.argv[2].split(","); train = [f for f in allf if W.fam[f] not in hold]; test = [f for f in allf if W.fam[f] in hold]; t = time.time()
        b = W.es(train, W.beta0, log=lambda s: print(s, flush=True)); R = W.report(b, test); R.to_csv(R11 + f"nested_{'_'.join(hold)}.csv", index=False)
        print(f"\nheld-out {hold}: mean delta {R.delta.mean():+.4f} worst {R.delta.min():+.4f} improved {int((R.delta>0).sum())}/{len(R)} [{time.time()-t:.0f}s]\n", R.round(4).to_string(index=False))
    else:
        b = W.es(allf, W.beta0, log=lambda s: print(s, flush=True)); json.dump({"levels": W.levels, "offs": W.offs, "beta": b.tolist(), "beta0": W.beta0.tolist()}, open(R11 + "beta_final.json", "w"), indent=1)
        print("final beta:", {f"{c}={v}": round(b[W.offs[c] + i], 2) for c in TOPK for i, v in enumerate(W.levels[c])})
