"""Expt 5: per-feature mode search. Start from real normal rows; each feature gets a
transform chosen to maximise ECOD+IForest AUPRC on labeled holdouts (leave-attack-cluster-out)."""
import sys, json, time
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from pyod.models.ecod import ECOD

ROOT = "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/"
LABEL = "is_anomaly"
MODES = ["real", "const", "uniform", "shuffle", "widen"]


def load(name):
    df = pd.read_csv(ROOT + f"train_{name}.csv")
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c not in cats + [LABEL]]
    return df, cats, nums


def feats(train, test, cats, nums):
    sc = StandardScaler()
    a = sc.fit_transform(train[nums].to_numpy(float)); b = sc.transform(test[nums].to_numpy(float))
    if cats:
        d = pd.get_dummies(train[cats]); cols = d.columns
        a = np.hstack([a, d.to_numpy(float)])
        b = np.hstack([b, pd.get_dummies(test[cats]).reindex(columns=cols, fill_value=0).to_numpy(float)])
    return np.nan_to_num(a), np.nan_to_num(b)


def detectors(Xa, Xb, y, seeds=3, trees=100):
    e = average_precision_score(y, ECOD().fit(Xa).decision_function(Xb))
    f = np.mean([average_precision_score(y, -IsolationForest(n_estimators=trees, random_state=s, n_jobs=-1).fit(Xa).score_samples(Xb)) for s in range(seeds)])
    return e, f


def apply_modes(base, modes, cols_info, rng):
    out = base.copy()
    for c, m in modes.items():
        if m == "real": continue
        v = base[c]
        if m == "const":
            out[c] = v.mode().iloc[0] if c in cols_info["cats"] else v.median()
        elif c in cols_info["cats"]:
            if m == "shuffle": out[c] = rng.permutation(v.to_numpy())
        elif m == "uniform":
            lo, hi = v.quantile(.01), v.quantile(.99)
            x = rng.uniform(lo, hi, len(v)); out[c] = np.rint(x) if c in cols_info["ints"] else x
        elif m == "shuffle":
            out[c] = rng.permutation(v.to_numpy())
        elif m == "widen":  # extend right tail with exponential slack, keep >=0 dtype
            x = v.to_numpy(float) + rng.exponential(0.5 * v.std() + 1e-12, len(v)) * (rng.random(len(v)) < .1)
            out[c] = np.rint(x) if c in cols_info["ints"] else x
    return out


def make_folds(df, nums, k, seed=0):
    ab = df[df[LABEL] == 1]
    Xs = StandardScaler().fit_transform(np.log1p(np.abs(ab[nums].to_numpy(float))))
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Xs)
    return pd.Series(lab, index=ab.index)


class Search:
    def __init__(self, name, k=4, seed=0):
        self.df, self.cats, self.nums = load(name)
        self.info = {"cats": set(self.cats), "ints": {c for c in self.nums if np.allclose(self.df[c], np.rint(self.df[c]))}}
        self.cl = make_folds(self.df, self.nums, k, seed)
        self.k = k
        rng = np.random.default_rng(seed)
        nrm = self.df[self.df[LABEL] == 0]
        self.is_a = pd.Series(rng.random(len(nrm)) < 0.5, index=nrm.index)  # normal split A (generator) / B (test)
        self.nrm = nrm; self.seed = seed

    def evaluate(self, modes, held, rng_seed=0, full=False):
        """held: cluster id used as test attacks. normals A -> synth base, B -> test. Other-cluster attacks unused by generator."""
        base = self.nrm[self.is_a].drop(columns=LABEL).reset_index(drop=True)
        rng = np.random.default_rng(rng_seed)
        syn = apply_modes(base, modes, self.info, rng)
        te = pd.concat([self.nrm[~self.is_a], (self.df.loc[self.cl.index] if held == -1 else self.df.loc[self.cl.index[self.cl == held]])])
        Xa, Xb = feats(syn, te.drop(columns=LABEL), self.cats, self.nums)
        return detectors(Xa, Xb, te[LABEL].to_numpy(), *( (10, 300) if full else (3, 100)))

    def score(self, modes, folds):
        r = [self.evaluate(modes, h) for h in folds]
        return float(np.mean([(e + f) / 2 for e, f in r])), r

    def greedy(self, folds, passes=2, log=print):
        cols = self.cats + self.nums
        modes = {c: "real" for c in cols}
        best, _ = self.score(modes, folds); log(f"init {best:.4f}")
        for p in range(passes):
            for c in cols:
                cur = modes[c]; cand = {}
                for m in MODES:
                    if m == cur: continue
                    modes[c] = m; cand[m] = self.score(modes, folds)[0]
                m = max(cand, key=cand.get)
                if cand[m] > best + 5e-4: best = cand[m]; modes[c] = m; log(f"p{p} {c}: {cur}->{m} {best:.4f}")
                else: modes[c] = cur
        return modes, best


if __name__ == "__main__":
    name = sys.argv[1]
    S = Search(name)
    t = time.time()
    # nested: search on 3 clusters' worth of folds, test on the 4th (honest estimate), then search on all
    res = {}
    base_all = {h: S.evaluate({c: "real" for c in S.cats + S.nums}, h, full=True) for h in (-1, 0, 1, 2, 3)}
    print("baseline real-normal per-fold (ECOD,IF):", base_all, flush=True)
    for h in (0, 2):
        others = [x for x in range(S.k) if x != h]
        modes, b = S.greedy(others, passes=1, log=lambda s: None)
        e, f = S.evaluate(modes, h, full=True)
        res[h] = {"search_score": b, "heldout": (e + f) / 2, "baseline_heldout": sum(base_all[h]) / 2}
        print(f"fold {h}: heldout {res[h]['heldout']:.4f} vs real-normal {res[h]['baseline_heldout']:.4f}  [{time.time()-t:.0f}s]", flush=True)
    modes, b = S.greedy([-1] + list(range(S.k)), passes=2, log=lambda s: print(s, flush=True))
    json.dump({"modes": modes, "search_score": b, "nested": res}, open(ROOT + f"expt-5/out/modes_{name}.json", "w"), indent=1)
    print("done", b, {c: m for c, m in modes.items() if m != "real"})
