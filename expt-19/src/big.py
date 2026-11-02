"""Expt 19b (UNSW harness only, blend base): two candidate big levers.
clf  : row-level weights exp(-beta * z), z = standardised attack-likeness logit of each generated row from a classifier fit on the fold's TRAIN rows (normal vs attack);
       'oracle' fits the classifier on the fold's TEST rows (cheating: upper bound for any row-level reweighting). beta=0 is the current recipe.
cells: 40 category cells instead of 14 for the weight fit (ES), in-sample and nested (held-out families)."""
import sys, time, json
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-19/src")
from blend_harness import *
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

def enc(df, cats):
    x = df.copy()
    for c in cats: x[c] = x[c].astype("category")
    return x

def attack_logit(fit_df, pool, cats, nums, seed=0):
    cols = nums + cats; X = enc(fit_df[cols], cats); y = fit_df.is_anomaly.to_numpy()
    clf = HGB(max_iter=150, learning_rate=0.1, categorical_features="from_dtype", random_state=seed).fit(X, y)
    P = pool[cols].copy()
    for c in cats: P[c] = pd.Categorical(P[c], categories=X[c].cat.categories)
    p = np.clip(clf.predict_proba(P)[:, 1], 1e-4, 1 - 1e-4); z = np.log(p / (1 - p)); return (z - z.mean()) / (z.std() + 1e-9)

def weighted(C, fn, z, beta):
    w = np.exp(np.r_[C.th, 0.0][C.W.cellid[fn]] - beta * z); w /= w.sum()
    return C.W.S0[fn].iloc[np.random.default_rng(0).choice(len(w), size=len(w), p=w)].reset_index(drop=True)

if __name__ == "__main__":
    part = sys.argv[1]; t0 = time.time()
    if part == "clf":
        C = Ctx("gatev_blend_w60", theta_old()); W = C.W; rows = []; betas = (0.0, 0.25, 0.5, 1.0, 2.0)
        for fn in W.F:
            tr_i, te_i, fam = W.F[fn]; tr = W.df.loc[tr_i]; te = W.te[fn]; pool = W.S0[fn]
            for kind, fit in (("train", tr), ("oracle", te)):
                z = attack_logit(fit, pool, W.cats, W.nums)
                for b in betas:
                    if kind == "oracle" and b == 0.0: continue
                    s = weighted(C, fn, z, b); rows.append((fn, fam, kind, b, C.sc(fn, C.caps(C.fill(C.jitter(s), SEL4)))))
            print(fn, "done [%.0fs]" % (time.time() - t0), flush=True)
        R = pd.DataFrame(rows, columns=["fold", "family", "kind", "beta", "S"]); R.to_csv(OUT + "clf.csv", index=False)
        b0 = R[(R.kind == "train") & (R.beta == 0)].set_index("fold").S
        R["d"] = R.S - R.fold.map(b0); print("\nmean S and delta vs beta=0 (current recipe), 14 folds")
        print(R.groupby(["kind", "beta"]).agg(S=("S", "mean"), d=("d", "mean"), worst=("d", "min"), up=("d", lambda x: int((x > 0).sum()))).round(4).to_string())
        print(R[R.kind == "train"].groupby(["family", "beta"]).d.mean().unstack().round(4).to_string())
    if part == "cells":
        W = get_W("gatev_blend_w60"); m = 40
        key = W.df[W.df.is_anomaly == 0][W.cats].astype(str).agg("|".join, axis=1); W.cells = list(key.value_counts().head(m).index); W.m = m; idx = {c: i for i, c in enumerate(W.cells)}
        for fn in W.F: W.cellid[fn] = W.S0[fn][W.cats].astype(str).agg("|".join, axis=1).map(idx).fillna(m).astype(int).to_numpy()
        print("cells cover %.1f%% of normals" % (100 * key.isin(W.cells).mean()), flush=True)
        for hold in [("sig", "combo"), ("atk",)]:
            tr = [f for f in W.F if W.fam[f] not in hold]; te = [f for f in W.F if W.fam[f] in hold]; th = W.es(tr, gens=30, lam=12, log=lambda s: None); R = W.report(th, te)
            print(f"m=40 held-out {hold}: mean delta {R.delta.mean():+.4f} worst {R.delta.min():+.4f} improved {(R.delta>0).sum()}/{len(R)} [{time.time()-t0:.0f}s]", flush=True)
        th = W.es(list(W.F), gens=30, lam=12, log=lambda s: None); R = W.report(th, list(W.F)); print("m=40 in-sample all folds: mean delta %+.4f worst %+.4f improved %d/14" % (R.delta.mean(), R.delta.min(), (R.delta > 0).sum()), flush=True)
        json.dump({"cells": W.cells, "theta": th.tolist()}, open(OUT + "weights_theta_unsw_blend40.json", "w"), indent=1)
