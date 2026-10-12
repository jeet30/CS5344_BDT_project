"""Expt 6 unified pipeline (one code path for both datasets; only CFG differs).

Stage 1  generator   : contrastive-reweighted normal sampler. copula=True -> Contrastive DVS-Copula (expt-4);
                       copula=False -> plain resampling of real normals (the T=0 / no-copula end of the same family).
Stage 2  neutralise  : data-driven. For every feature, measure on leave-group-out folds (std / unseen attack cluster /
                       unseen categorical group / PC1 tail) how much holding that feature constant changes S.
                       Hold constant every feature with mean gain > thr that helps in >= vote of folds.
                       No feature name is hard-coded; thr/vote are the only knobs.
Stage 3  constraints : optional physical-law projection (dataset hook).
"""
import sys, os, io, contextlib, argparse
import numpy as np, pandas as pd
import harness as h
sys.path.insert(0, h.ROOT + "expt-4/src")

CFG = {
    "unsw":   dict(copula=True,  alpha=35.0, tau_utility=0.15, tau_low=0.05, n_bags=5, thr=0.003, vote=0.6, constraints="unsw", n=70000, wan_slice=None),
    # NSL-KDD: neutralisation OFF (thr=inf). thr=0.0 selected 11 features and scored 0.91799 on the public LB vs 0.92919 for plain bootstrap
    # (the NSL-KDD harness does not track the LB; ECOD is hurt by constant columns whose test values vary).
    "nslkdd": dict(copula=False, thr=float("inf"), vote=0.6, constraints=None, n=40000),
}


def generate(tr, n, seed, cfg, cats, nums):
    if cfg["copula"]:
        from generators.dvs_copula_v2 import ContrastiveDVSCopulaGenerator as G
        with contextlib.redirect_stdout(io.StringIO()):
            g = G(random_state=seed, alpha_dispersion=cfg["alpha"], tau_utility=cfg["tau_utility"], tau_low=cfg["tau_low"], n_bags=cfg["n_bags"]).fit(tr)
        return g.sample(n)
    return h.g_bootstrap(tr, n, seed, cats, nums)


def neutral_value(s, c, cats):
    return s[c].mode().iloc[0] if c in cats else s[c].median()


def select_features(table, cols, thr, vote):
    M = table[table.feat != "_base"].pivot(index="feat", columns="fold", values="d")
    return [c for c in cols if M.loc[c].mean() > thr and (M.loc[c] > 0).mean() >= vote]


def loo_table(name, cfg, cache):
    if os.path.exists(cache): return pd.read_csv(cache)
    df, cats, nums = h.load(name); F = h.make_folds(df, cats, nums); rows = []
    for fn, (tr_i, te_i, fam) in F.items():
        tr, te = df.loc[tr_i], df.loc[te_i]; s = generate(tr, h.N_SYN[name], 0, {**cfg, "n_bags": 2}, cats, nums)
        b = h.score(s, te, cats, nums, seeds=2)[2]; rows.append((fn, "_base", b))
        for c in nums + cats:
            t = s.copy(); t[c] = neutral_value(s, c, cats); rows.append((fn, c, h.score(t, te, cats, nums, seeds=2)[2] - b))
    T = pd.DataFrame(rows, columns=["fold", "feat", "d"]); T.to_csv(cache, index=False); return T


def run(name, cache, out):
    cfg = CFG[name]; df, cats, nums = h.load(name.split('_')[0])
    table = loo_table(name.split('_')[0], cfg, cache)
    sel = select_features(table, nums + cats, cfg["thr"], cfg["vote"]); print(name, "neutralised features:", sel)
    ws = cfg.get("wan_slice"); nw = int(cfg["n"] * ws["m"]) if ws else 0
    s = generate(df, cfg["n"] - nw, 42, cfg, cats, nums)
    if cfg["constraints"] == "unsw":
        from constraints import enforce_unsw_constraints; s = enforce_unsw_constraints(s)
    if ws:  # stage 1b: confident off-mode normals (within-segment classifier) re-added, sentinels partly relocated to the safe value
        p, sv = h.nonlan_pattack(df, cats, nums); rng = np.random.default_rng(42)
        pool = df.loc[p.index[(df.loc[p.index, h.LABEL] == 0) & (p < ws["tau"])]].drop(columns=h.LABEL).copy()
        for c, v in sv.items(): pool.loc[rng.random(len(pool)) >= ws["reloc"], c] = v
        print("slice pool", len(pool), "safe-value columns", sv); s = pd.concat([s, pool.sample(nw, replace=True, random_state=42)[s.columns]], ignore_index=True)
    for c in sel: s[c] = neutral_value(s, c, cats)
    s = s[[c for c in df.columns if c != h.LABEL]]; s.insert(0, "id", np.arange(len(s)))
    assert len(s) == cfg["n"] and s.isna().sum().sum() == 0
    s.to_csv(out, index=False); print("wrote", out, s.shape)


CFG["unsw_wanslice"] = {**CFG["unsw"], "wan_slice": dict(m=0.05, tau=0.05, reloc=0.5)}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--loo", default=None); ap.add_argument("--cache"); ap.add_argument("--out")
    a = ap.parse_args(); run(a.name, a.cache or h.ROOT + f"expt-6/out/loo_const_{a.name.split('_')[0]}_pipeline.csv", a.out or h.ROOT + f"expt-6/submissions/pipeline_{a.name}.csv")
