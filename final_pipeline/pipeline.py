"""One common pipeline for NSL-KDD and UNSW-NB15 (parameters differ in config.py, code does not).

    python final_pipeline/pipeline.py unsw   [--out FILE]
    python final_pipeline/pipeline.py nslkdd [--out FILE]

Stages (each is a no-op when its parameter is neutral):
 1 generate   : dvs_copula (contrastive reweighting + protocol-conditioned Gaussian copula + dispersion)  |  real_resample (bootstrap of real normals)
 2 weights    : resample the pool with replacement by category-cell weights  w = exp(scale * theta[cell])
 3 stretch    : columns whose attack-vs-normal direction agrees across attack clusters get the OPPOSITE tail stretched (never below the observed minimum)
 4 constraints: physical-law projection (dataset rules)
 5 neutralize : hold a data-selected set of columns constant (selector: stability-gated leave-one-fold-out table | single-feature ECOD lift)
 6 bounds     : hard physical caps
"""
from __future__ import annotations
import argparse, contextlib, io, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pyod.models.ecod import ECOD
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))     # cleaner.py, constraints.py, dvs_copula_v2.py
from config import CONFIG

LABEL = "is_anomaly"


# ----------------------------------------------------------------------------- data
def load(cfg):
    df = pd.read_csv(cfg["train_csv"])
    cats = [c for c in df.columns if c != LABEL and not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c != LABEL and c not in cats]
    return df, cats, nums


# ----------------------------------------------------------------------------- stage 1
def generate(df, cats, nums, cfg):
    if cfg["generator"] == "dvs_copula":
        from dvs_copula_v2 import ContrastiveDVSCopulaGenerator
        with contextlib.redirect_stdout(io.StringIO()):
            g = ContrastiveDVSCopulaGenerator(random_state=cfg["seed"], **cfg["dvs"]).fit(df)
        return g.sample(cfg["n"])
    if cfg["generator"] == "itemset_copula_hybrid":            # teammate's generator (vendor/generators/): all scrubbed real normals + per-itemset-cluster copula rows with tail slack
        from generators.itemset_copula_hybrid import ItemsetCopulaHybridSampler
        normal = pd.read_csv(cfg["foundation_csv"])
        return ItemsetCopulaHybridSampler(random_state=cfg["seed"], **cfg["hybrid"]).fit(normal).sample(cfg["n"])
    if cfg["generator"] == "real_resample":
        nm = df[df[LABEL] == 0].drop(columns=LABEL)
        return nm.sample(cfg["n"], replace=True, random_state=cfg["seed"]).reset_index(drop=True)
    raise ValueError(cfg["generator"])


# ----------------------------------------------------------------------------- stage 2
def apply_weights(s, cats, cfg):
    w_cfg = cfg["weights"]
    if not w_cfg["scale"]: return s
    T = json.load(open(w_cfg["theta"])); idx = {c: i for i, c in enumerate(T["cells"])}; th = np.array(T["theta"] + [0.0])
    k = s[cats].astype(str).agg("|".join, axis=1).map(idx).fillna(len(T["theta"])).astype(int).to_numpy()
    w = np.exp(w_cfg["scale"] * th[k]); w /= w.sum()
    return s.iloc[np.random.default_rng(cfg["seed"]).choice(len(s), size=len(s), p=w)].reset_index(drop=True)


# ----------------------------------------------------------------------------- stage 2b (optional)
def jitter_duplicates(s, df, cfg):
    """tiny Gaussian jitter on the REPEATED rows only (repeats come from the with-replacement weighting); continuous columns, never below the observed minimum."""
    jc = cfg.get("jitter")
    if not jc: return s
    nm = df[df[LABEL] == 0]; cont = [c for c in nm.columns if c != LABEL and pd.api.types.is_numeric_dtype(nm[c]) and nm[c].nunique() > 100 and not np.allclose(nm[c], np.rint(nm[c]))]
    mask = s.duplicated(keep="first").to_numpy() if jc.get("duplicates_only", True) else np.ones(len(s), bool); rng = np.random.default_rng(jc.get("seed", cfg["seed"])); s = s.copy()
    for c in cont: s.loc[mask, c] = np.maximum(s.loc[mask, c].to_numpy() + rng.normal(0, jc["sigma"] * nm[c].std(), int(mask.sum())), nm[c].min())
    return s


# ----------------------------------------------------------------------------- stage 3
def feature_direction(df, nums, K, min_auc, min_agree, seed, max_norm=6000):
    """{col: +1 (attacks consistently higher) | -1 (consistently lower)} when the direction agrees across attack clusters."""
    nm = df[df[LABEL] == 0]; ab = df[df[LABEL] == 1]
    X = StandardScaler().fit_transform(np.log1p(np.abs(ab[nums].to_numpy(float))))
    cl = pd.Series(KMeans(K, n_init=5, random_state=seed).fit_predict(X), index=ab.index)
    nms = nm.sample(min(max_norm, len(nm)), random_state=seed); out = {}
    for c in nums:
        if nm[c].nunique() <= 10: continue
        a = []
        for k in range(K):
            v = ab.loc[cl.index[cl == k], c].to_numpy(float)
            if len(v) < 30: continue
            y = np.r_[np.ones(len(v)), np.zeros(len(nms))]; a.append(roc_auc_score(y, np.r_[v, nms[c].to_numpy(float)]))
        if not a: continue
        a = np.array(a); up = (a >= min_auc).mean() >= min_agree; dn = (a <= 1 - min_auc).mean() >= min_agree
        if up and not dn: out[c] = 1
        elif dn and not up: out[c] = -1
    return out


def stretch_tails(s, df, nums, cfg):
    st = cfg["stretch"]
    if st["factor"] == 1.0: return s, []
    dirs = feature_direction(df, nums, st["K"], st["min_auc"], st["min_agree"], cfg["seed"]); real = df[df[LABEL] == 0]; s = s.copy(); touched = []
    for c, d in dirs.items():
        v = s[c].to_numpy(float).copy()
        if d > 0: u = np.quantile(v, 0.10); m = v < u; v[m] = u + (v[m] - u) * st["factor"]
        else: u = np.quantile(v, 0.90); m = v > u; v[m] = u + (v[m] - u) * st["factor"]
        v = np.maximum(v, real[c].min())
        if np.allclose(real[c], np.rint(real[c])): v = np.rint(v)
        if (v != s[c].to_numpy(float)).any(): touched.append(c)
        s[c] = v
    return s, touched


# ----------------------------------------------------------------------------- stage 4
def apply_constraints(s, cfg):
    if cfg["constraints"] == "unsw":
        from constraints import enforce_unsw_constraints; return enforce_unsw_constraints(s)
    if cfg["constraints"] == "nslkdd":
        from constraints import enforce_nslkdd_constraints; return enforce_nslkdd_constraints(s)
    return s


# ----------------------------------------------------------------------------- stage 5
def dominant_share(x): return float(x.value_counts(normalize=True).iloc[0])


def select_loo_table(df, cats, nums, nc):
    """stability-gated selection on a leave-one-fold-out table (fold, feat, d): mean gain > thr, positive in >= vote of folds, stable across fold-bootstraps,
    bounded worst-3 loss, never a column where one value holds >= guard of the normals."""
    L = pd.read_csv(nc["table"]); M = L[~L.feat.str.startswith("_")].pivot(index="feat", columns="fold", values="d"); normals = df[df[LABEL] == 0]
    rng = np.random.default_rng(0); F = M.shape[1]; idx = rng.integers(0, F, (200, F)); out = []
    for c in nums + cats:
        if c not in M.index or dominant_share(normals[c]) >= nc["guard"]: continue
        v = M.loc[c].to_numpy()
        if not (v.mean() > nc["thr"] and (v > 0).mean() >= nc["vote"]): continue
        if (v[idx].mean(axis=1) > nc["thr"]).mean() < nc["freq"]: continue
        if np.sort(v)[:3].mean() < -nc["cvar_tol"]: continue
        out.append(c)
    return out


def select_single_feature(df, cats, nums, nc):
    """label-based: columns whose single-feature ECOD (fit on normals) detects attacks no better than chance, excluding sparse columns."""
    nm = df[df[LABEL] == 0]; y = df[LABEL].to_numpy(); chance = y.mean(); out = []
    for c in nums:
        if nm[c].nunique() < 2 or dominant_share(nm[c]) >= nc["guard"]: continue
        lift = average_precision_score(y, ECOD().fit(nm[[c]].to_numpy(float)).decision_function(df[[c]].to_numpy(float))) - chance
        if lift <= nc["lift_max"]: out.append((lift, c))
    return [c for _, c in sorted(out)]                                      # ascending lift => fixed order


def fill_columns(s, df, cats, cols, how, seed, skip=()):
    """neutralise columns: 'constant' = median (mode for categoricals); 'uniform' = independent uniform draw on the real normals' [q01, q99]."""
    s = s.copy(); nm = df[df[LABEL] == 0]; rng = np.random.default_rng(seed)
    for c in cols:
        dtype = s[c].dtype
        if how == "uniform" and c not in cats:
            x = rng.uniform(nm[c].quantile(.01), nm[c].quantile(.99), len(s))           # always drawn, so skipping a column leaves every other column's draws unchanged
            if c in skip: continue
            s[c] = np.rint(x) if np.allclose(nm[c], np.rint(nm[c])) else x
        else:
            s[c] = s[c].mode().iloc[0] if c in cats else s[c].median()
        if dtype.kind in "iu" or np.allclose(nm[c], np.rint(nm[c])) and c not in cats: s[c] = s[c].round().astype(dtype)   # keep integer columns integer
    return s


def neutralize(s, df, cats, nums, cfg):
    nc = cfg["neutralize"]
    if nc is None or nc.get("selector") is None: return s, []
    sel = {"loo_table": select_loo_table, "single_feature_ecod": select_single_feature}[nc["selector"]](df, cats, nums, nc)
    sel = sel + [c for c in nc.get("extra_columns", []) if c not in sel]          # later selection rounds (appended last so earlier uniform draws are unchanged)
    skip = set(nc.get("skip_columns", []))
    return fill_columns(s, df, cats, sel, nc.get("fill", "constant"), cfg["seed"], skip), [c for c in sel if c not in skip]


# ----------------------------------------------------------------------------- driver
def run(name, out=None, verbose=True):
    cfg = CONFIG[name]; df, cats, nums = load(cfg)
    s = generate(df, cats, nums, cfg)
    s = apply_weights(s, cats, cfg)
    s = jitter_duplicates(s, df, cfg)
    s, stretched = stretch_tails(s, df, nums, cfg)
    s = apply_constraints(s, cfg)
    s, const = neutralize(s, df, cats, nums, cfg)
    s = s.copy()
    for c, m in cfg["bounds"].items(): s[c] = np.minimum(s[c], m)
    s = s[[c for c in df.columns if c != LABEL]]; s.insert(0, "id", np.arange(len(s)))
    assert len(s) == cfg["n"] and not s.isna().any().any() and s.id.is_unique
    out = Path(out) if out else Path(__file__).resolve().parent / "out" / f"final_{name}.csv"; out.parent.mkdir(parents=True, exist_ok=True); s.to_csv(out, index=False)
    if verbose: print(f"[{name}] generator={cfg['generator']} weights_scale={cfg['weights']['scale']} stretch={cfg['stretch']['factor']}x on {stretched} | constants={const} | bounds={list(cfg['bounds'])} -> {out} {s.shape}")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", choices=list(CONFIG)); ap.add_argument("--out"); a = ap.parse_args(); run(a.dataset, a.out)
