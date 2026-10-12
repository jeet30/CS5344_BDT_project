"""Expt 6: shift-aware validation harness.
Every fold hides something the generator never sees, mimicking the hidden test split:
  std   : random 70/30 split (what earlier experiments used)
  atk_h : one attack cluster entirely unseen (novel attack)
  sig_c_g: a group of categorical values (e.g. UNSW state / NSL service) absent from generator's normals,
           present among test normals (unseen-but-legit normal behaviour)
  tail  : top 15% of normals along PC1 absent from generator (extrapolation)
Generators are fit on the fold's train rows only. Reported: mean S, worst-fold S, per-family means.
Calibration: known leaderboard scores of reproducible generators -> Spearman per fold family.
"""
import sys, io, json, time, contextlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from pyod.models.ecod import ECOD

ROOT = "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/"
sys.path.insert(0, ROOT + "expt-4/src")
LABEL = "is_anomaly"
N_SYN = {"nslkdd": 20000, "unsw": 35000}
# public-LB points for generators reproducible from expt-4 code (NSL bootstrap / phase7 are approximate "~" values from expt-4 notes)
LB = {"nslkdd": {"bootstrap": 0.92919, "starter": 0.93155, "cpb": 0.92993, "dvs": 0.8990},
      "unsw":   {"bootstrap": 0.42100, "starter": 0.43384, "cpb": 0.43872, "dvs": 0.53307}}


def load(name):
    df = pd.read_csv(ROOT + f"train_{name}.csv")
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c not in cats + [LABEL]]
    return df, cats, nums


# ---------------- folds ----------------
def make_folds(df, cats, nums, seed=0, k_atk=3):
    rng = np.random.default_rng(seed)
    nrm_idx = df.index[df[LABEL] == 0].to_numpy(); atk_idx = df.index[df[LABEL] == 1].to_numpy()
    is_tr_n = pd.Series(rng.random(len(df)) < 0.7, index=df.index)  # per-row 70% train for both classes
    X = StandardScaler().fit_transform(np.log1p(np.abs(df.loc[atk_idx, nums].to_numpy(float))))
    cl = pd.Series(KMeans(4, n_init=10, random_state=seed).fit_predict(X), index=atk_idx)
    big = cl.value_counts().index[:k_atk]
    base_tr = is_tr_n
    folds = {}
    folds["std"] = (df.index[base_tr].to_numpy(), df.index[~base_tr].to_numpy(), "std")
    for h in big:
        tr = df.index[base_tr & ~df.index.isin(cl.index[cl == h])]
        te = df.index[~base_tr & ~df.index.isin(cl.index[cl == h])].union(cl.index[cl == h])
        folds[f"atk{h}"] = (tr.to_numpy(), te.to_numpy(), "atk")
    # categorical signature holdouts
    for c in [x for x in cats if df[x].nunique() > 3][:2]:
        vc = df.loc[nrm_idx, c].value_counts()
        grp = {v: i % 3 for i, v in enumerate(vc.index)}
        for g in range(3):
            held = df[c].map(grp).eq(g) & (df[LABEL] == 0)
            if held.sum() < 200: continue
            tr = df.index[base_tr & ~held]
            te = df.index[(~base_tr & ~held) | held]
            folds[f"sig_{c}{g}"] = (tr.to_numpy(), te.to_numpy(), "sig")
    # PC1 tail of normals
    Z = StandardScaler().fit_transform(np.log1p(np.abs(df.loc[nrm_idx, nums].to_numpy(float))))
    pc = pd.Series(PCA(1, random_state=seed).fit_transform(Z)[:, 0], index=nrm_idx)
    heldt = df.index.isin(pc.index[pc > pc.quantile(.85)])
    folds["tail"] = (df.index[base_tr & ~heldt].to_numpy(), df.index[(~base_tr & ~heldt) | heldt].to_numpy(), "tail")
    return folds


# ---------------- scoring ----------------
def feats(train, test, cats, nums):
    sc = StandardScaler(); a = sc.fit_transform(train[nums].to_numpy(float)); b = sc.transform(test[nums].to_numpy(float))
    if cats:
        d = pd.get_dummies(train[cats]); a = np.hstack([a, d.to_numpy(float)])
        b = np.hstack([b, pd.get_dummies(test[cats]).reindex(columns=d.columns, fill_value=0).to_numpy(float)])
    return np.nan_to_num(a), np.nan_to_num(b)


def score(syn, te, cats, nums, seeds=3, trees=100):
    Xa, Xb = feats(syn, te.drop(columns=LABEL), cats, nums); y = te[LABEL].to_numpy()
    e = average_precision_score(y, ECOD().fit(Xa).decision_function(Xb))
    f = np.mean([average_precision_score(y, -IsolationForest(n_estimators=trees, random_state=s, n_jobs=-1).fit(Xa).score_samples(Xb)) for s in range(seeds)])
    return float(e), float(f), float((e + f) / 2)


# ---------------- generators: fn(train_labeled_df, n, seed, cats, nums) -> feature df ----------------
def g_bootstrap(tr, n, seed, cats, nums):
    return tr[tr[LABEL] == 0].drop(columns=LABEL).sample(n, replace=True, random_state=seed).reset_index(drop=True)


def g_starter(tr, n, seed, cats, nums):
    nm = tr[tr[LABEL] == 0].drop(columns=LABEL); rng = np.random.default_rng(seed)
    s = nm.sample(20, random_state=seed).sample(n, replace=True, random_state=seed).reset_index(drop=True)
    for c in nums:
        v = nm[c].to_numpy(float)
        if set(np.unique(v)) <= {0, 1} or v.std() == 0: continue
        x = np.clip(s[c].to_numpy(float) + rng.normal(0, .5 * v.std(), n), v.min(), v.max())
        s[c] = np.rint(x) if np.allclose(v, np.rint(v)) else x
    return s


def g_cpb(tr, n, seed, cats, nums):
    from generators.stratified_cpb_v2 import ControlledPartialBalancingV2Generator as G
    return G(0.3, 5, seed).fit(tr).sample(n)


def g_dvs(tr, n, seed, cats, nums):
    from generators.dvs_copula_v2 import ContrastiveDVSCopulaGenerator as G
    with contextlib.redirect_stdout(io.StringIO()):
        return G(random_state=seed, n_bags=2).fit(tr).sample(n)


def g_dvs_p(**kw):
    def f(tr, n, seed, cats, nums):
        from generators.dvs_copula_v2 import ContrastiveDVSCopulaGenerator as G
        with contextlib.redirect_stdout(io.StringIO()):
            return G(random_state=seed, **{"n_bags": 2, **kw}).fit(tr).sample(n)
    return f


_PUR = {}
def purity(tr, cats, nums, seed=0):
    """out-of-fold P(attack|x) for each normal row of tr, from a GBM trained on labeled rows (cached per fold)."""
    key = (len(tr), int(tr.index[0]), int(tr.index[-1]))
    if key not in _PUR:
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        from sklearn.model_selection import cross_val_predict
        X = pd.concat([tr[nums], pd.get_dummies(tr[cats]).astype(float)], axis=1).to_numpy(float)
        p = cross_val_predict(H(max_iter=100, random_state=seed), X, tr[LABEL].to_numpy(), cv=4, method="predict_proba")[:, 1]
        _PUR[key] = pd.Series(p, index=tr.index)
    return _PUR[key][tr[LABEL] == 0]


def g_pure(mode, thr=0.3, power=2.0, jitter=0.0):
    def f(tr, n, seed, cats, nums):
        p = purity(tr, cats, nums); nm = tr[tr[LABEL] == 0].drop(columns=LABEL)
        if mode == "drop": w = (p.to_numpy() <= thr).astype(float)
        elif mode == "pow": w = (1 - p.to_numpy()) ** power
        else: w = np.ones(len(nm))
        s = nm.sample(n, replace=True, weights=w / w.sum(), random_state=seed).reset_index(drop=True)
        if jitter:
            rng = np.random.default_rng(seed)
            for c in nums:
                v = nm[c].to_numpy(float)
                if set(np.unique(v)) <= {0, 1} or v.std() == 0: continue
                x = np.clip(s[c].to_numpy(float) + rng.normal(0, jitter * v.std(), n), v.min(), v.max())
                s[c] = np.rint(x) if np.allclose(v, np.rint(v)) else x
        return s
    return f


def g_clr(tr, n, seed, cats, nums):
    """DVS's CLR purification alone (no copula, no dispersion): isolates its contribution."""
    from generators.dvs_copula_v2 import ContrastiveDVSCopulaGenerator as G
    with contextlib.redirect_stdout(io.StringIO()):
        g = G(random_state=seed, n_bags=1).fit(tr)
    return g.purified_normal_df_.drop(columns=[LABEL], errors="ignore").sample(n, replace=True, random_state=seed).reset_index(drop=True)[tr.drop(columns=LABEL).columns]


def g_clrT(T=1.0, clip=10.0, tau_u=0.15, jitter=0.0):
    """CLR reweighting with temperature T; bootstrap straight from the weighted normals (no copula)."""
    def f(tr, n, seed, cats, nums):
        from generators.dvs_copula_v2 import ContrastiveDVSCopulaGenerator as G
        with contextlib.redirect_stdout(io.StringIO()):
            g = G(random_state=seed, n_bags=1, tau_utility=tau_u).fit(tr)
        nm = tr[tr[LABEL] == 0]; ab = tr[tr[LABEL] == 1]; lr = np.zeros(len(nm))
        for c in g.top_utility_cols_:
            pn = nm[c].map(nm[c].value_counts(normalize=True)).to_numpy(); pa = nm[c].map(ab[c].value_counts(normalize=True)).fillna(1e-4).to_numpy()
            lr += np.log(pn + 1e-6) - np.log(pa + 1e-6)
        w = np.exp(np.clip(T * lr, -clip, clip)); w /= w.sum()
        s = nm.drop(columns=LABEL).sample(n, replace=True, weights=w, random_state=seed).reset_index(drop=True)
        if jitter:
            rng = np.random.default_rng(seed)
            for c in nums:
                v = nm[c].to_numpy(float)
                if set(np.unique(v)) <= {0, 1} or v.std() == 0: continue
                x = np.clip(s[c].to_numpy(float) + rng.normal(0, jitter * v.std(), n), v.min(), v.max())
                s[c] = np.rint(x) if np.allclose(v, np.rint(v)) else x
        return s
    return f


def g_hard(thr=0.0, use_cats=True, maxu=60, keep_min=0.2, jitter=0.0):
    """hard purification on discrete (low-cardinality) columns: keep normals whose summed log(p_norm/p_attack) > thr."""
    def f(tr, n, seed, cats, nums):
        nm = tr[tr[LABEL] == 0]; ab = tr[tr[LABEL] == 1]
        cols = [c for c in nums if nm[c].nunique() <= maxu and nm[c].nunique() > 1] + (cats if use_cats else [])
        lr = np.zeros(len(nm))
        for c in cols:
            pn = nm[c].map(nm[c].value_counts(normalize=True)).to_numpy(); pa = nm[c].map(ab[c].value_counts(normalize=True)).fillna(1e-4).to_numpy()
            lr += np.log(pn + 1e-6) - np.log(pa + 1e-6)
        keep = lr > np.quantile(lr, 1 - 1) if False else lr > thr
        if keep.mean() < keep_min: keep = lr >= np.quantile(lr, 1 - keep_min)
        pool = nm[keep].drop(columns=LABEL)
        return pool.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    return f


def _lr_filter(tr, cats, nums, keep, maxu=60):
    nm = tr[tr[LABEL] == 0]; ab = tr[tr[LABEL] == 1]
    cols = [c for c in nums if 1 < nm[c].nunique() <= maxu] + cats; lr = np.zeros(len(nm))
    for c in cols:
        pn = nm[c].map(nm[c].value_counts(normalize=True)).to_numpy(); pa = nm[c].map(ab[c].value_counts(normalize=True)).fillna(1e-4).to_numpy()
        lr += np.log(pn + 1e-6) - np.log(pa + 1e-6)
    return pd.concat([nm[lr >= np.quantile(lr, 1 - keep)], ab])


def g_dvs_top(keep, **kw):
    def f(tr, n, seed, cats, nums):
        return g_dvs_p(**kw)(_lr_filter(tr, cats, nums, keep), n, seed, cats, nums)
    return f


def g_wanmix(m=0.1, q=0.5, base="dvs", cat_only=True):
    """DVS rows + a controlled slice m of 'cleaned' WAN normals: within the purged mode (sttl>=200),
    keep the q fraction of normals with best within-mode contrastive log-ratio (sub-regions attacks do not occupy)."""
    def f(tr, n, seed, cats, nums):
        nm = tr[tr[LABEL] == 0]; ab = tr[tr[LABEL] == 1]
        wn, wa = nm[nm.sttl >= 200], ab[ab.sttl >= 200]
        cols = cats + ([c for c in nums if 1 < wn[c].nunique() <= 60 and c not in ("sttl", "dttl")] if not cat_only else [])
        lr = np.zeros(len(wn))
        for c in cols:
            pn = wn[c].map(wn[c].value_counts(normalize=True)).to_numpy(); pa = wn[c].map(wa[c].value_counts(normalize=True)).fillna(1e-4).to_numpy()
            lr += np.log(pn + 1e-6) - np.log(pa + 1e-6)
        pool = wn[lr >= np.quantile(lr, 1 - q)].drop(columns=LABEL)
        nw = int(n * m); core = g_dvs_p()(tr, n - nw, seed, cats, nums)
        return pd.concat([core, pool.sample(nw, replace=True, random_state=seed)], ignore_index=True)
    return f


def g_dvs_cat(T=0.5, which=("service",), clip=3.0):
    """Skew categorical composition away from values attacks over-use (esp. service in WAN), then run DVS on the reweighted normals."""
    def f(tr, n, seed, cats, nums):
        nm = tr[tr[LABEL] == 0]; ab = tr[tr[LABEL] == 1]; cc = [c for c in which if c in cats] or cats; lr = np.zeros(len(nm))
        for c in cc:
            pn = nm[c].map(nm[c].value_counts(normalize=True)).to_numpy(); pa = nm[c].map(ab[c].value_counts(normalize=True)).fillna(1e-3).to_numpy()
            lr += np.clip(np.log(pn + 1e-6) - np.log(pa + 1e-6), -clip, clip)
        w = np.exp(T * lr); w /= w.sum(); rs = nm.sample(len(nm), replace=True, weights=w, random_state=seed)
        return g_dvs_p()(pd.concat([rs, ab]), n, seed, cats, nums)
    return f


def safe_values(tr, nums, cats, maxu=60, min_mass=0.5, max_atk=0.01):
    """data-driven sentinel rule: for a discrete column, a 'safe' value is one that holds >= min_mass of normals and <= max_atk of attacks."""
    nm = tr[tr[LABEL] == 0]; ab = tr[tr[LABEL] == 1]; out = {}
    for c in nums:
        if not (1 < nm[c].nunique() <= maxu): continue
        pn = nm[c].value_counts(normalize=True); pa = ab[c].value_counts(normalize=True)
        v = [k for k in pn.index if pn[k] >= min_mass and pa.get(k, 0) <= max_atk]
        if v: out[c] = v[0]
    return out


def g_reloc(eps=0.0, post=None, min_mass=0.5, keep_frac=1.0, jitter=0.0):
    """Keep ALL normals; relocate sentinel columns of off-mode rows to the safe value (with prob 1-eps). post='dvs' runs DVS on top."""
    def f(tr, n, seed, cats, nums):
        sv = safe_values(tr, nums, cats, min_mass=min_mass); rng = np.random.default_rng(seed)
        nm = tr[tr[LABEL] == 0].copy()
        for c, v in sv.items():
            off = (nm[c] != v).to_numpy(); mv = off & (rng.random(len(nm)) >= eps); nm.loc[mv, c] = v
        if post == "dvs":
            return g_dvs_p()(pd.concat([nm, tr[tr[LABEL] == 1]]), n, seed, cats, nums)
        return nm.drop(columns=LABEL).sample(n, replace=True, random_state=seed).reset_index(drop=True)
    return f


_WP = {}
def nonlan_pattack(tr, cats, nums, seed=0):
    """OOF P(attack | x) from a GBM trained ONLY on non-LAN rows (rows whose sentinel columns are off the safe value)."""
    key = ("w", len(tr), int(tr.index[0]), int(tr.index[-1]))
    if key not in _WP:
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        from sklearn.model_selection import cross_val_predict
        sv = safe_values(tr, nums, cats); off = np.zeros(len(tr), bool)
        for c, v in sv.items(): off |= (tr[c] != v).to_numpy()
        sub = tr[off]; X = pd.concat([sub[nums], pd.get_dummies(sub[cats]).astype(float)], axis=1).to_numpy(float)
        p = cross_val_predict(H(max_iter=120, random_state=seed), X, sub[LABEL].to_numpy(), cv=4, method="predict_proba")[:, 1]
        _WP[key] = (pd.Series(p, index=sub.index), sv)
    return _WP[key]


def g_wanclf(m=0.05, tau=0.05, reloc=0.0, relocate=True):
    """DVS rows + slice m of non-LAN normals the within-segment classifier is confident are normal (P(attack)<tau).
    relocate: move sentinel columns of that slice back to the safe value with prob 1-reloc."""
    def f(tr, n, seed, cats, nums):
        p, sv = nonlan_pattack(tr, cats, nums); nm = tr.loc[p.index[(tr.loc[p.index, LABEL] == 0) & (p < tau)]].drop(columns=LABEL).copy()
        rng = np.random.default_rng(seed)
        if relocate:
            for c, v in sv.items(): nm.loc[(rng.random(len(nm)) >= reloc), c] = v
        nw = int(n * m); core = g_dvs_p()(tr, n - nw, seed, cats, nums)
        if len(nm) == 0 or nw == 0: return core
        return pd.concat([core, nm.sample(nw, replace=True, random_state=seed)], ignore_index=True)
    return f


def g_reject(base="dvs", q=0.15, over=1.0):
    """Gemini idea: generate, score the pool with ECOD+IForest fit on REAL train normals, discard the q most 'anomalous' synthetic rows.
    q<0 : the opposite (discard the most central rows) as a control."""
    def f(tr, n, seed, cats, nums):
        gen = GENS[base]; pool = gen(tr, int(n / (1 - abs(q))) if q > 0 else n, seed, cats, nums)
        real = tr[tr[LABEL] == 0].drop(columns=LABEL)
        Xr, Xp = feats(real, pool, cats, nums)
        e = ECOD().fit(Xr).decision_function(Xp); i = -IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xr).score_samples(Xp)
        sc = pd.Series(e).rank(pct=True).to_numpy() + pd.Series(i).rank(pct=True).to_numpy()
        order = np.argsort(sc if q > 0 else -sc); return pool.iloc[order[:n]].reset_index(drop=True)
    return f


def g_coverage(tr, n, seed, cats, nums, power=0.5):
    """new candidate: tempered signature coverage (flatter than empirical over proto/service/state-like combos)."""
    nm = tr[tr[LABEL] == 0].drop(columns=LABEL).reset_index(drop=True)
    sig = nm[cats].astype(str).agg("|".join, axis=1) if cats else pd.Series("all", index=nm.index)
    cnt = sig.map(sig.value_counts()); w = cnt ** (power - 1.0)   # row weight => signature mass ~ count**power
    return nm.sample(n, replace=True, weights=w / w.sum(), random_state=seed).reset_index(drop=True)


GENS = {"rej_dvs10": g_reject("dvs", .10), "rej_dvs20": g_reject("dvs", .20), "rej_dvs_ctrl": g_reject("dvs", -.15), "rej_boot10": g_reject("bootstrap", .10), "rej_boot20": g_reject("bootstrap", .20),
        "wc_m02_t05": g_wanclf(.02, .05), "wc_m05_t05": g_wanclf(.05, .05), "wc_m10_t05": g_wanclf(.10, .05), "wc_m05_t02": g_wanclf(.05, .02), "wc_m10_t02": g_wanclf(.10, .02),
        "wc_m05_t05_keep": g_wanclf(.05, .05, relocate=False), "wc_m10_t02_keep": g_wanclf(.10, .02, relocate=False), "wc_m05_t05_r1": g_wanclf(.05, .05, reloc=.5),
        "rl0": g_reloc(0.0), "rl003": g_reloc(0.003), "rl02": g_reloc(0.02), "rl10": g_reloc(0.10), "rl003_dvs": g_reloc(0.003, "dvs"), "rl02_dvs": g_reloc(0.02, "dvs"),
        "dc_s025": g_dvs_cat(.25), "dc_s05": g_dvs_cat(.5), "dc_s1": g_dvs_cat(1.0), "dc_ss05": g_dvs_cat(.5, ("service", "state")), "dc_all05": g_dvs_cat(.5, ("service", "state", "proto")), "dc_all1": g_dvs_cat(1.0, ("service", "state", "proto")),
        "wm05": g_wanmix(.05), "wm10": g_wanmix(.10), "wm20": g_wanmix(.20), "wm30": g_wanmix(.30), "wm10_q25": g_wanmix(.10, .25), "wm20_q25": g_wanmix(.20, .25), "wm10_num": g_wanmix(.10, .5, cat_only=False), "wm20_num": g_wanmix(.20, .5, cat_only=False),
        "dvs_top50": g_dvs_top(.5), "dvs_top70": g_dvs_top(.7), "dvs_top85": g_dvs_top(.85),
        "hard0": g_hard(0.0), "hard_n2": g_hard(-2.0), "hard2": g_hard(2.0), "hard0_nocat": g_hard(0.0, False), "hard5": g_hard(5.0), "hard_top50": g_hard(1e9, keep_min=.5), "hard_top70": g_hard(1e9, keep_min=.7),
        "clrT05": g_clrT(.5), "clrT1": g_clrT(1.), "clrT15": g_clrT(1.5), "clrT2": g_clrT(2.), "clrT3": g_clrT(3., clip=30), "clrT2c30": g_clrT(2., clip=30),
        "clrT1_j2": g_clrT(1., jitter=.2), "clrT2_j2": g_clrT(2., jitter=.2),
        "clr": g_clr, "pure_drop10": g_pure("drop", .10), "pure_drop30": g_pure("drop", .30), "pure_pow2": g_pure("pow", power=2), "pure_pow4": g_pure("pow", power=4),
        "pure_drop10_j2": g_pure("drop", .10, jitter=.2), "pure_pow4_j2": g_pure("pow", power=4, jitter=.2),
        "dvs_a0": g_dvs_p(alpha_dispersion=0.0), "dvs_a15": g_dvs_p(alpha_dispersion=15.0), "dvs_a70": g_dvs_p(alpha_dispersion=70.0),
        "dvs_tl03": g_dvs_p(tau_low=0.03), "dvs_tl10": g_dvs_p(tau_low=0.10), "dvs_tu08": g_dvs_p(tau_utility=0.08), "dvs_tu25": g_dvs_p(tau_utility=0.25),
        "dvs_m20": g_dvs_p(shrinkage_m=20.0), "dvs_m400": g_dvs_p(shrinkage_m=400.0), "dvs_g200": g_dvs_p(min_group_size=200),
        "bootstrap": g_bootstrap, "starter": g_starter, "cpb": g_cpb, "dvs": g_dvs, "coverage": g_coverage}


def run(name, gens, out):
    df, cats, nums = load(name); folds = make_folds(df, cats, nums)
    rows = []
    for fn, (tr_i, te_i, fam) in folds.items():
        tr, te = df.loc[tr_i], df.loc[te_i]
        for g in gens:
            t = time.time(); syn = GENS[g](tr, N_SYN[name], 0, cats, nums)
            e, f, s = score(syn, te, cats, nums)
            rows.append(dict(dataset=name, fold=fn, family=fam, gen=g, ecod=e, iforest=f, S=s)); print(rows[-1], f"{time.time()-t:.0f}s", flush=True)
    R = pd.DataFrame(rows); R.to_csv(out + f"folds_{name}.csv", index=False)
    summ = R.groupby("gen").agg(mean_S=("S", "mean"), worst_S=("S", "min")).join(R.pivot_table(index="gen", columns="family", values="S"))
    lb = pd.Series(LB[name], name="LB"); summ = summ.join(lb)
    cal = {c: spearmanr(summ.loc[lb.index.intersection(summ.index), c], summ.loc[lb.index.intersection(summ.index), "LB"])[0] for c in ["mean_S", "worst_S", "std", "atk", "sig", "tail"] if c in summ}
    summ.to_csv(out + f"summary_{name}.csv"); print(summ.round(4).to_string()); print("Spearman vs LB:", {k: round(v, 3) for k, v in cal.items()})


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else list(GENS), ROOT + "expt-6/out/")
