import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "expt-6/src"); import harness as h
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import cross_val_predict, StratifiedKFold
df, cats, nums = h.load("unsw"); y = df.is_anomaly.to_numpy()
ab = df[y == 1]; X = StandardScaler().fit_transform(np.log1p(np.abs(ab[nums].to_numpy(float)))); cl = pd.Series(KMeans(4, n_init=10, random_state=42).fit_predict(X), index=ab.index)
cell = df[["proto", "service", "state"]].astype(str).agg("|".join, axis=1)
c3 = df.index.isin(cl.index[cl == 3]); nrm = y == 0; tcpfin = (cell == "tcp|-|FIN").to_numpy()
pool = pd.read_csv("final_pipeline/scored/unsw_product3__0.61452.csv")
def ecdf_score(ref, x):                     # ECOD-style per-feature score: -log of the smaller tail probability under ref's ECDF
    r = np.sort(ref); lo = np.searchsorted(r, x, side="right") / len(r); hi = 1 - np.searchsorted(r, x, side="left") / len(r)
    return -np.log(np.clip(np.minimum(lo, hi), 1 / (len(r) + 1), 1))
real = df.loc[nrm]; rows = []
for c in nums:
    if real[c].nunique() < 3: continue
    v = df[c].to_numpy(float); sel_cell = c3 | (nrm & tcpfin); sel_all = c3 | nrm
    raw_cell = roc_auc_score(c3[sel_cell], v[sel_cell]); raw_all = roc_auc_score(c3[sel_all], v[sel_all])
    e_pool = roc_auc_score(c3[sel_all], ecdf_score(pool[c].to_numpy(float), v)[sel_all]); e_real = roc_auc_score(c3[sel_all], ecdf_score(real[c].to_numpy(float), v)[sel_all])
    rows.append((c, raw_cell, raw_all, e_real, e_pool))
R = pd.DataFrame(rows, columns=["feature", "raw AUROC vs same-cell normals", "raw AUROC vs all normals", "ECOD-view AUROC (real normals)", "ECOD-view AUROC (our pool)"]).set_index("feature")
R["signal"] = (R.iloc[:, 0] - 0.5).abs(); R = R.sort_values("signal", ascending=False)
print("cluster 3 vs normals, per feature (0.5 = no signal; raw >0.5 = attacks higher)"); print(R.drop(columns="signal").head(16).round(3).to_string())
sel = c3 | (nrm & tcpfin); Xs = df.loc[sel, nums + cats].copy()
for c in cats: Xs[c] = Xs[c].astype("category")
p = cross_val_predict(HGB(max_iter=150, categorical_features="from_dtype", random_state=0), Xs, c3[sel], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print("\nsupervised ceiling, cluster 3 vs tcp|-|FIN normals (5-fold HGB, all features):", round(roc_auc_score(c3[sel], p), 3), "| best single feature raw:", round(0.5 + R.signal.iloc[0], 3))
print("n cluster-3 attacks:", int(c3.sum()), "| tcp|-|FIN normals:", int((nrm & tcpfin).sum()))
