import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "expt-6/src"); import harness as h
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
df, cats, nums = h.load("unsw"); y = df.is_anomaly.to_numpy()
ab = df[y == 1]; X = StandardScaler().fit_transform(np.log1p(np.abs(ab[nums].to_numpy(float)))); cl = pd.Series(KMeans(4, n_init=10, random_state=42).fit_predict(X), index=ab.index)
cell = df[["proto", "service", "state"]].astype(str).agg("|".join, axis=1).to_numpy(); c3 = df.index.isin(cl.index[cl == 3]); nrm = y == 0; cellmask = cell == "tcp|-|FIN"
sel = c3 | (nrm & cellmask); L = np.log1p(np.abs(df[nums].to_numpy(float))); Z = StandardScaler().fit(L[nrm & cellmask]).transform(L)
# 1) how many cluster-3 attacks sit outside the normals' [q01,q99] box on at least one / three features (what tail-based detectors can see)
nn = df.loc[nrm & cellmask, nums]; lo, hi = nn.quantile(.01), nn.quantile(.99); out = ((df[nums] < lo) | (df[nums] > hi)).sum(axis=1).to_numpy()
for k in (1, 3, 6): print(f"outside [q01,q99] on >= {k} columns: cluster-3 attacks {np.mean(out[c3] >= k):.2f} | tcp|-|FIN normals {np.mean(out[nrm & cellmask] >= k):.2f}")
# 2) joint (covariance-aware) distance vs per-feature tails
lw = LedoitWolf().fit(Z[nrm & cellmask]); d = lw.mahalanobis(Z); print("Mahalanobis (fit on tcp|-|FIN normals) AUROC cluster 3 vs same-cell normals: %.3f" % roc_auc_score(c3[sel], d[sel]))
tail = np.max(np.abs(Z), axis=1); print("max per-feature |z| (tail view) AUROC: %.3f" % roc_auc_score(c3[sel], tail[sel]))
print("sum of per-feature |z| AUROC: %.3f" % roc_auc_score(c3[sel], np.abs(Z).sum(axis=1)[sel]))
