"""Which attacks does our best file's detector pair catch? Train ECOD + IForest on a synthetic file, score every real training row, break the result down by attack cluster and by category cell."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "expt-6/src"); import harness as h
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from pyod.models.ecod import ECOD
df, cats, nums = h.load("unsw"); y = df.is_anomaly.to_numpy()
ab = df[y == 1]; X = StandardScaler().fit_transform(np.log1p(np.abs(ab[nums].to_numpy(float)))); cl = pd.Series(KMeans(4, n_init=10, random_state=42).fit_predict(X), index=ab.index)
cell = df[["proto", "service", "state"]].astype(str).agg("|".join, axis=1)
def detect(syn):
    Xa, Xb = h.feats(syn.drop(columns=["id"], errors="ignore"), df.drop(columns=h.LABEL), cats, nums)
    e = ECOD().fit(Xa).decision_function(Xb); f = np.mean([-IsolationForest(n_estimators=300, random_state=s, n_jobs=-1).fit(Xa).score_samples(Xb) for s in range(2)], axis=0)
    return (rankdata(e) + rankdata(f)) / (2 * len(e))
files = {"real normals (bootstrap)": df[y == 0].drop(columns=h.LABEL).sample(70000, replace=True, random_state=0), "unsw_product3 (best)": pd.read_csv("final_pipeline/scored/unsw_product3__0.61452.csv")}
res = {}
for name, syn in files.items():
    s = detect(syn); res[name] = s; print(f"\n== {name}: AUPRC {average_precision_score(y, s):.4f}  AUROC {roc_auc_score(y, s):.4f}")
nrm = y == 0; rows = []
for k in range(4):
    idx = cl.index[cl == k]; mask = np.zeros(len(df), bool); mask[idx] = True; sel = nrm | mask
    r = {"attack_cluster": k, "n_attacks": len(idx), "share": round(len(idx) / len(ab), 3), "top_cells": ", ".join(f"{c} {v:.0%}" for c, v in cell[idx].value_counts(normalize=True).head(2).items())}
    for name, s in res.items(): r["AUROC " + name.split(" (")[0]] = round(roc_auc_score(mask[sel], s[sel]), 3)
    rows.append(r)
print("\nby attack cluster (AUROC of that cluster vs all normals)"); print(pd.DataFrame(rows).to_string(index=False))
rows = []
for c in cell[y == 1].value_counts().head(7).index:
    m = (cell == c).to_numpy(); a = m & (y == 1); n_ = m & (y == 0); sel = a | n_
    r = {"cell": c, "attacks": int(a.sum()), "normals": int(n_.sum())}
    for name, s in res.items(): r["within-cell AUROC " + name.split(" (")[0]] = round(roc_auc_score(a[sel], s[sel]), 3) if n_.sum() > 20 else np.nan
    rows.append(r)
print("\nby category cell (attacks vs normals in the SAME cell)"); print(pd.DataFrame(rows).to_string(index=False))
