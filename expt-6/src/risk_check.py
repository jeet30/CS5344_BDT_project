import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import harness as h
from sklearn.metrics import roc_auc_score as auc
pd.set_option("display.width", 200)
df, cats, nums = h.load("unsw"); K = 6
C4 = ["sinpkt", "sjit", "proto", "state"]
ab = df[df.is_anomaly == 1]; nm = df[df.is_anomaly == 0]
X = h.StandardScaler().fit_transform(np.log1p(np.abs(ab[nums].to_numpy(float))))
cl = pd.Series(h.KMeans(K, n_init=10, random_state=1).fit_predict(X), index=ab.index)
print("clusters (size, top service, top state, sttl segment):")
for k in range(K):
    a = ab.loc[cl.index[cl == k]]
    print(k, len(a), a.service.value_counts(normalize=True).head(2).round(2).to_dict(), a.state.value_counts(normalize=True).head(2).round(2).to_dict(), "WAN %.2f" % (a.sttl >= 200).mean())
# 1. how much does each of the 4 columns separate each attack cluster from normals?
print("\nsingle-column separability: AUROC (cluster vs ALL normals; numeric cols) / share of cluster in a category rare among normals (<2%)")
rows = []
for k in range(K):
    a = ab.loc[cl.index[cl == k]]; r = {"cluster": k, "n": len(a)}
    for c in ["sinpkt", "sjit"]:
        v = np.concatenate([a[c], nm[c]]); y = np.r_[np.ones(len(a)), np.zeros(len(nm))]; r[c] = round(max(auc(y, v), 1 - auc(y, v)), 3)
    for c in ["proto", "state"]:
        fn = nm[c].value_counts(normalize=True); r[c + "_rare%"] = round(float((a[c].map(fn).fillna(0) < 0.02).mean() * 100), 1)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
# 2. held-out-cluster effect of making the columns constant (generator never sees that cluster's attacks)
rng = np.random.default_rng(0); is_tr = pd.Series(rng.random(len(df)) < 0.7, index=df.index)
def const(s, cols):
    s = s.copy()
    for c in cols: s[c] = s[c].mode().iloc[0] if c in cats else s[c].median()
    return s
res = []
for k in range(K):
    held = cl.index[cl == k]; tr = df[is_tr & ~df.index.isin(held)]; te = pd.concat([df[~is_tr & ~df.index.isin(held)], df.loc[held]])
    syn = h.g_dvs(tr, 35000, 0, cats, nums); b = h.score(syn, te, cats, nums)[2]; row = {"held_cluster": k, "n_attacks": len(held), "S_dvs": b, "all4": h.score(const(syn, C4), te, cats, nums)[2] - b}
    for c in C4: row[c] = h.score(const(syn, [c]), te, cats, nums)[2] - b
    res.append(row); print(row, flush=True)
R = pd.DataFrame(res); R.to_csv("../out/risk_check_unsw.csv", index=False)
print("\nΔS from making column(s) constant, per held-out attack cluster (negative = risk):"); print(R.round(4).to_string(index=False))
print("\nmean:", R[["all4"] + C4].mean().round(4).to_dict(), "| worst:", R[["all4"] + C4].min().round(4).to_dict())
