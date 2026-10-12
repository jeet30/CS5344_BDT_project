import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import harness as h
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from pyod.models.ecod import ECOD
df, cats, nums = h.load('unsw')
wan = df[df.sttl >= 200].copy(); lan = df[(df.sttl < 200) & (df.sttl > 0)].copy()
for nm, d in (("WAN", wan), ("LAN", lan)):
    print(nm, len(d), "attack rate %.3f" % d.is_anomaly.mean())
y = wan.is_anomaly.to_numpy()
X = pd.concat([wan[nums], pd.get_dummies(wan[cats]).astype(float)], axis=1)
p = cross_val_predict(H(max_iter=150, random_state=0), X.to_numpy(float), y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print("WAN supervised ceiling: AUROC %.3f AUPRC %.3f (chance %.3f)" % (auc(y, p), ap(y, p), y.mean()))
# per-feature separability within WAN: single-feature AUROC (direction-free) and which tail attacks sit in relative to WAN normals
rows = []
wn = wan[wan.is_anomaly == 0]
for c in nums:
    a = auc(y, wan[c].to_numpy(float)); rows.append((c, max(a, 1 - a), "attacks higher" if a > .5 else "attacks lower",
        wn[c].nunique(), float((wan.loc[wan.is_anomaly == 1, c] > wn[c].quantile(.99)).mean()), float((wan.loc[wan.is_anomaly == 1, c] < wn[c].quantile(.01)).mean())))
S = pd.DataFrame(rows, columns=["feature", "auroc", "direction", "n_unique_normal", "atk_above_p99_normal", "atk_below_p1_normal"]).sort_values("auroc", ascending=False)
print(S.head(14).round(3).to_string(index=False))
for c in cats:
    t = pd.crosstab(wan[c], wan.is_anomaly, normalize="columns").round(3); t["ratio_atk/norm"] = (t[1] / t[0].replace(0, np.nan)).round(2); print("\n", c); print(t.sort_values(1, ascending=False).head(6))
# What does ECOD-on-WAN-normals-only achieve inside WAN, i.e. how separable are WAN attacks by normal-only modeling?
tr, te = wan.sample(frac=.6, random_state=0), None
trn = tr[tr.is_anomaly == 0]; te = wan.drop(tr.index)
Xa, Xb = h.feats(trn.drop(columns='is_anomaly'), te.drop(columns='is_anomaly'), cats, nums)
print("\nWAN-only: ECOD trained on WAN normals -> AUPRC %.3f (chance %.3f)" % (ap(te.is_anomaly, ECOD().fit(Xa).decision_function(Xb)), te.is_anomaly.mean()))
