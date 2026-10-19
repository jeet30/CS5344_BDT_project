"""Expt 14: RFOD-inspired probe. For every numeric column of the NORMAL rows, how well do the OTHER columns predict it? (random-forest out-of-bag R^2 on the rank-transformed column)
Question: do the columns that benefit from uniform fill share a predictability profile (e.g. unpredictable = noise-like), across both datasets?"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
R = "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/"; name = sys.argv[1]
df = pd.read_csv(R + f"train_{name}.csv"); nm = df[df.is_anomaly == 0].drop(columns="is_anomaly").sample(8000, random_state=0).reset_index(drop=True)
cats = [c for c in nm.columns if not pd.api.types.is_numeric_dtype(nm[c])]; nums = [c for c in nm.columns if c not in cats and nm[c].nunique() > 1]
D = pd.get_dummies(nm[cats]).astype(float) if cats else pd.DataFrame(index=nm.index); rows = []
for c in nums:
    X = pd.concat([nm[[k for k in nums if k != c]], D], axis=1).to_numpy(float); y = (rankdata(nm[c]) - 0.5) / len(nm)
    rf = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, max_features=0.5, oob_score=True, n_jobs=-1, random_state=0).fit(X, y)
    rows.append((c, float(rf.oob_score_), float((nm[c] == 0).mean()), int(nm[c].nunique())))
T = pd.DataFrame(rows, columns=["col", "oob_R2", "zero_share", "n_unique"]).sort_values("oob_R2"); T["pct_rank"] = T.oob_R2.rank(pct=True)
T.to_csv(R + f"expt-14/out/predictability_{name}.csv", index=False); print(T.round(3).to_string(index=False))
