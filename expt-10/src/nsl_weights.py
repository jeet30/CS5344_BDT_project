"""Expt 10 (NSL-KDD): category-cell reweighting of REAL normals (the lever that worked on UNSW), tested in BOTH directions.
w_row = exp(kappa * clip(logLR_service + logLR_flag, -3, 3)), logLR = log(p_attack(cat)/p_normal(cat)) with additive smoothing.
kappa > 0 : up-weight categories attacks over-use (private, S0/REJ/RSTR ...)   kappa < 0 : down-weight them."""
import numpy as np, pandas as pd
R = "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/"; N = 40000; SEED = 42
df = pd.read_csv(R + "train_nslkdd.csv"); nm = df[df.is_anomaly == 0].drop(columns="is_anomaly").reset_index(drop=True); ab = df[df.is_anomaly == 1]
def loglr(col, a=5.0):
    pn = (nm[col].value_counts() + a) / (len(nm) + a * nm[col].nunique()); pa = (ab[col].value_counts().reindex(pn.index).fillna(0) + a) / (len(ab) + a * len(pn))
    return np.log(pa / pn)
lr = nm["service"].map(loglr("service")) + nm["flag"].map(loglr("flag")); lr = np.clip(lr.to_numpy(), -3, 3)
def make(kappa, name):
    w = np.exp(kappa * lr); w /= w.sum(); s = nm.sample(N, replace=True, weights=w, random_state=SEED).reset_index(drop=True)
    o = s.copy(); o.insert(0, "id", np.arange(N)); assert o.isna().sum().sum() == 0; o.to_csv(R + f"expt-10/submissions/{name}.csv", index=False)
    print(f"{name:14s} kappa {kappa:+.2f} | private {(s.service=='private').mean():.3f} (real {(nm.service=='private').mean():.3f}) | flag SF {(s.flag=='SF').mean():.3f} (real {(nm.flag=='SF').mean():.3f}) REJ {(s.flag=='REJ').mean():.3f} (real {(nm.flag=='REJ').mean():.3f}) | http {(s.service=='http').mean():.3f} (real {(nm.service=='http').mean():.3f}) | duplicates {s.duplicated().mean():.3f}")
print("real normals: n =", len(nm)); make(+0.5, "nsl_wup_k05"); make(-0.5, "nsl_wdown_k05"); make(0.0, "nsl_w0_resample")
