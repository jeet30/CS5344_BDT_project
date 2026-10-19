"""Expt 10 (NSL-KDD): decompose the starter kit (0.93155 on the public LB) into its two ingredients.
  proto20   : 20 prototype normals resampled, NO jitter          -> tests categorical/row concentration alone
  jfull_sXX : ALL real normals resampled, starter-style jitter    -> tests jitter alone (XX = sigma*100)
  starter   : exact kit logic (20 prototypes + 0.5 sigma jitter)  -> local replica, used only to verify the logic vs the scored kit
Jitter rule = starter kit: skip binary and constant columns, add N(0, s*std), clip to the normals' [min, max], round integer columns."""
import numpy as np, pandas as pd
R = "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/"; N = 40000; SEED = 42
df = pd.read_csv(R + "train_nslkdd.csv"); nm = df[df.is_anomaly == 0].drop(columns="is_anomaly").reset_index(drop=True)
cats = [c for c in nm.columns if not pd.api.types.is_numeric_dtype(nm[c])]; nums = [c for c in nm.columns if c not in cats]

def jitter(s, sigma, seed=SEED):
    s = s.copy(); rng = np.random.default_rng(seed)
    for c in nums:
        v = nm[c].to_numpy(float)
        if set(np.unique(v)) <= {0, 1} or v.std() == 0: continue
        x = np.clip(s[c].to_numpy(float) + rng.normal(0, sigma * v.std(), len(s)), v.min(), v.max())
        s[c] = np.rint(x) if np.allclose(v, np.rint(v)) else x
    return s

def save(s, name):
    s = s.reset_index(drop=True).copy(); s.insert(0, "id", np.arange(len(s))); assert len(s) == N and s.isna().sum().sum() == 0
    s.to_csv(R + f"expt-10/submissions/{name}.csv", index=False); print(name, s.shape, "| distinct rows %d | services %d | flags %d" % (s.drop(columns="id").drop_duplicates().shape[0], s.service.nunique(), s.flag.nunique()))

protos = nm.sample(20, random_state=SEED)
proto_pool = protos.sample(N, replace=True, random_state=SEED).reset_index(drop=True)
full_pool = nm.sample(N, replace=True, random_state=SEED).reset_index(drop=True)
save(proto_pool, "nsl_proto20")
save(jitter(full_pool, 0.5), "nsl_jfull_s50")
save(jitter(full_pool, 0.25), "nsl_jfull_s25")
