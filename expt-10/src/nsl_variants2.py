"""Starter-kit family around the 0.93155 point: (K prototypes, jitter sigma, prototype seed). Same jitter rule as the kit."""
import numpy as np, pandas as pd
import nsl_variants as v
def starter(K, sigma, proto_seed, name, pool_seed=42):
    protos = v.nm.sample(K, random_state=proto_seed); pool = protos.sample(v.N, replace=True, random_state=pool_seed).reset_index(drop=True)
    s = v.jitter(pool, sigma, seed=pool_seed); v.save(s, name); return protos
for K, sg, ps, nm_ in [(20, 0.5, 7, "nsl_starter_seed7"), (50, 0.5, 42, "nsl_starter_K50"), (20, 0.25, 42, "nsl_starter_s25")]:
    p = starter(K, sg, ps, nm_); print("   prototype services:", p.service.value_counts().to_dict(), "| flags:", p.flag.value_counts().to_dict())
