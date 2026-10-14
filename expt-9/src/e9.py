"""Expt 9: (a) looser cross-cluster gating thresholds => stretch more columns; (b) no-duplicate weighted sampling."""
import sys
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-8/src")
from e8 import *
R9 = h.ROOT + "expt-9/out/"
CONF = {"a58_m75": (0.58, 0.75), "a55_m75": (0.55, 0.75), "a58_m50": (0.58, 0.50), "a55_m50": (0.55, 0.50), "a52_m50": (0.52, 0.50)}
for k, (a, m) in CONF.items():
    for w in (3.0, 6.0): h.GENS[f"gv_{k}_w{int(w)}"] = g_gate("dvs", 1.0, w, min_auc=a, min_agree=m, valid=True)

if __name__ == "__main__":
    name = sys.argv[1]
    if sys.argv[2] == "count":      # how many columns each setting would stretch on the full data
        df, cats, nums = h.load(name)
        for k, (a, m) in CONF.items(): d = feature_direction(df, nums, min_auc=a, min_agree=m, seed=42); print(f"{k}: {len(d)} cols  up={[c for c,v in d.items() if v>0]}  down={[c for c,v in d.items() if v<0]}")
    else:
        robust_table(name, sys.argv[2].split(","), "gv")
