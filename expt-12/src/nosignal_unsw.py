"""Does the NSL-style 'no single-feature signal' constant rule help on UNSW (weighted + stretched base)?  Harness is reliable on UNSW."""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from reselect import *
NOSIG = ['dloss', 'spkts', 'dpkts', 'dload', 'dbytes', 'dwin', 'stcpb', 'swin', 'dtcpb', 'sloss']
W = Weighter("unsw", base="gatev_dvs_w60"); th = np.array(json.load(open(R8 + "weights_theta_unsw.json"))["theta"]) * SCALE; rows = []
for fn in W.F:
    w = np.exp(np.r_[th, 0.0][W.cellid[fn]]); w /= w.sum(); s = W.S0[fn].iloc[np.random.default_rng(0).choice(len(w), size=len(w), p=w)].reset_index(drop=True)
    sc = lambda cols: h.score(const(s, cols, W.cats), W.te[fn], W.cats, W.nums, seeds=2)[2]
    b = sc([]); rows.append((fn, W.fam[fn], sc(OLD) - b, sc(NOSIG) - b, sc(OLD + NOSIG) - b)); print(fn, np.round(rows[-1][2:], 4), flush=True)
R = pd.DataFrame(rows, columns=["fold", "family", "old_pair", "no_signal10", "both"]); print("\nmean change in S vs no constants:", R[["old_pair", "no_signal10", "both"]].mean().round(4).to_dict(), "| worst:", R[["old_pair", "no_signal10", "both"]].min().round(4).to_dict())
print(R.groupby("family")[["old_pair", "no_signal10", "both"]].mean().round(4).to_string()); R.to_csv(R12 + "nosignal_unsw.csv", index=False)
