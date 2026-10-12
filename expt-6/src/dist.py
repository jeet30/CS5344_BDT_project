import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import harness as h
pd.set_option("display.width", 200)
def tab(df, key, name, top=8):
    g = df.groupby(key).is_anomaly.agg(n="size", attacks="sum"); g["normals"] = g.n - g.attacks
    g["attack_rate"] = g.attacks / g.n; g["share_of_attacks"] = g.attacks / g.attacks.sum(); g["share_of_normals"] = g.normals / g.normals.sum()
    g["lift"] = g.share_of_attacks / g.share_of_normals.replace(0, np.nan)
    print(f"\n[{name}]"); print(g.sort_values("n", ascending=False).head(top).round(3).to_string())
    return g
u, uc, un = h.load("unsw")
print("UNSW rows", len(u), "attack rate %.3f" % u.is_anomaly.mean(), "| normals", (u.is_anomaly == 0).sum(), "attacks", u.is_anomaly.sum())
u["segment"] = np.select([u.sttl >= 200, (u.sttl >= 60) & (u.sttl < 100), (u.sttl > 0) & (u.sttl < 60), u.sttl == 0], ["WAN (sttl 254/255)", "Mid (sttl 60-64)", "Campus LAN (sttl 31/32)", "sttl=0"], "other")
tab(u, "segment", "UNSW by network segment (source TTL)")
u["dseg"] = np.select([u.dttl >= 200, (u.dttl >= 60) & (u.dttl < 100), (u.dttl > 0) & (u.dttl < 60), u.dttl == 0], ["WAN (dttl 252)", "Mid (dttl 60-64)", "LAN (dttl 29)", "dttl=0"], "other")
tab(u, ["segment", "dseg"], "UNSW sttl x dttl", 10)
for c in ["proto", "state", "service"]: tab(u, c, f"UNSW by {c}")
tab(u, ["segment", "service"], "UNSW segment x service", 14)
tab(u, ["segment", "state"], "UNSW segment x state", 12)
# attack clusters x segment
F = h.make_folds(u, uc, un); ab = u[u.is_anomaly == 1]
X = h.StandardScaler().fit_transform(np.log1p(np.abs(ab[un].to_numpy(float)))); cl = pd.Series(h.KMeans(4, n_init=10, random_state=0).fit_predict(X), index=ab.index)
print("\n[UNSW attack clusters (k-means) x segment: % of cluster]"); print((pd.crosstab(cl.rename("cluster"), ab.segment, normalize="index") * 100).round(1).assign(size=cl.value_counts().sort_index()).to_string())
print("\n[cluster profile: top service / state]"); print(pd.DataFrame({"service": ab.groupby(cl).service.agg(lambda s: s.value_counts(normalize=True).head(3).round(2).to_dict()), "state": ab.groupby(cl).state.agg(lambda s: s.value_counts(normalize=True).head(2).round(2).to_dict())}).to_string())
n, nc, nn = h.load("nslkdd")
print("\n\nNSL-KDD rows", len(n), "attack rate %.3f" % n.is_anomaly.mean(), "| normals", (n.is_anomaly == 0).sum(), "attacks", n.is_anomaly.sum())
for c in ["protocol_type", "flag"]: tab(n, c, f"NSL-KDD by {c}")
tab(n, "service", "NSL-KDD by service", 10)
n["loggedin"] = n.logged_in; n["srvcnt"] = pd.cut(n["count"], [-1, 1, 10, 100, 1000], labels=["1", "2-10", "11-100", ">100"]); n["serr"] = pd.cut(n.serror_rate, [-.01, 0, .5, 1], labels=["0", "(0,.5]", ">.5"])
tab(n, "loggedin", "NSL-KDD by logged_in"); tab(n, "srvcnt", "NSL-KDD by connection count (same-host window)"); tab(n, "serr", "NSL-KDD by SYN-error rate")
ab = n[n.is_anomaly == 1]; X = h.StandardScaler().fit_transform(np.log1p(np.abs(ab[nn].to_numpy(float)))); cl = pd.Series(h.KMeans(4, n_init=10, random_state=0).fit_predict(X), index=ab.index)
print("\n[NSL attack clusters x protocol / flag]"); print(pd.concat([(pd.crosstab(cl, ab.protocol_type, normalize="index") * 100).round(1), (pd.crosstab(cl, ab.flag == "SF", normalize="index") * 100).round(1).rename(columns={True: "flag=SF", False: "flag!=SF"}), cl.value_counts().sort_index().rename("size")], axis=1).to_string())
print("\n[NSL cluster top services]"); print(ab.groupby(cl).service.agg(lambda s: s.value_counts(normalize=True).head(3).round(2).to_dict()).to_string())
