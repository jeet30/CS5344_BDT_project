"""Figures for the final report. Run from the repo root: .venv/bin/python report/figs.py"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from pyod.models.ecod import ECOD

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "report" / "figs"; OUT.mkdir(exist_ok=True, parents=True)
NRM, ATK, OURS, GREY, NAVY = "#3B6FB6", "#E4572E", "#12968A", "#8A94A0", "#0B132B"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                     "legend.fontsize": 6.8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
                     "pdf.fonttype": 42, "axes.titleweight": "bold"})
u = pd.read_csv(ROOT / "train_unsw.csv"); n = pd.read_csv(ROOT / "train_nslkdd.csv")
LAB = "is_anomaly"


def single_ecod(df):
    """AUPRC of ECOD fitted on ONE column of the real normals, scoring every real row (attack = positive)."""
    y = df[LAB].to_numpy(); rows = []
    for c in df.columns.drop(LAB):
        if not pd.api.types.is_numeric_dtype(df[c]): continue
        x = df[[c]].to_numpy(float)
        if np.ptp(x) == 0: continue
        s = ECOD().fit(x[y == 0]).decision_function(x)
        rows.append((c, average_precision_score(y, s)))
    return pd.DataFrame(rows, columns=["col", "ap"]).sort_values("ap")


su, sn = single_ecod(u), single_ecod(n)
su.to_csv(OUT / "single_ecod_unsw.csv", index=False); sn.to_csv(OUT / "single_ecod_nsl.csv", index=False)
chance_u, chance_n = u[LAB].mean(), n[LAB].mean()


def util_panel(ax, s, chance, title, k=12):
    t = pd.concat([s.head(k // 2), s.tail(k // 2)])
    col = [OURS if a > chance + 0.02 else (ATK if a < chance - 0.0 else GREY) for a in t.ap]
    ax.barh(t.col, t.ap, color=col, height=0.7); ax.axvline(chance, color=NAVY, lw=0.8, ls="--")
    ax.set_title(title); ax.set_xlabel("1-feature ECOD AUPRC")


# ---- Fig 1: UNSW EDA ----
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.15, 1, 1]})
cell = u[["proto", "service", "state"]].astype(str).agg("|".join, axis=1)
t = pd.crosstab(cell, u[LAB]); t["n"] = t.sum(axis=1); t["share"] = t[1] / t["n"]; t = t.sort_values("n", ascending=False).head(9).iloc[::-1]
ax[0].barh(t.index, t.share, color=[ATK if s > .3 else NRM for s in t.share], height=0.7)
for i, (s, k) in enumerate(zip(t.share, t.n)): ax[0].text(s + 0.01, i, f"n={k:,}", va="center", fontsize=6)
ax[0].set_xlim(0, 1.05); ax[0].set_xlabel("attack share of the cell"); ax[0].set_title("(a) Attack share per cell", loc="left")
bins = np.arange(-0.5, 256.5, 8)
ax[1].hist(u.loc[u[LAB] == 0, "sttl"], bins=bins, color=NRM, alpha=.9, label="normal"); ax[1].hist(u.loc[u[LAB] == 1, "sttl"], bins=bins, color=ATK, alpha=.9, label="attack")
ax[1].set_yscale("log"); ax[1].set_xlabel("sttl (no attack at 31/32)"); ax[1].set_ylabel("rows (log)"); ax[1].legend(frameon=False); ax[1].set_title("(b) Source TTL")
util_panel(ax[2], su, chance_u, "(c) Feature signal")
fig.tight_layout(); fig.savefig(OUT / "eda_unsw.pdf"); plt.close(fig)

# ---- Fig 2: NSL EDA ----
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.1, 1, 1]})
items = []
for col, vals in (("logged_in", [0, 1]), ("protocol_type", ["tcp", "udp", "icmp"]), ("flag", ["SF", "S0", "REJ"])):
    for v in vals:
        m = n[col] == v; items.append((f"{col}={v}", n.loc[m, LAB].mean(), int(m.sum())))
d = pd.DataFrame(items, columns=["g", "share", "n"]).iloc[::-1]
ax[0].barh(d.g, d.share, color=[ATK if s > .3 else NRM for s in d.share], height=0.7)
for i, (s, k) in enumerate(zip(d.share, d.n)): ax[0].text(min(s + 0.01, 0.8), i, f"n={k:,}", va="center", fontsize=6)
ax[0].set_xlim(0, 1.2); ax[0].set_xlabel("attack share"); ax[0].set_title("(a) Attack share per group", loc="left")
ax[1].hist([n.loc[n[LAB] == 0, "srv_count"], n.loc[n[LAB] == 1, "srv_count"]], bins=np.linspace(0, 511, 40), color=[NRM, ATK], label=["normal", "attack"], density=True)
ax[1].set_xlabel("srv_count"); ax[1].set_ylabel("density"); ax[1].legend(frameon=False); ax[1].set_title("(b) srv_count: no separation")
util_panel(ax[2], sn, chance_n, "(c) Feature signal")
fig.tight_layout(); fig.savefig(OUT / "eda_nsl.pdf"); plt.close(fig)

# ---- Fig 3: what shaping does to marginals (real normals vs attacks vs our final file) ----
fu = pd.read_csv(ROOT / "final_pipeline" / "scored" / "unsw_product3__0.61452.csv")
fn = pd.read_csv(ROOT / "final_pipeline" / "scored" / "nsl_silk_unif6__0.93734_regenerated.csv")
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9))
def hist3(a, df, syn, col, title, log=False, xlab=None):
    r, k, s = df.loc[df[LAB] == 0, col].to_numpy(float), df.loc[df[LAB] == 1, col].to_numpy(float), syn[col].to_numpy(float)
    if log: r, k, s = (np.log10(v[v > 0]) for v in (r, k, s))
    lo, hi = np.percentile(np.r_[r, k, s], [0.5, 99.0]); b = np.linspace(lo, hi, 36)
    for v, c, l in ((r, NRM, "real normal"), (k, ATK, "real attack"), (s, OURS, "our final file")):
        a.hist(v, bins=b, histtype="step", density=True, color=c, lw=1.2, label=l)
    a.set_title(title); a.set_xlabel(xlab or col); a.set_yticks([])
hist3(ax[0], u, fu, "dload", "UNSW dload (stretch)", log=True, xlab="log10 dload (>0)")
def ecdf3(a, df, syn, col, title, log=False):
    for v, c, l in ((df.loc[df[LAB] == 0, col], NRM, "real normal"), (df.loc[df[LAB] == 1, col], ATK, "real attack"), (syn[col], OURS, "our final file")):
        v = np.sort(v.to_numpy(float)); 
        if log: v = v[v > 0]
        a.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=1.3, label=l)
    if log: a.set_xscale("log")
    a.set_title(title); a.set_xlabel(col + (" (log scale)" if log else "")); a.set_ylabel("ECDF" if col == "sinpkt" else "")
ecdf3(ax[1], u, fu, "sinpkt", "UNSW sinpkt (fill)", log=True)
ecdf3(ax[2], n, fn, "dst_host_count", "NSL dst_host_count")
ecdf3(ax[3], n, fn, "duration", "NSL duration (fill)", log=True)
h_, l_ = ax[1].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False); fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "shaping.pdf"); plt.close(fig)

# ---- Fig 4: journeys (public leaderboard before the reveal; README evidence table) ----
uj = [("real\nnormals", 0.4210), ("DVS\ncopula", 0.53263), ("+const\nsinpkt,sjit", 0.54803), ("+cell\nweights x3", 0.56940), ("+stretch10x\n+caps", 0.57067),
      ("uniform\nfill", 0.57698), ("+ct_dst_src\n_ltm", 0.59066), ("+jitter,\n+ct_dst_ltm", 0.60905), ("proto|state\nDVS", 0.61392), ("proto|svc|\nstate DVS", 0.61452)]
nj = [("real\nbootstrap", 0.92919), ("hybrid\nalone", 0.93092), ("+4 cols\nuniform", 0.93331), ("+7 cols\n(1 harmful)", 0.93512), ("6 cols\nselected", 0.93674), ("SILK\nk=20", 0.93734)]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.2, 1]})
for a, J, ttl, lo, hi in ((ax[0], uj, "(a) UNSW-NB15", 0.38, 0.68), (ax[1], nj, "(b) NSL-KDD", 0.920, 0.943)):
    J = J[::-1]; v = [x[1] for x in J]; cols = [NAVY] + [OURS] * (len(J) - 2) + [GREY]
    a.barh(range(len(J)), [x - lo for x in v], left=lo, color=cols, height=0.68)
    for i, x in enumerate(v): a.text(x + (hi - lo) * 0.01, i, f"{x:.4f}" if lo > 0.5 else f"{x:.3f}", va="center", fontsize=6)
    a.set_yticks(range(len(J))); a.set_yticklabels([x[0].replace("\n", " ") for x in J], fontsize=6.2); a.set_xlim(lo, hi); a.set_title(ttl, loc="left"); a.set_xlabel("public AUPRC (before the reveal)")
fig.tight_layout(); fig.savefig(OUT / "journey.pdf"); plt.close(fig)
print("ok")

# ---- Fig 5: which attacks the final file catches (expt-19/src/attack_profile.py, in-sample, KMeans k=4) ----
cl = [("C0: tcp http", 22.7, 0.718, 0.939), ("C1: udp INT", 13.6, 0.913, 0.827), ("C2: dns INT", 9.2, 0.991, 0.991), ("C3: tcp FIN", 54.5, 0.496, 0.775)]
fig, ax = plt.subplots(figsize=(3.5, 2.1)); x = np.arange(4); w = 0.38
ax.bar(x - w / 2, [c[2] for c in cl], w, color=GREY, label="trained on real normals"); ax.bar(x + w / 2, [c[3] for c in cl], w, color=OURS, label="trained on our final file")
for i, c in enumerate(cl):
    ax.text(i - w / 2, c[2] + 0.01, f"{c[2]:.2f}", ha="center", fontsize=5.8); ax.text(i + w / 2, c[3] + 0.01, f"{c[3]:.2f}", ha="center", fontsize=5.8)
ax.set_xticks(x); ax.set_xticklabels([f"{c[0]}\n{c[1]:.1f}%" for c in cl], fontsize=5.8); ax.set_ylim(0.3, 1.12); ax.axhline(0.5, color=NAVY, lw=0.6, ls=":")
ax.set_ylabel("AUROC, cluster vs all normals"); ax.legend(frameon=False, loc="upper left", fontsize=5.8)
fig.tight_layout(); fig.savefig(OUT / "clusters.pdf"); plt.close(fig)

# ---- Fig 6: sensitivity around the final recipe ----
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [0.8, 1.2]})
xs = [0, 1, 2, 3, 4]; ys = [0.55048, 0.55468, 0.56254, 0.56940, 0.56106]
ax[0].plot(xs, ys, "-o", color=OURS, ms=4, lw=1.4)
for x_, y_ in zip(xs, ys): ax[0].text(x_, y_ + 0.0012, f"{y_:.4f}", ha="center", fontsize=5.8)
ax[0].set_xticks(xs); ax[0].set_xlabel("category-weight scale s in exp(s*theta)"); ax[0].set_ylabel("UNSW public AUPRC"); ax[0].set_ylim(0.548, 0.5745); ax[0].set_title("(a) Leaderboard: peak at s = 3", loc="left")
alt = [("q05-q95 fill range", -0.0075), ("min-max fill range", -0.0077), ("fill 75% of rows", -0.0078), ("fill 50% of rows", -0.0137), ("log1p-uniform fill", -0.0149),
       ("log-uniform, 75% rows", -0.0198), ("fill from real marginal", -0.0251), ("weight scale 3.5", -0.0033), ("weight scale 2.5", -0.0047)]
alt = sorted(alt, key=lambda t: t[1]); 
ax[1].barh([a[0] for a in alt], [a[1] for a in alt], color=ATK, height=0.65); ax[1].axvline(0, color=NAVY, lw=0.8)
for i, a in enumerate(alt): ax[1].text(a[1] - 0.0004, i, f"{a[1]:+.4f}", ha="right", va="center", fontsize=5.8)
ax[1].set_xlim(-0.032, 0.002); ax[1].set_xlabel("harness change vs final recipe (14 folds)"); ax[1].set_title("(b) Every alternative is worse", loc="left")
fig.tight_layout(); fig.savefig(OUT / "sensitivity.pdf"); plt.close(fig)
