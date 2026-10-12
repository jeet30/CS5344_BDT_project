"""Item 7: shift diagnostics of submitted files vs real normals (per-feature KS/Wasserstein, energy-distance, classifier two-sample AUC)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/Users/aayushmajumdar/Desktop/CS5344_BDTechrel/expt-6/src")
import harness as h
def diagnose(name, files):
    df, cats, nums = h.load(name); real = df[df.is_anomaly == 0].drop(columns="is_anomaly"); rows = []; per = {}
    for tag, f in files.items():
        s = pd.read_csv(h.ROOT + f).drop(columns="id"); rec = {"file": tag}
        ks = {c: ks_2samp(real[c], s[c]).statistic for c in nums if s[c].nunique() > 1 or real[c].nunique() > 1}
        sd = real[nums].std().replace(0, 1)
        wd = {c: wasserstein_distance(real[c], s[c]) / sd[c] for c in nums}
        rec.update(mean_KS=np.mean(list(ks.values())), max_KS=max(ks.values()), mean_W1_std=np.mean(list(wd.values())),
                   const_cols=int(sum(s[c].nunique() == 1 for c in s.columns)))
        for c in cats: rec[f"TVdist_{c}"] = 0.5 * float((real[c].value_counts(normalize=True).sub(s[c].value_counts(normalize=True), fill_value=0)).abs().sum())
        a = real.sample(min(8000, len(real)), random_state=0); b = s.sample(min(8000, len(s)), random_state=0); X = pd.concat([pd.concat([a, b]).drop(columns=cats), pd.get_dummies(pd.concat([a, b])[cats]).astype(float)], axis=1)
        y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        rec["classifier_AUC"] = roc_auc_score(y, cross_val_predict(H(max_iter=60, random_state=0), X.to_numpy(float), y, cv=3, method="predict_proba")[:, 1])
        rows.append(rec); per[tag] = pd.Series(ks)
    D = pd.DataFrame(rows).set_index("file"); K = pd.DataFrame(per)
    D.round(4).to_csv(h.ROOT + f"expt-8/out/diag_{name}.csv"); K.round(4).to_csv(h.ROOT + f"expt-8/out/diag_ks_{name}.csv")
    print(f"\n[{name}] summary\n", D.round(3).to_string()); print("\nlargest per-feature KS (last file listed vs real):\n", K.iloc[:, -1].sort_values(ascending=False).head(8).round(3).to_string())
if __name__ == "__main__":
    diagnose("unsw", {"const4 (DVS+4 const, 0.54529)": "expt-6/submissions/submission_unsw_const4.csv", "pipeline7 (0.54803)": "expt-7/submissions/pipeline7_unsw.csv", "pipeline7_eb20 (0.54836)": "expt-7/submissions/pipeline7_eb20_unsw.csv", "expt4_DVS (0.53263)": "expt-4/submissions/submission_unsw.csv"})
    diagnose("nslkdd", {"pipeline7=bootstrap": "expt-7/submissions/pipeline7_nslkdd.csv", "pipeline7_eb20": "expt-7/submissions/pipeline7_eb20_nslkdd.csv", "pipeline_failed(11 const)": "expt-6/submissions/pipeline_nslkdd.csv"})
