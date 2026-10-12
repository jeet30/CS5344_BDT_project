import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import harness as h
from pyod.models.ecod import ECOD
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score as ap
df,cats,nums=h.load('unsw'); F=h.make_folds(df,cats,nums)
tr_i,te_i,_=F['std']; tr,te=df.loc[tr_i],df.loc[te_i]; y=te.is_anomaly.to_numpy()
syn={"bootstrap":h.g_bootstrap(tr,35000,0,cats,nums),"dvs":h.g_dvs(tr,35000,0,cats,nums)}
def scores(s,cols_n,cols_c):
    Xa,Xb=h.feats(s,te.drop(columns='is_anomaly'),cols_c,cols_n)
    e=ECOD().fit(Xa).decision_function(Xb); f=-IsolationForest(n_estimators=150,random_state=0,n_jobs=-1).fit(Xa).score_samples(Xb)
    return e,f
res={}
for k,s in syn.items():
    e,f=scores(s,nums,cats); res[k]=(e,f); print(k,"ECOD",round(ap(y,e),4),"IF",round(ap(y,f),4))
# 1. error anatomy for dvs: attacks by segment, and normals by segment, mean rank-percentile (higher = flagged more)
te=te.assign(**{f"{k}_{d}":pd.Series(v,index=te.index).rank(pct=True) for k,(e,f) in res.items() for d,v in (("e",e),("f",f))})
te["seg"]=np.where(te.sttl>=200,"WAN(sttl>=200)",np.where(te.sttl==0,"sttl0","LAN"))
print("\nmean flagged-percentile by segment (1.0=most anomalous)")
print(te.groupby(["is_anomaly","seg"])[["bootstrap_e","dvs_e","bootstrap_f","dvs_f"]].agg("mean").round(3).join(te.groupby(["is_anomaly","seg"]).size().rename("n")))
# 2. leave-one-feature-out on dvs training data (drop col from train+test)
print("\nleave-one-feature-out delta in S (dvs-trained); positive = removing the feature HELPS")
base=(ap(y,res["dvs"][0])+ap(y,res["dvs"][1]))/2; out=[]
for c in nums+cats:
    n2=[x for x in nums if x!=c]; c2=[x for x in cats if x!=c]
    e,f=scores(syn["dvs"].drop(columns=c),n2,c2); out.append((c,(ap(y,e)+ap(y,f))/2-base,ap(y,e)-ap(y,res["dvs"][0]),ap(y,f)-ap(y,res["dvs"][1])))
o=pd.DataFrame(out,columns=["feature","dS","dECOD","dIF"]).sort_values("dS"); print(o.round(4).to_string(index=False))
o.to_csv("../out/loo_dvs_unsw.csv",index=False)
# 3. single-feature ECOD AUPRC: bootstrap vs dvs training
sf=[]
for c in nums:
    r=[]
    for k in syn:
        e=ECOD().fit(syn[k][[c]].to_numpy(float)).decision_function(te[[c]].to_numpy(float)); r.append(ap(y,e))
    sf.append((c,*r))
print("\nsingle-feature ECOD AUPRC (chance=%.3f)"%y.mean()); print(pd.DataFrame(sf,columns=["feature","bootstrap","dvs"]).sort_values("dvs",ascending=False).round(3).head(20).to_string(index=False))
