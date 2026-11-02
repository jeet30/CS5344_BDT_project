"""Expt 19 phase 1: candidate files that combine pieces we already have (stage 1 from expt-17/18, stages 2-6 from final_pipeline)."""
import sys, copy, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "expt-18" / "src"))
import crosscheck as cc
base, CONFIG, pd = cc.base, cc.CONFIG, cc.pd
SUB = HERE.parent / "submissions"; OUT = HERE.parent / "out"

def pool(key, make):
    f = OUT / f"pool_{key}.pkl"
    if f.exists(): return pickle.load(open(f, "rb"))
    s = make(); pickle.dump(s, open(f, "wb")); return s

def variant(name, cfg, s, df, cats, nums):
    cc.finish(name, s.copy(), df, cats, nums, cfg, SUB / f"{name}.csv")

def unsw(which):
    cfg = copy.deepcopy(CONFIG["unsw"]); df, cats, nums = base.load(cfg)
    if which == "blend_sel5":
        cfg["neutralize"]["extra_columns"] = ["ct_dst_ltm", "trans_depth"]
        variant("unsw_blend_sel5", cfg, pool("unsw_blend", lambda: cc.gen_unsw(df, cfg, {"proto": .5, "state": .5})), df, cats, nums)
    if which == "blend_smean":                      # round-3 column picked by the harness on the blend base
        cfg["neutralize"]["extra_columns"] = ["ct_dst_ltm", "smean"]
        variant("unsw_blend_smean", cfg, pool("unsw_blend", lambda: cc.gen_unsw(df, cfg, {"proto": .5, "state": .5})), df, cats, nums)
    if which == "blend_theta":                      # theta re-fitted on the blend base (harness ES, all 14 folds)
        cfg["weights"]["theta"] = OUT / "weights_theta_unsw_blend.json"
        variant("unsw_blend_theta", cfg, pool("unsw_blend", lambda: cc.gen_unsw(df, cfg, {"proto": .5, "state": .5})), df, cats, nums)
    if which in ("product", "product_shrink30"):    # ONE DVS model conditioned on the joint proto|state instead of a 50/50 mix of two models
        if which.endswith("30"): cfg["dvs"] = {**cfg["dvs"], "shrinkage_m": 30.0}
        def make():
            d2 = df.assign(proto_state=df.proto + "|" + df.state)
            return cc.gen_unsw(d2, cfg, {"proto_state": 1.0}).drop(columns="proto_state")
        variant("unsw_" + which, cfg, pool("unsw_" + which, make), df, cats, nums)
    if which == "product3":                         # joint proto|service|state (the same cells the category weights use)
        def make():
            d2 = df.assign(pss=df.proto + "|" + df.service + "|" + df.state)
            return cc.gen_unsw(d2, cfg, {"pss": 1.0}).drop(columns="pss")
        variant("unsw_product3", cfg, pool("unsw_product3", make), df, cats, nums)
    if which == "state_silk":
        variant("unsw_state50_silk50", cfg, pool("unsw_state_silk", lambda: cc.gen_unsw(df, cfg, {"state": .5, "silk": .5})), df, cats, nums)

def nsl(which):
    cfg = copy.deepcopy(CONFIG["nslkdd"]); df, cats, nums = base.load(cfg)
    if which == "silk_slack12":
        c = copy.deepcopy(cfg); c["hybrid"]["tail_slack_gamma"] = 1.2
        variant("nsl_silk_slack12", c, pool("nsl_silk_slack12", lambda: cc.gen_nsl(c, 0.0)), df, cats, nums)
    if which.startswith("silk_stretch"):
        f = float(which.split("stretch")[1]); c = copy.deepcopy(cfg); c["stretch"]["factor"] = f
        variant(f"nsl_silk_stretch{int(f)}", c, pool("nsl_silk", lambda: cc.gen_nsl(cfg, 0.0)), df, cats, nums)

if __name__ == "__main__":
    for w in sys.argv[1:]:
        (unsw if w in ("blend_sel5", "blend_smean", "blend_theta", "product", "product_shrink30", "product3", "state_silk") else nsl)(w)
