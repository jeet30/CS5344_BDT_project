"""Regenerate both submissions with the common pipeline and check them against the scored files they must reproduce."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline
from config import ROOT
HERE = Path(__file__).resolve().parent

REFERENCE = {   # config -> (scored file, public score)
    "unsw": (HERE / "scored/unsw_product3__0.61452.csv", 0.61452),                                    # the file that was submitted
    "nslkdd": (HERE / "scored/nsl_silk_unif6__0.93734_regenerated.csv", 0.93734),                     # regenerated: identical (1e-9) to the output of the teammate's own runner, whose submitted CSV is not in the repo
    "unsw_prev_0.60905": (HERE / "scored/unsw_sel4_dupjit__0.60905.csv", 0.60905),                    # previous defaults must still reproduce (regression check)
    "nslkdd_prev_0.93674": (HERE / "scored/nsl_hybrid_unif6__0.93674.csv", 0.93674),
}

if __name__ == "__main__":
    ok = True
    for name, (ref, score) in REFERENCE.items():
        new = pipeline.run(name, verbose=False)
        if not ref.exists(): print(f"[{name}] reference file missing: {ref}"); ok = False; continue
        old = pd.read_csv(ref); a = pd.read_csv(Path(pipeline.__file__).resolve().parent / "out" / f"final_{name}.csv")
        try: pd.testing.assert_frame_equal(a, old, check_exact=False, rtol=1e-9, atol=1e-12); same = True      # tolerance: copula linear algebra differs ~1e-23 between runs
        except AssertionError as e: same = False; print("   ", str(e).splitlines()[0])
        print(f"[{name}] reproduces {ref.name} (public score {score}): {'OK' if same else 'MISMATCH'}"); ok &= same
    sys.exit(0 if ok else 1)
