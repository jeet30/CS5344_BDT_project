"""Regenerate both submissions with the common pipeline and check them against the scored files they must reproduce."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline
from config import ROOT
HERE = Path(__file__).resolve().parent

REFERENCE = {   # dataset -> (scored file, public score)
    "unsw": (HERE / "scored/unsw_uniform__0.57698.csv", 0.57698),
    "nslkdd": (HERE / "scored/nsl_hybrid_unif4__0.93331.csv", 0.93331),
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
