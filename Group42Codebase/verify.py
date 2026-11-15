"""Regenerate both submissions and check them against the reference files in reference/.

    python verify.py

Numeric cells are compared with a 1e-9 relative tolerance: the copula linear algebra differs by ~1e-23 between runs.
"""
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline

REFERENCE = {   # config -> (reference file, public leaderboard score)
    "unsw": (HERE / "reference/unsw_selected__0.61452.csv", 0.61452),
    "nslkdd": (HERE / "reference/nslkdd_selected__0.93734.csv", 0.93734),
}

if __name__ == "__main__":
    ok = True
    for name, (ref, score) in REFERENCE.items():
        new = pipeline.run(name, verbose=True)
        if not ref.exists():
            print(f"[{name}] reference file missing: {ref}"); ok = False; continue
        old = pd.read_csv(ref); a = pd.read_csv(HERE / "out" / f"final_{name}.csv")
        try:
            pd.testing.assert_frame_equal(a, old, check_exact=False, rtol=1e-9, atol=1e-12); same = True
        except AssertionError as e:
            same = False; print("   ", str(e).splitlines()[0])
        print(f"[{name}] reproduces {ref.name} (public score {score}): {'OK' if same else 'MISMATCH'}"); ok &= same
    sys.exit(0 if ok else 1)
