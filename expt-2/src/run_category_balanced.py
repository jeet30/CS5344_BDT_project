"""Run the standalone notebook-derived category-balancing strategy.

This entry point is intentionally separate from ``benchmark.py`` and does
not alter the teammate's Experiment 2 generator pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from constraints import enforce_nslkdd_constraints, enforce_unsw_constraints
from data_loader import SUBMISSION_SPECS
from generators.category_balanced import CategoryBalancedNormalSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["nslkdd", "unsw"])
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Repository root. Defaults to the root containing this script.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--lambda", dest="balance_lambda", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root_dir = (
        Path(args.root_dir).resolve()
        if args.root_dir is not None
        else Path(__file__).resolve().parents[2]
    )
    balance_feature = "protocol_type" if args.dataset == "nslkdd" else "state"
    target_n = SUBMISSION_SPECS[args.dataset]["target_n"]
    # Match the teammate's flow: this standalone strategy consumes the
    # already-generated EDA normal foundation directly. The raw train file is
    # only needed when creating that EDA output, not when generating a
    # submission from it.
    scrubbed_path = root_dir / "expt-2" / "eda" / "outputs" / f"{args.dataset}_scrubbed_normal.csv"
    if not scrubbed_path.exists():
        raise FileNotFoundError(
            f"EDA output not found: {scrubbed_path}. Run run_eda_analysis.py first."
        )
    scrubbed_normal = pd.read_csv(scrubbed_path)

    sampler = CategoryBalancedNormalSampler(
        balance_feature=balance_feature,
        balance_lambda=args.balance_lambda,
        random_state=args.seed,
    )
    submission = sampler.fit(scrubbed_normal).sample(target_n)
    submission = (
        enforce_nslkdd_constraints(submission)
        if args.dataset == "nslkdd"
        else enforce_unsw_constraints(submission)
    )
    submission.insert(0, "id", np.arange(len(submission), dtype=int))

    output = Path(args.output) if args.output else root_dir / "expt-2" / "submissions" / f"category_balanced_{args.dataset}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Wrote {len(submission):,} rows to {output}")


if __name__ == "__main__":
    main()
