"""Generate standalone hybrid itemset-clustered copula submissions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from constraints import audit_constraints, enforce_nslkdd_constraints, enforce_unsw_constraints
from data_loader import SUBMISSION_SPECS
from generators.itemset_copula_hybrid import ItemsetCopulaHybridSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["nslkdd", "unsw"])
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--max-support", type=float, default=0.35)
    parser.add_argument("--min-copula-rows", type=int, default=500)
    parser.add_argument("--tail-slack-gamma", type=float, default=1.05)
    parser.add_argument("--n-copula-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve() if args.root_dir else Path(__file__).resolve().parents[2]
    scrubbed_path = root_dir / "expt-2" / "eda" / "outputs" / f"{args.dataset}_scrubbed_normal.csv"
    if not scrubbed_path.exists():
        raise FileNotFoundError(f"EDA output not found: {scrubbed_path}. Run run_eda_analysis.py first.")
    normal = pd.read_csv(scrubbed_path)

    sampler = ItemsetCopulaHybridSampler(
        min_support=args.min_support,
        max_support=args.max_support,
        min_copula_rows=args.min_copula_rows,
        tail_slack_gamma=args.tail_slack_gamma,
        n_copula_seeds=args.n_copula_seeds,
        random_state=args.seed,
    )
    submission = sampler.fit(normal).sample(SUBMISSION_SPECS[args.dataset]["target_n"])
    submission = enforce_nslkdd_constraints(submission) if args.dataset == "nslkdd" else enforce_unsw_constraints(submission)
    submission.insert(0, "id", np.arange(len(submission), dtype=int))

    expected = ["id"] + list(normal.columns)
    if submission.columns.tolist() != expected:
        raise ValueError("Submission columns do not match the scrubbed normal data.")
    if submission["id"].duplicated().any() or submission.isna().any().any():
        raise ValueError("Submission contains duplicate IDs or missing values.")
    numeric = submission.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Submission contains NaN or infinite numeric values.")

    output = Path(args.output) if args.output else root_dir / "expt-2" / "submissions" / f"itemset_copula_hybrid_{args.dataset}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Diagnostics: {sampler.diagnostics_}")
    print(f"Constraint violations: {audit_constraints(submission, args.dataset)}")
    print(f"Wrote {len(submission):,} rows to {output}")


if __name__ == "__main__":
    main()
