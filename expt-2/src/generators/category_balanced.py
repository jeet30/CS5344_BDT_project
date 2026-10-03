"""Notebook-compatible category-balanced normal sampler.

This is the shared implementation of the NSL-KDD/UNSW notebook strategy:
keep all normal rows, then sample only the additional rows required to reach
the submission size. The additional-row distribution is a convex combination
of the observed category distribution and a uniform distribution.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class CategoryBalancedNormalSampler:
    """Generate submissions using the notebook's partial-balancing strategy."""

    def __init__(
        self,
        balance_feature: str,
        balance_lambda: float = 0.3,
        random_state: int = 42,
    ) -> None:
        if not 0.0 <= balance_lambda <= 1.0:
            raise ValueError("balance_lambda must be between 0 and 1.")
        self.balance_feature = balance_feature
        self.balance_lambda = balance_lambda
        self.random_state = random_state
        self.normal_: Optional[pd.DataFrame] = None
        self.additional_counts_: Optional[pd.Series] = None
        self.feature_cols_: list[str] = []

    def fit(self, df: pd.DataFrame, label_col: str = "is_anomaly") -> "CategoryBalancedNormalSampler":
        # EDA scrubbed files contain normal rows only and therefore have no
        # label column. Raw training data remains supported as well.
        if label_col not in df.columns:
            normal = df.copy()
        else:
            normal = df.loc[df[label_col] == 0].drop(columns=[label_col]).copy()
        if self.balance_feature not in df.columns:
            raise ValueError(f"Missing balance feature '{self.balance_feature}'.")
        if normal.empty:
            raise ValueError("No normal rows available for sampling.")
        if normal.isna().any().any():
            raise ValueError("Normal rows contain missing values.")
        self.normal_ = normal.reset_index(drop=True)
        self.feature_cols_ = list(normal.columns)
        return self

    def sample(self, target_n: int) -> pd.DataFrame:
        if self.normal_ is None:
            raise RuntimeError("Call fit() before sample().")
        additional_n = target_n - len(self.normal_)
        if additional_n < 0:
            raise ValueError("The number of normal rows is greater than target_n.")
        if additional_n == 0:
            return self.normal_.copy()

        counts = self.normal_[self.balance_feature].value_counts(dropna=False)
        observed = counts / counts.sum()
        uniform = pd.Series(1.0 / len(counts), index=counts.index)
        proportions = (1.0 - self.balance_lambda) * observed + self.balance_lambda * uniform

        exact = proportions * additional_n
        sample_counts = np.floor(exact).astype(int)
        remainder = int(additional_n - sample_counts.sum())
        if remainder:
            fractional = (exact - sample_counts).sort_values(ascending=False)
            sample_counts.loc[fractional.index[:remainder]] += 1
        self.additional_counts_ = sample_counts

        groups = []
        for group_number, (category, n_rows) in enumerate(sample_counts.items()):
            if n_rows == 0:
                continue
            group = self.normal_.loc[self.normal_[self.balance_feature] == category]
            groups.append(group.sample(
                n=int(n_rows),
                replace=int(n_rows) > len(group),
                random_state=self.random_state + group_number,
            ))
        additional = pd.concat(groups, ignore_index=True)
        result = pd.concat([self.normal_, additional], ignore_index=True)
        return result.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
