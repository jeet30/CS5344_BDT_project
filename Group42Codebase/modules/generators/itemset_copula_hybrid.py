# Author: Vadiraj Rao (Group 42).
"""Hybrid itemset-cluster allocation plus tail-slack copula generation.

Itemset clusters determine how many synthetic rows each normal subgroup gets;
CopulaTailSlackGenerator (copula_slack.py) generates the rows within each group.
This module is standalone and does not modify the generic copula generator
or any other pipeline stage.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from data_loader import classify_features
from generators.copula_slack import CopulaTailSlackGenerator
from generators.itemset_clustered import ItemsetClusteredNormalSampler


class ItemsetCopulaHybridSampler:
    """Use itemset clusters for allocation and the tail-slack copula for synthesis."""

    def __init__(
        self,
        min_support: float = 0.01,
        max_support: float = 0.35,
        max_itemset_size: int = 2,
        min_cluster_size: int = 50,
        min_copula_rows: int = 500,
        tail_slack_gamma: float = 1.05,
        n_copula_seeds: int = 2,
        random_state: int = 42,
    ) -> None:
        self.cluster_sampler = ItemsetClusteredNormalSampler(
            min_support=min_support,
            max_support=max_support,
            max_itemset_size=max_itemset_size,
            min_cluster_size=min_cluster_size,
            minority_boost=0.0,
            random_state=random_state,
        )
        self.min_copula_rows = min_copula_rows
        self.tail_slack_gamma = tail_slack_gamma
        self.n_copula_seeds = n_copula_seeds
        self.random_state = random_state
        self.normal_: Optional[pd.DataFrame] = None
        self.models_: dict[str, list[CopulaTailSlackGenerator]] = {}
        self.schema_: Optional[dict[str, Any]] = None
        self.cluster_counts_: dict[str, int] = {}
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, label_col: str = "is_anomaly") -> "ItemsetCopulaHybridSampler":
        self.cluster_sampler.fit(df, label_col=label_col)
        self.normal_ = self.cluster_sampler.normal_.copy()
        self.schema_ = classify_features(self.normal_)
        group_col = self._group_column()
        bounded_cols = [c for c in self.schema_["continuous_cols"] if "rate" in c]

        global_models = [
            self._fit_copula(self.normal_, group_col, bounded_cols, self.random_state + seed_offset)
            for seed_offset in range(self.n_copula_seeds)
        ]
        self.models_ = {"__global__": global_models}
        for cluster, indices in self.cluster_sampler.cluster_rows_.items():
            cluster_df = self.normal_.iloc[indices].reset_index(drop=True)
            if len(cluster_df) >= self.min_copula_rows:
                self.models_[cluster] = [
                    self._fit_copula(
                        cluster_df, group_col, bounded_cols,
                        self.random_state + seed_offset + len(self.models_),
                    )
                    for seed_offset in range(self.n_copula_seeds)
                ]

        self.cluster_counts_ = dict(self.cluster_sampler.cluster_counts_)
        self.diagnostics_ = {
            "itemset_clusters": len(self.cluster_counts_),
            "copula_models": sum(len(models) for models in self.models_.values()) - len(self.models_["__global__"]),
            "global_copula_fallback": sorted(
                set(self.cluster_counts_) - set(self.models_)
            ),
            "cluster_counts": self.cluster_counts_,
        }
        return self

    def sample(self, target_n: int) -> pd.DataFrame:
        if self.normal_ is None:
            raise RuntimeError("Call fit() before sample().")
        additional_n = target_n - len(self.normal_)
        if additional_n < 0:
            raise ValueError("target_n cannot be smaller than the normal foundation.")
        if additional_n == 0:
            return self.normal_.copy()

        labels = sorted(self.cluster_counts_)
        counts = np.array([self.cluster_counts_[label] for label in labels], dtype=float)
        exact = counts / counts.sum() * additional_n
        allocation = np.floor(exact).astype(int)
        remainder = int(additional_n - allocation.sum())
        order = sorted(
            range(len(labels)),
            key=lambda i: (-(exact[i] - allocation[i]), labels[i]),
        )
        for i in order[:remainder]:
            allocation[i] += 1

        batches = [self.normal_]
        for group_number, (label, n_rows) in enumerate(zip(labels, allocation)):
            if n_rows == 0:
                continue
            models = self.models_.get(label, self.models_["__global__"])
            seed_allocations = [n_rows // len(models)] * len(models)
            for seed_index in range(n_rows % len(models)):
                seed_allocations[seed_index] += 1
            for model, seed_n in zip(models, seed_allocations):
                if seed_n:
                    batches.append(model.sample(int(seed_n)))

        result = pd.concat(batches, ignore_index=True)
        return result.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

    def _group_column(self) -> str:
        assert self.schema_ is not None
        for candidate in ("protocol_type", "proto"):
            if candidate in self.schema_["categorical_cols"]:
                return candidate
        return self.schema_["categorical_cols"][0] if self.schema_["categorical_cols"] else ""

    def _fit_copula(
        self,
        df: pd.DataFrame,
        group_col: str,
        bounded_cols: list[str],
        seed: int,
    ) -> CopulaTailSlackGenerator:
        assert self.schema_ is not None
        model = CopulaTailSlackGenerator(
            random_state=seed,
            tail_slack_gamma=self.tail_slack_gamma,
        )
        model.fit(
            df,
            numeric_cols=self.schema_["numeric_cols"],
            categorical_cols=self.schema_["categorical_cols"],
            continuous_cols=self.schema_["continuous_cols"],
            bounded_cols=bounded_cols,
            group_col=group_col,
        )
        return model
