# Vendored from teammate Vadiraj Rao (branch vadi/nslkdd, commit 27f2229, expt-2/src/generators/) - unmodified.
"""Discretized frequent-itemset clustering sampler.

This module is deliberately independent of the teammate's generators. It
turns mixed-type normal traffic into deterministic row tokens, mines frequent
itemsets of size 1--3, assigns each row to one strongest matching itemset, and
resamples rows proportionally by the resulting cluster sizes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd


class ItemsetClusteredNormalSampler:
    """Cluster normal rows by discretized frequent itemsets and resample them."""

    RESIDUAL_CLUSTER = "__residual__"

    def __init__(
        self,
        min_support: float = 0.01,
        max_support: float = 0.50,
        n_bins: int = 5,
        max_itemset_size: int = 3,
        min_cluster_size: int = 50,
        minority_boost: float = 0.5,
        random_state: int = 42,
    ) -> None:
        if not 0 < min_support <= 1:
            raise ValueError("min_support must be in (0, 1].")
        if not min_support <= max_support <= 1:
            raise ValueError("max_support must be between min_support and 1.")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if max_itemset_size not in (1, 2, 3):
            raise ValueError("max_itemset_size must be 1, 2, or 3.")
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be positive.")
        if not 0.0 <= minority_boost <= 1.0:
            raise ValueError("minority_boost must be between 0 and 1.")

        self.min_support = min_support
        self.max_support = max_support
        self.n_bins = n_bins
        self.max_itemset_size = max_itemset_size
        self.min_cluster_size = min_cluster_size
        self.minority_boost = minority_boost
        self.random_state = random_state

        self.normal_: Optional[pd.DataFrame] = None
        self.feature_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.exact_numeric_cols_: list[str] = []
        self.continuous_cols_: list[str] = []
        self.constant_cols_: list[str] = []
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.row_tokens_: list[tuple[str, ...]] = []
        self.itemset_supports_: dict[tuple[str, ...], float] = {}
        self.itemset_counts_: dict[tuple[str, ...], int] = {}
        self.row_clusters_: pd.Series = pd.Series(dtype=object)
        self.cluster_rows_: dict[str, np.ndarray] = {}
        self.cluster_counts_: dict[str, int] = {}
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, label_col: str = "is_anomaly") -> "ItemsetClusteredNormalSampler":
        """Fit discretization, frequent itemsets, and non-overlapping clusters."""
        if label_col in df.columns:
            normal = df.loc[df[label_col] == 0].drop(columns=[label_col]).copy()
        else:
            normal = df.copy()
        if normal.empty:
            raise ValueError("No normal rows available for itemset clustering.")
        if normal.isna().any().any():
            raise ValueError("Normal rows contain missing values.")

        self.normal_ = normal.reset_index(drop=True)
        self.feature_cols_ = list(normal.columns)
        self._classify_columns()
        self.row_tokens_ = self._tokenize_rows(self.normal_)
        frequent = self._mine_frequent_itemsets(self.row_tokens_)
        self.itemset_counts_ = frequent
        n_rows = len(self.normal_)
        self.itemset_supports_ = {
            k: v / n_rows for k, v in frequent.items()
            if v / n_rows <= self.max_support
        }
        self.itemset_counts_ = {
            k: v for k, v in frequent.items() if k in self.itemset_supports_
        }

        self._assign_clusters()
        self.diagnostics_ = {
            "n_rows": n_rows,
            "n_frequent_itemsets": len(self.itemset_counts_),
            "n_clusters": len(self.cluster_rows_),
            "residual_rows": int(self.cluster_counts_.get(self.RESIDUAL_CLUSTER, 0)),
            "cluster_counts": dict(self.cluster_counts_),
            "bin_edges": {k: v.tolist() for k, v in self.bin_edges_.items()},
        }
        return self

    def sample(self, target_n: int) -> pd.DataFrame:
        """Preserve fitted rows and proportionally sample additional rows."""
        if self.normal_ is None:
            raise RuntimeError("Call fit() before sample().")
        additional_n = target_n - len(self.normal_)
        if additional_n < 0:
            raise ValueError("target_n cannot be smaller than the normal foundation.")
        if additional_n == 0:
            return self.normal_.copy()

        labels = list(self.cluster_rows_)
        counts = np.array([self.cluster_counts_[label] for label in labels], dtype=float)
        proportional = counts / counts.sum()
        inverse_frequency = 1.0 / counts
        inverse_frequency /= inverse_frequency.sum()
        sampling_weights = (
            (1.0 - self.minority_boost) * proportional
            + self.minority_boost * inverse_frequency
        )
        exact = sampling_weights * additional_n
        allocation = np.floor(exact).astype(int)
        remainder = int(additional_n - allocation.sum())
        if remainder:
            fractional_order = sorted(
                range(len(labels)),
                key=lambda i: (-(exact[i] - allocation[i]), labels[i]),
            )
            for i in fractional_order[:remainder]:
                allocation[i] += 1

        batches = [self.normal_]
        for group_number, (label, n_sample) in enumerate(zip(labels, allocation)):
            if n_sample == 0:
                continue
            indices = self.cluster_rows_[label]
            rng = np.random.default_rng(self.random_state + group_number)
            selected = rng.choice(indices, size=int(n_sample), replace=n_sample > len(indices))
            batches.append(self.normal_.iloc[selected])

        result = pd.concat(batches, ignore_index=True)
        return result.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

    def _classify_columns(self) -> None:
        assert self.normal_ is not None
        for col in self.feature_cols_:
            series = self.normal_[col]
            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                self.categorical_cols_.append(col)
                continue

            values = series.to_numpy(dtype=float)
            unique = np.unique(values)
            if len(unique) <= 1:
                self.constant_cols_.append(col)
            elif set(unique.tolist()).issubset({0, 1}) or (
                np.all(np.isclose(values, np.round(values))) and len(unique) <= 100
            ):
                self.exact_numeric_cols_.append(col)
            else:
                self.continuous_cols_.append(col)

    def _tokenize_rows(self, df: pd.DataFrame) -> list[tuple[str, ...]]:
        token_columns: dict[str, list[str]] = {}
        for col in self.feature_cols_:
            values = df[col]
            if col in self.continuous_cols_:
                raw = values.to_numpy(dtype=float)
                edges = np.unique(np.quantile(raw, np.linspace(0, 1, self.n_bins + 1)))
                self.bin_edges_[col] = edges
                if len(edges) <= 2:
                    labels = np.zeros(len(raw), dtype=int)
                else:
                    labels = np.searchsorted(edges[1:-1], raw, side="right")
                token_columns[col] = [f"{col}=q{int(x)}" for x in labels]
            elif col in self.constant_cols_:
                token_columns[col] = [f"{col}=constant"] * len(values)
            else:
                token_columns[col] = [f"{col}={value}" for value in values.astype(str)]

        return [
            tuple(sorted(token_columns[col][i] for col in self.feature_cols_))
            for i in range(len(df))
        ]

    def _mine_frequent_itemsets(self, rows: list[tuple[str, ...]]) -> dict[tuple[str, ...], int]:
        n_rows = len(rows)
        min_count = max(1, int(np.ceil(self.min_support * n_rows)))
        frequent: dict[tuple[str, ...], int] = {}

        singleton_counts = Counter(token for row in rows for token in set(row))
        level = {
            (token,): count
            for token, count in singleton_counts.items()
            if count >= min_count
        }
        frequent.update(level)

        for size in range(2, self.max_itemset_size + 1):
            if not level:
                break
            candidate_counts: Counter[tuple[str, ...]] = Counter()
            if size == 2:
                allowed = {item[0] for item in level}
                for row in rows:
                    tokens = sorted(set(row).intersection(allowed))
                    candidate_counts.update(combinations(tokens, 2))
            else:
                # Generate Apriori 3-item candidates by joining frequent
                # pairs, then count each candidate through inverted row
                # indexes. This avoids enumerating every row-level triple.
                frequent_pairs = {item for item in frequent if len(item) == 2}
                pair_by_token: dict[str, set[tuple[str, str]]] = defaultdict(set)
                for pair in frequent_pairs:
                    for token in pair:
                        pair_by_token[token].add(pair)
                candidate_triples: set[tuple[str, str, str]] = set()
                for token, pairs in pair_by_token.items():
                    pair_list = sorted(pairs)
                    for left, right in combinations(pair_list, 2):
                        union = tuple(sorted(set(left).union(right)))
                        if len(union) == 3 and all(
                            pair in frequent_pairs for pair in combinations(union, 2)
                        ):
                            candidate_triples.add(union)

                token_rows: dict[str, set[int]] = defaultdict(set)
                for row_index, row in enumerate(rows):
                    for token in set(row):
                        token_rows[token].add(row_index)
                for candidate in candidate_triples:
                    support_rows = set.intersection(*(token_rows[token] for token in candidate))
                    if len(support_rows) >= min_count:
                        candidate_counts[candidate] = len(support_rows)
            level = {
                item: count for item, count in candidate_counts.items() if count >= min_count
            }
            frequent.update(level)
        return frequent

    def _assign_clusters(self) -> None:
        assert self.normal_ is not None
        itemsets = list(self.itemset_counts_)
        # Index itemsets by token so a row only tests itemsets sharing at
        # least one of its tokens, rather than scanning every itemset.
        itemsets_by_token: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for itemset in itemsets:
            for token in itemset:
                itemsets_by_token[token].append(itemset)
        assignments: list[str] = []
        for row in self.row_tokens_:
            token_set = set(row)
            # Search from most-specific itemsets to least-specific ones. A
            # row normally matches many singleton/pair patterns, so stopping
            # after the first non-empty size avoids a large all-itemset scan.
            matches: list[tuple[str, ...]] = []
            for size in range(self.max_itemset_size, 0, -1):
                candidate_items = {
                    item
                    for token in token_set
                    for item in itemsets_by_token.get(token, [])
                    if len(item) == size
                }
                matches = [item for item in candidate_items if all(token in token_set for token in item)]
                if matches:
                    break
            if not matches:
                assignments.append(self.RESIDUAL_CLUSTER)
                continue
            best = min(matches, key=lambda item: (-len(item), -self.itemset_supports_[item], item))
            assignments.append("itemset:" + " | ".join(best))

        grouped: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(assignments):
            grouped[label].append(index)
        tiny = [label for label, indices in grouped.items() if len(indices) < self.min_cluster_size]
        residual = grouped.setdefault(self.RESIDUAL_CLUSTER, [])
        for label in tiny:
            if label != self.RESIDUAL_CLUSTER:
                residual.extend(grouped.pop(label))

        self.cluster_rows_ = {
            label: np.asarray(indices, dtype=int)
            for label, indices in sorted(grouped.items())
            if indices
        }
        self.cluster_counts_ = {label: len(indices) for label, indices in self.cluster_rows_.items()}
        final_labels = np.empty(len(assignments), dtype=object)
        for label, indices in self.cluster_rows_.items():
            final_labels[indices] = label
        self.row_clusters_ = pd.Series(final_labels, index=self.normal_.index, name="cluster")
