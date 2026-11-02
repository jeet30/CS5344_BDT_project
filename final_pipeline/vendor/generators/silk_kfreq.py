"""SILK-style MinHash seeding followed by iterative k-FreqItems clustering."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd


class SILKKFreqClusterer:
    """Cluster discretised mixed rows using LSH seeds and Jaccard refinement.

    Numerical columns are quantile-tokenised; categorical and low-cardinality
    integer columns remain exact.  SILK candidate buckets are formed from
    MinHash bands, then weighted Jaccard k-means++ chooses separated seeds.
    """

    def __init__(self, k=15, n_bins=5, n_hashes=48, n_bands=12,
                 center_frequency=0.25, max_iter=25, min_cluster_size=500,
                 random_state=42):
        if n_hashes % n_bands:
            raise ValueError("n_hashes must be divisible by n_bands")
        self.k, self.n_bins = k, n_bins
        self.n_hashes, self.n_bands = n_hashes, n_bands
        self.center_frequency, self.max_iter = center_frequency, max_iter
        self.min_cluster_size, self.random_state = min_cluster_size, random_state
        self.normal_: pd.DataFrame | None = None
        self.tokens_: list[frozenset[str]] = []
        self.cluster_rows_: dict[str, np.ndarray] = {}
        self.cluster_counts_: dict[str, int] = {}
        self.centers_: list[frozenset[str]] = []
        self.bin_edges_: dict[str, list[float]] = {}
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, label_col="is_anomaly"):
        normal = df.loc[df[label_col] == 0].drop(columns=label_col) if label_col in df else df
        self.normal_ = normal.reset_index(drop=True).copy()
        self.tokens_ = self._tokenize(self.normal_)
        candidates, weights = self._silk_candidates(self.tokens_)
        centers = self._seeds(candidates, weights, min(self.k, len(candidates)))
        labels = np.full(len(self.tokens_), -1, dtype=int)
        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            new_labels = self._assign(self.tokens_, centers)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            centers = self._update(self.tokens_, labels, centers)
        labels = self._merge_small(labels, centers)
        unique = sorted(np.unique(labels))
        remap = {old: new for new, old in enumerate(unique)}
        labels = np.array([remap[x] for x in labels], dtype=int)
        self.cluster_rows_ = {f"silk_{i:02d}": np.flatnonzero(labels == i) for i in sorted(np.unique(labels))}
        self.cluster_counts_ = {k: len(v) for k, v in self.cluster_rows_.items()}
        self.centers_ = self._update(self.tokens_, labels, [frozenset()] * len(self.cluster_rows_))
        self.diagnostics_ = dict(candidate_buckets=len(candidates), requested_k=self.k,
                                 final_clusters=len(self.cluster_rows_), iterations=iterations,
                                 cluster_counts=self.cluster_counts_, bin_edges=self.bin_edges_)
        return self

    def _tokenize(self, df):
        columns = {}
        for c in df:
            s = df[c]
            if not pd.api.types.is_numeric_dtype(s):
                columns[c] = [f"{c}={x}" for x in s.astype(str)]
                continue
            x = s.to_numpy(float); u = np.unique(x)
            if len(u) <= 100 and np.allclose(x, np.rint(x)):
                columns[c] = [f"{c}={v:g}" for v in x]
            elif len(u) == 1:
                columns[c] = [f"{c}=constant"] * len(x)
            else:
                edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)))
                self.bin_edges_[c] = edges.tolist()
                q = np.searchsorted(edges[1:-1], x, side="right")
                columns[c] = [f"{c}=q{v}" for v in q]
        return [frozenset(columns[c][i] for c in df) for i in range(len(df))]

    @staticmethod
    def _stable_hash(token):
        # FNV-1a is deterministic across Python processes.
        h = 1469598103934665603
        for b in token.encode():
            h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
        return h

    def _silk_candidates(self, rows):
        prime = np.uint64(4294967311); rng = np.random.default_rng(self.random_state)
        a = rng.integers(1, int(prime), self.n_hashes, dtype=np.uint64)
        b = rng.integers(0, int(prime), self.n_hashes, dtype=np.uint64)
        sig = np.empty((len(rows), self.n_hashes), dtype=np.uint64)
        for i, row in enumerate(rows):
            h = np.array([self._stable_hash(t) % int(prime) for t in row], dtype=np.uint64)
            sig[i] = ((h[:, None] * a + b) % prime).min(axis=0)
        width = self.n_hashes // self.n_bands; buckets = defaultdict(list)
        for i in range(len(rows)):
            for band in range(self.n_bands):
                sl = sig[i, band * width:(band + 1) * width]
                buckets[(band, sl.tobytes())].append(i)
        groups = sorted((v for v in buckets.values() if len(v) >= 2), key=lambda x: (-len(x), x[0]))
        candidates, weights, seen = [], [], set()
        for ids in groups:
            counts = Counter(t for i in ids for t in rows[i])
            center = frozenset(t for t, n in counts.items() if n / len(ids) >= self.center_frequency)
            if center and center not in seen:
                seen.add(center); candidates.append(center); weights.append(len(ids))
        # Ensure isolated modes remain eligible.
        for row, n in Counter(rows).most_common():
            if row not in seen:
                candidates.append(row); weights.append(n)
        return candidates, np.asarray(weights, float)

    @staticmethod
    def _distance(a, b):
        union = len(a | b)
        return 1.0 - len(a & b) / union if union else 0.0

    def _seeds(self, candidates, weights, k):
        rng = np.random.default_rng(self.random_state)
        first = int(np.argmax(weights)); chosen = [first]
        nearest = np.array([self._distance(x, candidates[first]) for x in candidates])
        while len(chosen) < k:
            p = weights * nearest ** 2; p[chosen] = 0
            if p.sum() == 0: break
            nxt = int(rng.choice(len(candidates), p=p / p.sum())); chosen.append(nxt)
            nearest = np.minimum(nearest, [self._distance(x, candidates[nxt]) for x in candidates])
        return [candidates[i] for i in chosen]

    def _assign(self, rows, centers):
        return np.array([min(range(len(centers)), key=lambda j: (self._distance(row, centers[j]), j))
                         for row in rows], dtype=int)

    def _update(self, rows, labels, old):
        centers = []
        for j in range(max(labels) + 1):
            ids = np.flatnonzero(labels == j)
            if not len(ids):
                centers.append(old[j]); continue
            counts = Counter(t for i in ids for t in rows[i])
            center = frozenset(t for t, n in counts.items() if n / len(ids) >= self.center_frequency)
            centers.append(center or rows[ids[0]])
        return centers

    def _merge_small(self, labels, centers):
        counts = np.bincount(labels, minlength=len(centers)); large = np.flatnonzero(counts >= self.min_cluster_size)
        if not len(large):
            return labels
        out = labels.copy()
        for small in np.flatnonzero(counts < self.min_cluster_size):
            target = min(large, key=lambda j: (self._distance(centers[small], centers[j]), j))
            out[labels == small] = target
        return out
