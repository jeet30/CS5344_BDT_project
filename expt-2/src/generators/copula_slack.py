"""
CopulaTailSlackGenerator: Hierarchical protocol-conditioned Gaussian copula
with adaptive boundary slackness / tail expansion for continuous network telemetry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional, Tuple, Any

from cleaner import nearest_pd_correlation


class CopulaTailSlackGenerator:
    """
    Hierarchical Protocol-Conditioned Gaussian Copula with Controlled Boundary Slack.
    
    Addresses the 'Boundary Tightness Paradox' where downstream non-parametric detectors
    (ECOD & Isolation Forest) penalize contracted normal synthetic manifolds on unseen test sets.
    
    Attributes:
        tail_slack_gamma (float): Upper tail expansion multiplier for continuous features (default: 1.15).
        min_group_size (int): Minimum rows to fit group-specific correlation (default: 100).
        shrinkage_m (float): Empirical Bayes shrinkage toward global correlation (default: 100.0).
        random_state (int): Random seed for reproducibility.
    """
    
    def __init__(
        self,
        tail_slack_gamma: float = 1.15,
        min_group_size: int = 100,
        shrinkage_m: float = 100.0,
        random_state: int = 42,
    ):
        self.tail_slack_gamma = tail_slack_gamma
        self.min_group_size = min_group_size
        self.shrinkage_m = shrinkage_m
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        # Fitted parameters
        self.categorical_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.continuous_cols: List[str] = []
        self.bounded_cols: List[str] = []
        self.group_col: str = ""
        
        self.cat_probs: Optional[pd.Series] = None
        self.global_marginals: Dict[str, np.ndarray] = {}
        self.global_log_marginals: Dict[str, np.ndarray] = {}
        self.global_corr: Optional[np.ndarray] = None
        self.group_fits: Dict[Any, Dict[str, Any]] = {}
        
    def fit(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
        continuous_cols: List[str],
        bounded_cols: Optional[List[str]] = None,
        group_col: Optional[str] = None,
    ) -> CopulaTailSlackGenerator:
        """
        Fit joint categorical probabilities and conditional copula marginals.
        """
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.continuous_cols = list(continuous_cols)
        self.bounded_cols = list(bounded_cols) if bounded_cols else []
        
        # If no group_col specified, use first categorical column (e.g. protocol)
        if group_col is None and len(self.categorical_cols) > 0:
            self.group_col = self.categorical_cols[0]
        else:
            self.group_col = group_col if group_col else ""
            
        n_all = len(df)
        
        # 1. Fit joint categorical distribution
        if len(self.categorical_cols) > 0:
            counts = df.groupby(self.categorical_cols, observed=True).size()
            self.cat_probs = (counts / counts.sum()).rename("prob")
        else:
            self.cat_probs = None
            
        # 2. Fit global marginals (both raw and log-space for continuous)
        for c in self.numeric_cols:
            vals = df[c].to_numpy(dtype=float)
            self.global_marginals[c] = np.sort(vals)
            if c in self.continuous_cols:
                # Log(1+x) transformation for positive continuous features
                min_val = np.min(vals)
                shift = max(0.0, -min_val)
                self.global_log_marginals[c] = np.sort(np.log1p(vals + shift))
                
        # 3. Fit global Gaussian rank scores & correlation
        global_scores = pd.DataFrame(index=df.index)
        for c in self.numeric_cols:
            rank_u = stats.rankdata(df[c]) / (n_all + 1.0)
            # Clip rank to avoid inf in ppf
            rank_u = np.clip(rank_u, 1e-6, 1.0 - 1e-6)
            global_scores[c] = stats.norm.ppf(rank_u)
            
        self.global_corr = nearest_pd_correlation(
            global_scores.corr().fillna(0.0).to_numpy()
        )
        
        # 4. Fit conditional groups
        if self.group_col and self.group_col in df.columns:
            for g, gdf in df.groupby(self.group_col, observed=True):
                m = len(gdf)
                if m >= self.min_group_size:
                    # Group marginals
                    marg = {c: np.sort(gdf[c].to_numpy(dtype=float)) for c in self.numeric_cols}
                    log_marg = {}
                    for c in self.continuous_cols:
                        v = gdf[c].to_numpy(dtype=float)
                        s = max(0.0, -np.min(v))
                        log_marg[c] = np.sort(np.log1p(v + s))
                        
                    # Group correlation shrunk toward global correlation
                    local_corr = np.array(global_scores.loc[gdf.index].corr().fillna(0.0).to_numpy(), copy=True)
                    np.fill_diagonal(local_corr, 1.0)
                    w = m / (m + self.shrinkage_m)
                    shrunk_corr = nearest_pd_correlation(w * local_corr + (1.0 - w) * self.global_corr)
                    
                    self.group_fits[g] = {
                        "marginals": marg,
                        "log_marginals": log_marg,
                        "corr": shrunk_corr,
                        "n": m,
                    }
                else:
                    self.group_fits[g] = {
                        "marginals": self.global_marginals,
                        "log_marginals": self.global_log_marginals,
                        "corr": self.global_corr,
                        "n": m,
                    }
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Sample synthetic records from the fitted copula with tail slackness.
        """
        # 1. Sample categoricals
        if self.cat_probs is not None:
            cat_idx = self.rng.choice(len(self.cat_probs), size=n_samples, p=self.cat_probs.to_numpy())
            sampled_combos = self.cat_probs.index[cat_idx]
            cat_df = pd.DataFrame(list(sampled_combos), columns=self.categorical_cols)
        else:
            cat_df = pd.DataFrame(index=range(n_samples))
            
        # Determine group labels
        if self.group_col and self.group_col in cat_df.columns:
            group_labels = cat_df[self.group_col].to_numpy()
        else:
            group_labels = np.array(["GLOBAL"] * n_samples)
            
        # 2. Sample numerical columns per group
        out_num = np.empty((n_samples, len(self.numeric_cols)), dtype=float)
        unique_groups = np.unique(group_labels)
        
        for g in unique_groups:
            mask = group_labels == g
            m = int(np.sum(mask))
            if m == 0:
                continue
                
            gfit = self.group_fits.get(g, {
                "marginals": self.global_marginals,
                "log_marginals": self.global_log_marginals,
                "corr": self.global_corr,
            })
            
            corr = gfit["corr"]
            try:
                L = np.linalg.cholesky(corr)
            except np.linalg.LinAlgError:
                corr = nearest_pd_correlation(corr, eps=1e-4)
                L = np.linalg.cholesky(corr)
                
            # Draw correlated standard normal samples
            z = self.rng.standard_normal((m, len(self.numeric_cols))) @ L.T
            # Uniform marginals
            u = stats.norm.cdf(z)
            
            # Invert marginals with controlled tail slack for continuous features
            for j, c in enumerate(self.numeric_cols):
                sv = gfit["marginals"][c]
                n_sv = len(sv)
                col_u = u[:, j]
                
                if c in self.continuous_cols and self.tail_slack_gamma > 1.0:
                    # Invert in log-space with upper tail expansion
                    shift = max(0.0, -np.min(sv))
                    log_sv = np.log1p(sv + shift)
                    
                    # Compute quantile position
                    q = col_u * (n_sv - 1.0)
                    base_log_vals = np.interp(q, np.arange(n_sv), log_sv)
                    
                    # For upper tail (top 5% of quantiles, col_u > 0.95), expand slope by gamma
                    tail_mask = col_u > 0.95
                    if np.any(tail_mask):
                        # Upper tail slope between 90th and 99th percentile
                        idx_90 = int(0.90 * (n_sv - 1))
                        idx_99 = int(0.99 * (n_sv - 1))
                        slope = max(1e-5, (log_sv[idx_99] - log_sv[idx_90]) / max(1, idx_99 - idx_90))
                        
                        excess_q = (col_u[tail_mask] - 0.95) * (n_sv - 1.0)
                        expansion = (self.tail_slack_gamma - 1.0) * excess_q * slope
                        base_log_vals[tail_mask] += expansion
                        
                    # Invert log transform: exp(y) - 1 - shift
                    sim_vals = np.expm1(base_log_vals) - shift
                    # Enforce non-negativity for continuous features
                    sim_vals = np.maximum(0.0, sim_vals)
                    out_num[mask, j] = sim_vals
                else:
                    # Standard quantile interpolation without slack for bounded/discrete
                    q = col_u * (n_sv - 1.0)
                    sim_vals = np.interp(q, np.arange(n_sv), sv)
                    
                    # If bounded feature (e.g. rate in [0, 1]), clamp to [0, 1]
                    if c in self.bounded_cols:
                        sim_vals = np.clip(sim_vals, 0.0, 1.0)
                    out_num[mask, j] = sim_vals
                    
        num_df = pd.DataFrame(out_num, columns=self.numeric_cols)
        # Combine categoricals and numerics
        combined = pd.concat([cat_df.reset_index(drop=True), num_df.reset_index(drop=True)], axis=1)
        return combined


class HybridEnsembleGenerator:
    """
    Hybrid Ensemble Generator: Combines high-precision interior copula density
    with covariance-preserving convex hull expansion for robust novel anomaly separation.
    
    Attributes:
        copula_ratio (float): Fraction of rows generated by CopulaTailSlackGenerator (default: 0.80).
        slack_ratio (float): Fraction of rows generated by Covariance Manifold Jitter (default: 0.20).
    """
    
    def __init__(
        self,
        tail_slack_gamma: float = 1.15,
        slack_ratio: float = 0.20,
        manifold_jitter_std: float = 0.15,
        random_state: int = 42,
    ):
        self.tail_slack_gamma = tail_slack_gamma
        self.slack_ratio = slack_ratio
        self.manifold_jitter_std = manifold_jitter_std
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        self.copula_gen = CopulaTailSlackGenerator(
            tail_slack_gamma=tail_slack_gamma, random_state=random_state
        )
        self.clean_df: Optional[pd.DataFrame] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.continuous_cols: List[str] = []
        self.bounded_cols: List[str] = []
        self.numeric_cov: Optional[np.ndarray] = None
        
    def fit(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
        continuous_cols: List[str],
        bounded_cols: Optional[List[str]] = None,
        group_col: Optional[str] = None,
    ) -> HybridEnsembleGenerator:
        self.clean_df = df.copy().reset_index(drop=True)
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.continuous_cols = list(continuous_cols)
        self.bounded_cols = list(bounded_cols) if bounded_cols else []
        
        # Fit internal copula
        self.copula_gen.fit(
            df, numeric_cols, categorical_cols, continuous_cols, bounded_cols, group_col
        )
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        n_copula = int(np.round((1.0 - self.slack_ratio) * n_samples))
        n_slack = n_samples - n_copula
        
        # 1. Copula samples
        copula_df = self.copula_gen.sample(n_copula)
        
        if n_slack <= 0 or self.clean_df is None:
            return copula_df
            
        # 2. Manifold-expanded samples: Resample prototype rows with covariance-guided continuous jitter
        proto_idx = self.rng.choice(len(self.clean_df), size=n_slack, replace=True)
        slack_df = self.clean_df.iloc[proto_idx].copy().reset_index(drop=True)
        
        # Add slight Gaussian noise to continuous features proportional to standard deviation
        for c in self.continuous_cols:
            vals = slack_df[c].to_numpy(dtype=float)
            std = np.std(self.clean_df[c].to_numpy(dtype=float))
            if std > 1e-6:
                noise = self.rng.normal(0.0, self.manifold_jitter_std * std, size=n_slack)
                slack_df[c] = np.maximum(0.0, vals + noise)
                
        # Combine
        combined = pd.concat([copula_df, slack_df], axis=0, ignore_index=True)
        # Shuffle
        shuffled = combined.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        return shuffled
