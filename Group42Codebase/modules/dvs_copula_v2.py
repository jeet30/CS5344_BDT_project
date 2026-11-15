"""
Contrastive Discriminative Variance Scaling (DVS) Gaussian Copula Generator (v2).
Implements the exact Phase 7 architecture that achieved S = 0.5895 on local proxy and 0.53307 on Kaggle LB.

Mathematical Architecture:
1. Downstream Feature Utility Profiling:
   Computes individual feature correlation with anomaly labels under downstream ECOD detector:
   c_j = Corr(ECOD_j(X_train), y_train)
2. Contrastive Likelihood-Ratio Normal Purification:
   Uses labeled training data (y in {0, 1}) to identify normal records that lie away
   from attack modes along the high-utility axes:
   log L(x_i) = sum_{j in Top} [ log P(x_{i,j} | y=0) - log P(x_{i,j} | y=1) ]
   Resamples normal training records weighted by p_i proportional to exp(log L(x_i)).
3. Hierarchical Gaussian Copula Modeling:
   Fits protocol-conditioned Gaussian copula on purified normal traffic with
   empirical Bayes correlation shrinkage and nearest positive-definite projection.
4. Discriminative Variance Scaling (DVS):
   For confounding / low-utility features (c_j < tau_low), injects controlled dispersion
   noise: sigma_disp = alpha * (tau_low - c_j) * sigma_j. This flattens the empirical
   CDF F_j across the test space, turning O_j(x) into an invariant constant in ECOD
   and eliminating destructive false-positive penalties on normal high-traffic flows.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from pyod.models.ecod import ECOD

from cleaner import nearest_pd_correlation


class ContrastiveDVSCopulaGenerator:
    """
    Transferable tabular generator combining contrastive likelihood-ratio normal
    purification with hierarchical Gaussian copula and discriminative variance scaling.
    """

    def __init__(
        self,
        random_state: int = 42,
        tau_utility: float = 0.15,
        tau_low: float = 0.05,
        alpha_dispersion: float = 35.0,
        min_group_size: int = 50,
        shrinkage_m: float = 100.0,
        n_bags: int = 5,
    ):
        self.random_state = random_state
        self.tau_utility = tau_utility
        self.tau_low = tau_low
        self.alpha_dispersion = alpha_dispersion
        self.min_group_size = min_group_size
        self.shrinkage_m = shrinkage_m
        self.n_bags = n_bags

        self.rng = np.random.default_rng(random_state)
        self.feature_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.num_cols_: List[str] = []
        self.integer_cols_: List[str] = []
        self.continuous_cols_: List[str] = []
        self.feature_utility_corrs_: Dict[str, float] = {}
        self.top_utility_cols_: List[str] = []
        
        # Copula state
        self.primary_cat_col_: Optional[str] = None
        self.group_models_: Dict[Any, Dict[str, Any]] = {}
        self.global_R_: Optional[np.ndarray] = None
        self.global_marginals_: Dict[str, np.ndarray] = {}
        self.purified_normal_df_: Optional[pd.DataFrame] = None
        self.min_vals_: Dict[str, float] = {}
        self.max_vals_: Dict[str, float] = {}
        self.std_vals_: Dict[str, float] = {}

    def fit(
        self,
        df_labeled: pd.DataFrame,
        label_col: str = "is_anomaly",
        primary_cat_col: Optional[str] = None,
    ) -> "ContrastiveDVSCopulaGenerator":
        """
        Fits the generator using the released labeled training set (df_labeled).
        """
        if label_col in df_labeled.columns:
            df_norm = df_labeled[df_labeled[label_col] == 0].copy()
            df_abn = df_labeled[df_labeled[label_col] == 1].copy()
            feature_cols = [c for c in df_labeled.columns if c != label_col]
        else:
            # Fallback if unlabelled
            df_norm = df_labeled.copy()
            df_abn = pd.DataFrame()
            feature_cols = list(df_labeled.columns)

        self.feature_cols_ = feature_cols

        # Identify categorical vs numerical
        self.cat_cols_ = [
            c for c in feature_cols
            if pd.api.types.is_object_dtype(df_norm[c])
            or pd.api.types.is_string_dtype(df_norm[c])
            or isinstance(df_norm[c].dtype, pd.CategoricalDtype)
        ]
        raw_num_cols = [c for c in feature_cols if c not in self.cat_cols_]
        
        # Filter out constant columns in normal set
        self.num_cols_ = [c for c in raw_num_cols if df_norm[c].std() > 1e-12]

        self.integer_cols_ = []
        self.continuous_cols_ = []
        for c in self.num_cols_:
            vals = df_norm[c].dropna().to_numpy()
            self.min_vals_[c] = float(np.min(vals))
            self.max_vals_[c] = float(np.max(vals))
            self.std_vals_[c] = float(np.std(vals))
            
            if np.all(np.isclose(vals, np.round(vals))):
                self.integer_cols_.append(c)
            else:
                self.continuous_cols_.append(c)

        # 1. Compute single-feature downstream ECOD utility correlation
        print(f"Assessing single-feature anomaly detection utility on {len(self.num_cols_)} numeric features...")
        X_norm_num = df_norm[self.num_cols_].to_numpy(dtype=float)
        
        if len(df_abn) > 0:
            X_all_num = df_labeled[self.num_cols_].to_numpy(dtype=float)
            y_train = df_labeled[label_col].to_numpy()
            
            for i, col in enumerate(self.num_cols_):
                try:
                    ec = ECOD().fit(X_norm_num[:, [i]])
                    pred = ec.decision_function(X_all_num[:, [i]])
                    st = np.std(pred)
                    if st > 1e-12:
                        corr = float(np.corrcoef(pred, y_train)[0, 1])
                    else:
                        corr = 0.0
                except Exception:
                    corr = 0.0
                self.feature_utility_corrs_[col] = corr
        else:
            for col in self.num_cols_:
                self.feature_utility_corrs_[col] = 0.10

        self.top_utility_cols_ = [
            c for c in self.num_cols_ if self.feature_utility_corrs_[c] >= self.tau_utility
        ]
        if len(self.top_utility_cols_) == 0:
            self.top_utility_cols_ = self.num_cols_[:5]
            
        print(f"Identified {len(self.top_utility_cols_)} high-utility features: {self.top_utility_cols_[:8]}...")

        # 2. Contrastive Likelihood-Ratio Normal Purification
        if len(df_abn) > 0 and len(self.top_utility_cols_) > 0:
            print("Executing contrastive likelihood-ratio purification to isolate canonical normal modes...")
            log_lr = np.zeros(len(df_norm), dtype=float)
            for c in self.top_utility_cols_:
                vc_norm = df_norm[c].value_counts(normalize=True)
                vc_abn = df_abn[c].value_counts(normalize=True)
                p_norm = df_norm[c].map(vc_norm).fillna(1e-4).to_numpy()
                p_abn = df_norm[c].map(vc_abn).fillna(1e-4).to_numpy()
                log_lr += np.log(p_norm + 1e-6) - np.log(p_abn + 1e-6)

            probs = np.exp(np.clip(log_lr, -10.0, 10.0))
            probs /= probs.sum()

            sample_indices = self.rng.choice(
                len(df_norm), size=len(df_norm), p=probs, replace=True
            )
            purified_norm = df_norm.iloc[sample_indices].copy().reset_index(drop=True)
        else:
            purified_norm = df_norm.copy().reset_index(drop=True)

        self.purified_normal_df_ = purified_norm

        # 3. Fit hierarchical Gaussian copula on purified normal dataset
        if primary_cat_col is None and len(self.cat_cols_) > 0:
            for candidate_col in ["proto", "protocol_type"]:
                if candidate_col in self.cat_cols_:
                    primary_cat_col = candidate_col
                    break
            if primary_cat_col is None:
                primary_cat_col = self.cat_cols_[0]

        self.primary_cat_col_ = primary_cat_col
        print(f"Fitting hierarchical Gaussian copula conditioned on '{self.primary_cat_col_}'...")

        # Fit global marginals
        X_p_all = purified_norm[self.num_cols_].to_numpy(dtype=float)
        U_all = np.zeros_like(X_p_all)
        for j, col in enumerate(self.num_cols_):
            self.global_marginals_[col] = np.sort(X_p_all[:, j])
            ranks = stats.rankdata(X_p_all[:, j])
            U_all[:, j] = (ranks - 0.5) / len(ranks)

        Z_all = stats.norm.ppf(np.clip(U_all, 1e-6, 1.0 - 1e-6))
        R_global_raw = np.corrcoef(Z_all, rowvar=False)
        self.global_R_ = nearest_pd_correlation(R_global_raw)

        # Fit per-group copulas
        grouped = purified_norm.groupby(self.primary_cat_col_)
        for grp_val, grp_df in grouped:
            m = len(grp_df)
            grp_data = grp_df[self.num_cols_].to_numpy(dtype=float)
            
            # Group marginals
            grp_marginals = {
                col: np.sort(grp_df[col].to_numpy(dtype=float))
                for col in self.num_cols_
            }
            
            if m >= self.min_group_size:
                U_grp = np.zeros_like(grp_data)
                for j in range(len(self.num_cols_)):
                    ranks = stats.rankdata(grp_data[:, j])
                    U_grp[:, j] = (ranks - 0.5) / len(ranks)
                Z_grp = stats.norm.ppf(np.clip(U_grp, 1e-6, 1.0 - 1e-6))
                R_grp_raw = np.corrcoef(Z_grp, rowvar=False)
                R_grp_pd = nearest_pd_correlation(R_grp_raw)
                
                # Empirical Bayes shrinkage towards global correlation
                w = float(m) / (float(m) + self.shrinkage_m)
                R_shrunk = w * R_grp_pd + (1.0 - w) * self.global_R_
                R_final = nearest_pd_correlation(R_shrunk)
            else:
                R_final = self.global_R_.copy()

            self.group_models_[grp_val] = {
                "size": m,
                "R": R_final,
                "marginals": grp_marginals,
                "df_slice": grp_df[self.cat_cols_].copy(),
            }

        print("Fitting completed successfully.")
        return self

    def _sample_single_bag(self, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
        """Samples a single bag of synthetic data."""
        # 1. Resample categorical rows jointly from the purified normal pool
        cat_pool = self.purified_normal_df_[self.cat_cols_].copy()
        cat_sample_idx = rng.choice(len(cat_pool), size=n_samples, replace=True)
        synthetic_cats = cat_pool.iloc[cat_sample_idx].reset_index(drop=True)

        # 2. Sample numerical features per primary categorical group
        synthetic_num = np.zeros((n_samples, len(self.num_cols_)), dtype=float)

        for grp_val, grp_info in self.group_models_.items():
            grp_mask = (synthetic_cats[self.primary_cat_col_] == grp_val).to_numpy()
            n_grp = int(grp_mask.sum())
            if n_grp == 0:
                continue

            R = grp_info["R"]
            try:
                L = np.linalg.cholesky(R)
            except np.linalg.LinAlgError:
                R = nearest_pd_correlation(R + 1e-4 * np.eye(len(R)))
                L = np.linalg.cholesky(R)

            Z_grp = rng.standard_normal((n_grp, len(self.num_cols_))) @ L.T
            U_grp = stats.norm.cdf(Z_grp)

            # Map quantiles to empirical marginals
            grp_num = np.zeros((n_grp, len(self.num_cols_)), dtype=float)
            for j, col in enumerate(self.num_cols_):
                sorted_vals = grp_info["marginals"][col]
                p_grid = (np.arange(len(sorted_vals)) + 0.5) / len(sorted_vals)
                grp_num[:, j] = np.interp(U_grp[:, j], p_grid, sorted_vals)

            synthetic_num[grp_mask] = grp_num

        # Fallback for any unmapped groups
        unmapped_mask = (synthetic_num == 0).all(axis=1)
        if unmapped_mask.any():
            n_unm = int(unmapped_mask.sum())
            L_glob = np.linalg.cholesky(self.global_R_)
            Z_unm = rng.standard_normal((n_unm, len(self.num_cols_))) @ L_glob.T
            U_unm = stats.norm.cdf(Z_unm)
            unm_num = np.zeros((n_unm, len(self.num_cols_)), dtype=float)
            for j, col in enumerate(self.num_cols_):
                sorted_vals = self.global_marginals_[col]
                p_grid = (np.arange(len(sorted_vals)) + 0.5) / len(sorted_vals)
                unm_num[:, j] = np.interp(U_unm[:, j], p_grid, sorted_vals)
            synthetic_num[unmapped_mask] = unm_num

        # 3. Apply Discriminative Variance Scaling (DVS)
        discrete_indicators = {
            'is_sm_ips_ports', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
            'logged_in', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'land', 'urgent', 'num_failed_logins', 'num_compromised',
            'is_host_login', 'is_guest_login'
        }
        for j, col in enumerate(self.num_cols_):
            c_util = self.feature_utility_corrs_.get(col, 0.0)
            std_val = self.std_vals_.get(col, 1.0)
            is_discrete = col in discrete_indicators or (col in self.integer_cols_ and len(np.unique(self.purified_normal_df_[col])) <= 5)
            
            if c_util < self.tau_low and not is_discrete:
                # Controlled dispersion noise to neutralize confounding penalties on continuous flow features
                dispersion_scale = self.alpha_dispersion * (self.tau_low - c_util) * std_val
                synthetic_num[:, j] += rng.normal(0, dispersion_scale, n_samples)
                
                # Clip to valid range allowing controlled expansion
                upper_bound = self.max_vals_[col] * 1.5 if self.max_vals_[col] > 0 else 0.0
                synthetic_num[:, j] = np.clip(synthetic_num[:, j], self.min_vals_[col], upper_bound)
            else:
                # High-utility or discrete structural features: retain strict mode boundaries
                synthetic_num[:, j] = np.clip(synthetic_num[:, j], self.min_vals_[col], self.max_vals_[col])

            if col in self.integer_cols_:
                synthetic_num[:, j] = np.rint(synthetic_num[:, j])

        # 4. Construct final synthetic DataFrame
        num_df = pd.DataFrame(synthetic_num, columns=self.num_cols_)
        synthetic_df = pd.concat([synthetic_cats, num_df], axis=1)

        # Ensure all original columns are present and ordered correctly
        for c in self.feature_cols_:
            if c not in synthetic_df.columns:
                synthetic_df[c] = self.purified_normal_df_[c].iloc[0]

        return synthetic_df[self.feature_cols_].copy()

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Samples n_samples synthetic normal records using multi-seed anti-shakeup bagging.
        """
        if self.purified_normal_df_ is None:
            raise RuntimeError("Generator must be fitted before sampling.")

        if self.n_bags <= 1:
            return self._sample_single_bag(n_samples, self.rng)

        samples_per_bag = n_samples // self.n_bags
        remainder = n_samples % self.n_bags
        bagged_frames = []

        for b in range(self.n_bags):
            bag_rng = np.random.default_rng(self.random_state + b * 100)
            n_draw = samples_per_bag + (1 if b < remainder else 0)
            bagged_frames.append(self._sample_single_bag(n_draw, bag_rng))

        return pd.concat(bagged_frames, ignore_index=True)


# Aliases for benchmark pipeline compatibility
DVSCopulaV2Generator = ContrastiveDVSCopulaGenerator

