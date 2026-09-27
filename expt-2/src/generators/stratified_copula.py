"""
Stratified Multi-Subpopulation Copula with Structural Freezing & Anti-Shakeup Bagging.
Experiment 2 - CS5344 Anomaly Detection.

This generator operationalizes:
1. Multi-Subpopulation Decomposition (LAN vs. WAN vs. UDP normal traffic).
2. Zero-Preserving Structural Feature Freezing (Continuous DVS without corrupting discrete flags).
3. Log-Domain Marginal Estimation for heavy-tailed flow features.
4. Stratified Multi-Generator Bagging to protect against 36% Public / 64% Private test shakeup.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import norm
from cleaner import nearest_pd_correlation


class MultiSubpopulationStratifiedCopula:
    """
    Advanced tabular normal data synthesizer designed to eliminate false alarms
    on heterogeneous normal modes (LAN vs. WAN) and prevent 36%/64% test shakeup.
    """

    def __init__(
        self,
        random_state: int = 42,
        tau_utility: float = 0.15,
        tau_low: float = 0.05,
        alpha_dispersion: float = 8.0,
        shrinkage: float = 0.05,
        bagging_slack_ratio: float = 0.20,
        bagging_slack_gamma: float = 1.05,
        bagging_anchor_ratio: float = 0.10,
    ):
        self.random_state = random_state
        self.tau_utility = tau_utility
        self.tau_low = tau_low
        self.alpha_dispersion = alpha_dispersion
        self.shrinkage = shrinkage
        self.bagging_slack_ratio = bagging_slack_ratio
        self.bagging_slack_gamma = bagging_slack_gamma
        self.bagging_anchor_ratio = bagging_anchor_ratio

        self.subpop_models_: Dict[str, Dict[str, Any]] = {}
        self.subpop_weights_: Dict[str, float] = {}
        self.utility_corrs_: Dict[str, float] = {}
        self.high_utility_features_: List[str] = []
        self.continuous_confounders_: List[str] = []
        self.structural_discrete_features_: List[str] = []
        self.log_transformed_features_: List[str] = []
        self.columns_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.clean_normal_base_: Optional[pd.DataFrame] = None
        self.is_unsw_: bool = False

    def _classify_feature_structures(self, df: pd.DataFrame):
        """Partitions numeric features into continuous vs. structural discrete."""
        self.structural_discrete_features_ = []
        self.log_transformed_features_ = []

        for col in self.numeric_cols_:
            series = df[col].dropna()
            n_unique = series.nunique()
            zero_ratio = (series == 0).mean()

            # Structural flags: integer dtype or low cardinality or zero-dominated indicator
            if (
                col in [
                    "logged_in", "root_shell", "su_attempted", "num_root", "num_file_creations",
                    "num_shells", "num_access_files", "is_host_login", "is_guest_login",
                    "land", "urgent", "num_failed_logins", "num_compromised",
                    "ct_state_ttl", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "is_sm_ips_ports"
                ]
                or n_unique <= 10
                or (zero_ratio > 0.98 and n_unique < 50)
            ):
                self.structural_discrete_features_.append(col)
            else:
                # Heavy-tailed continuous features
                if series.min() >= 0 and series.max() > 500:
                    self.log_transformed_features_.append(col)

    def _determine_subpopulation(self, row: pd.Series) -> str:
        """Assigns a record to its operational subpopulation."""
        if self.is_unsw_:
            proto = str(row.get("proto", "tcp")).lower()
            if proto == "tcp":
                sttl = float(row.get("sttl", 31))
                if sttl <= 64:
                    return "tcp_lan"
                else:
                    return "tcp_wan"
            elif proto == "udp":
                return "udp"
            else:
                return "other_proto"
        else:
            # NSL-KDD
            proto = str(row.get("protocol_type", "tcp")).lower()
            return proto if proto in ["tcp", "udp", "icmp"] else "tcp"

    def fit(self, df: pd.DataFrame, label_col: Optional[str] = None):
        """
        Fits multi-subpopulation copula models with structural freezing and log-domain smoothing.
        """
        rng = np.random.default_rng(self.random_state)
        df_work = df.copy()

        # Detect dataset architecture
        self.is_unsw_ = "sttl" in df_work.columns and "proto" in df_work.columns

        # Assess utility correlations if labeled
        if label_col and label_col in df_work.columns:
            y = df_work[label_col].to_numpy()
            X_df = df_work.drop(columns=[label_col])
            df_norm = df_work[df_work[label_col] == 0].drop(columns=[label_col]).reset_index(drop=True)
        else:
            X_df = df_work
            df_norm = df_work.reset_index(drop=True)
            y = None

        self.columns_ = list(X_df.columns)
        self.numeric_cols_ = list(X_df.select_dtypes(include=[np.number]).columns)
        self.categorical_cols_ = [c for c in self.columns_ if c not in self.numeric_cols_]
        self.clean_normal_base_ = df_norm.copy()

        self._classify_feature_structures(df_norm)

        # Compute feature utility correlations
        if y is not None:
            self.utility_corrs_ = {}
            for col in self.numeric_cols_:
                s_val = X_df[col].to_numpy(dtype=float)
                s_val = np.nan_to_num(s_val, nan=0.0, posinf=0.0, neginf=0.0)
                norm_vals = df_norm[col].to_numpy(dtype=float)
                norm_vals = np.nan_to_num(norm_vals, nan=0.0, posinf=0.0, neginf=0.0)

                # Compute ECOD-like marginal deviation utility
                med = np.median(norm_vals)
                iqr = np.percentile(norm_vals, 75) - np.percentile(norm_vals, 25)
                scale = iqr if iqr > 1e-6 else (np.std(norm_vals) + 1e-6)
                dev = np.abs(s_val - med) / scale

                std_dev = np.std(dev)
                std_y = np.std(y)
                if std_dev > 1e-9 and std_y > 1e-9:
                    corr = float(np.corrcoef(dev, y)[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                self.utility_corrs_[col] = corr

            self.high_utility_features_ = [
                c for c, cr in self.utility_corrs_.items() if cr >= self.tau_utility
            ]
            # Confounders: negative correlation, but ONLY true continuous features!
            self.continuous_confounders_ = [
                c for c, cr in self.utility_corrs_.items()
                if cr < self.tau_low and c not in self.structural_discrete_features_
            ]
        else:
            self.high_utility_features_ = []
            self.continuous_confounders_ = []

        # Subpopulation Decomposition
        subpop_labels = [self._determine_subpopulation(row) for _, row in df_norm.iterrows()]
        df_norm["__subpop__"] = subpop_labels
        subpop_counts = df_norm["__subpop__"].value_counts()
        total_norm = len(df_norm)

        self.subpop_weights_ = {}
        self.subpop_models_ = {}

        for sp_name, count in subpop_counts.items():
            weight = count / total_norm
            self.subpop_weights_[sp_name] = weight
            sp_df = df_norm[df_norm["__subpop__"] == sp_name].drop(columns=["__subpop__"]).reset_index(drop=True)

            # Fit subpopulation-specific copula model
            sp_model = self._fit_single_subpopulation(sp_df)
            self.subpop_models_[sp_name] = sp_model

        return self

    def _fit_single_subpopulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fits Gaussian copula with log-transformed marginals for a subpopulation."""
        n_samples = len(df)
        model: Dict[str, Any] = {
            "n_samples": n_samples,
            "marginals": {},
            "empirical_df": df.copy(),
            "cat_distributions": {},
        }

        # Store categorical joint distributions
        for col in self.categorical_cols_:
            val_counts = df[col].value_counts(normalize=True)
            model["cat_distributions"][col] = (val_counts.index.tolist(), val_counts.to_numpy())

        # Fit continuous marginals & prepare Gaussian copula latent matrix
        Z_cols = []
        valid_cols = []

        for col in self.numeric_cols_:
            raw_vals = df[col].to_numpy(dtype=float)
            raw_vals = np.nan_to_num(raw_vals, nan=0.0)

            # Check if feature should be log-transformed
            is_log = col in self.log_transformed_features_
            trans_vals = np.log1p(np.maximum(0.0, raw_vals)) if is_log else raw_vals

            sorted_vals = np.sort(trans_vals)
            p_grid = (np.arange(1, n_samples + 1) - 0.5) / n_samples

            ranks = (np.searchsorted(sorted_vals, trans_vals, side="right") - 0.5) / n_samples
            ranks = np.clip(ranks, 1e-6, 1.0 - 1e-6)
            z = norm.ppf(ranks)

            model["marginals"][col] = {
                "sorted_vals": sorted_vals,
                "p_grid": p_grid,
                "is_log": is_log,
                "std": float(np.std(raw_vals)),
                "min": float(np.min(raw_vals)),
                "max": float(np.max(raw_vals)),
                "is_structural": col in self.structural_discrete_features_,
            }

            if np.std(z) > 1e-6:
                Z_cols.append(z)
                valid_cols.append(col)

        # Fit empirical Bayes shrunk correlation matrix
        if len(valid_cols) > 1:
            Z_matrix = np.column_stack(Z_cols)
            raw_corr = np.corrcoef(Z_matrix, rowvar=False)
            d = len(valid_cols)
            shrunk_corr = (1.0 - self.shrinkage) * raw_corr + self.shrinkage * np.eye(d)
            clean_corr = nearest_pd_correlation(shrunk_corr)
            try:
                L = np.linalg.cholesky(clean_corr)
            except np.linalg.LinAlgError:
                evals, evecs = np.linalg.eigh(clean_corr)
                evals = np.maximum(evals, 1e-5)
                L = evecs @ np.diag(np.sqrt(evals))
        else:
            valid_cols = []
            L = None

        model["valid_numeric_cols"] = valid_cols
        model["cholesky_L"] = L
        return model

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Samples n_samples records using stratified multi-generator bagging:
        - 70% Core Multi-Subpopulation Stratified Copula
        - 20% Tail-Slack Expanded Copula (gamma=1.05)
        - 10% Empirical Micro-Jitter Anchors
        """
        rng = np.random.default_rng(self.random_state)

        # Allocate sample counts for bagging mixture
        n_anchor = int(round(self.bagging_anchor_ratio * n_samples))
        n_slack = int(round(self.bagging_slack_ratio * n_samples))
        n_core = n_samples - n_anchor - n_slack

        batches = []

        # 1. Core Stratified Copula Samples
        if n_core > 0:
            core_df = self._sample_stratified_copula(n_core, rng, tail_slack=1.00)
            batches.append(core_df)

        # 2. Tail-Slack Expanded Copula Samples
        if n_slack > 0:
            slack_df = self._sample_stratified_copula(n_slack, rng, tail_slack=self.bagging_slack_gamma)
            batches.append(slack_df)

        # 3. Empirical Anchors with Micro-Jitter (Ground-truth anchors)
        if n_anchor > 0 and self.clean_normal_base_ is not None:
            idx = rng.choice(len(self.clean_normal_base_), size=n_anchor, replace=True)
            anchor_df = self.clean_normal_base_.iloc[idx].copy().reset_index(drop=True)
            # Micro-jitter continuous features slightly
            for col in self.numeric_cols_:
                if col not in self.structural_discrete_features_:
                    std_val = float(anchor_df[col].std())
                    if std_val > 1e-6:
                        noise = rng.normal(0.0, 0.005 * std_val, size=n_anchor)
                        anchor_df[col] = np.maximum(0.0, anchor_df[col] + noise)
            batches.append(anchor_df)

        # Combine, shuffle, and enforce schema
        syn_df = pd.concat(batches, ignore_index=True)
        syn_df = syn_df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        return syn_df[self.columns_]

    def _sample_stratified_copula(self, n_sub: int, rng: np.random.Generator, tail_slack: float = 1.00) -> pd.DataFrame:
        """Samples allocated records from subpopulation copulas."""
        sub_batches = []

        # Allocate counts per subpopulation
        sp_counts = {}
        allocated = 0
        sp_names = list(self.subpop_weights_.keys())
        for sp_name in sp_names[:-1]:
            c = int(round(self.subpop_weights_[sp_name] * n_sub))
            sp_counts[sp_name] = c
            allocated += c
        sp_counts[sp_names[-1]] = max(0, n_sub - allocated)

        for sp_name, count in sp_counts.items():
            if count <= 0:
                continue
            sp_model = self.subpop_models_[sp_name]
            sp_df = self._sample_single_subpopulation(sp_model, count, rng, tail_slack=tail_slack)
            sub_batches.append(sp_df)

        return pd.concat(sub_batches, ignore_index=True)

    def _sample_single_subpopulation(
        self, sp_model: Dict[str, Any], count: int, rng: np.random.Generator, tail_slack: float = 1.00
    ) -> pd.DataFrame:
        """Samples count records from a single subpopulation model."""
        out_dict: Dict[str, np.ndarray] = {}

        # 1. Sample categoricals
        for col in self.categorical_cols_:
            cats, probs = sp_model["cat_distributions"][col]
            out_dict[col] = rng.choice(cats, size=count, p=probs)

        # 2. Sample Gaussian copula latent variables
        valid_cols = sp_model["valid_numeric_cols"]
        L = sp_model["cholesky_L"]

        if L is not None and len(valid_cols) > 0:
            d = len(valid_cols)
            z_ind = rng.standard_normal((count, d))
            # Apply tail slack expansion to Gaussian latent radius if requested
            if tail_slack > 1.0:
                r = np.linalg.norm(z_ind, axis=1, keepdims=True)
                u_dir = z_ind / np.maximum(r, 1e-9)
                r_exp = r * rng.uniform(1.0, tail_slack, size=(count, 1))
                z_ind = u_dir * r_exp
            z_corr = z_ind @ L.T
            u_corr = norm.cdf(z_corr)
        else:
            u_corr = None

        # 3. Map to marginal feature space
        for col in self.numeric_cols_:
            m_info = sp_model["marginals"][col]
            sorted_vals = m_info["sorted_vals"]
            p_grid = m_info["p_grid"]
            is_log = m_info["is_log"]
            is_struct = m_info["is_structural"]

            if col in valid_cols and u_corr is not None:
                col_idx = valid_cols.index(col)
                u = np.clip(u_corr[:, col_idx], 1e-6, 1.0 - 1e-6)
            else:
                u = rng.uniform(1e-6, 1.0 - 1e-6, size=count)

            # Invert marginal
            trans_sample = np.interp(u, p_grid, sorted_vals)

            if is_log:
                raw_sample = np.expm1(trans_sample)
                raw_sample = np.maximum(0.0, raw_sample)
            else:
                raw_sample = trans_sample

            # 4. Apply DVS dispersion ONLY on continuous confounding features
            # Discrete structural features are strictly frozen!
            if col in self.continuous_confounders_ and not is_struct:
                corr = self.utility_corrs_.get(col, 0.0)
                deficit = max(0.0, self.tau_low - corr)
                disp_scale = self.alpha_dispersion * deficit * (m_info["std"] + 1e-6)
                disp_noise = rng.normal(0.0, disp_scale, size=count)
                raw_sample = np.maximum(0.0, raw_sample + disp_noise)

            out_dict[col] = raw_sample

        res_df = pd.DataFrame(out_dict)
        return res_df

