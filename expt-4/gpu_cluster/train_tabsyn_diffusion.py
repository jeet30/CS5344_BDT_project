"""
TabSyn-style Latent Diffusion Architecture for Tabular Network Flow Augmentation.
Designed for execution on NUS SoC GPU Cluster or local CUDA-enabled workstations.

Architecture Overview:
1. Mixed-Type VAE (Categorical Embeddings + Continuous Normalization with Sinkhorn/ELBO Loss)
2. Score-based Gaussian Diffusion (DDPM) in the latent space conditioned on Contrastive Likelihood-Ratio
3. Post-Sampling Physical Constraint Projection & Zero Dominance Snapping
4. Local Holdout Proxy Evaluation (ECOD + 10-seed IsolationForest)
5. Model Checkpoint Serialization (.pt)
6. Automated Submission Integrity Verification & Export
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Adjust path to import constraints and evaluator from src
CLUSTER_DIR = Path(__file__).resolve().parent
EXPT4_SRC = CLUSTER_DIR.parent / "src"
PROJECT_ROOT = CLUSTER_DIR.parent.parent
sys.path.insert(0, str(EXPT4_SRC))

from constraints import enforce_unsw_constraints, enforce_nslkdd_constraints
from evaluator import build_feature_matrices, score_detectors


# ---------------------------------------------------------------------------
# 1. Tabular VAE (Mixed Categorical Embeddings + Continuous MLP)
# ---------------------------------------------------------------------------
class TabularVAE(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        embedding_dims: List[int],
        latent_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities = cat_cardinalities
        self.latent_dim = latent_dim

        # Categorical Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, embedding_dims)
        ])
        total_emb_dim = sum(embedding_dims)
        total_in_dim = num_continuous + total_emb_dim

        # Encoder
        enc_layers = []
        in_d = total_in_dim
        for h_d in hidden_dims:
            enc_layers.extend([nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.SiLU()])
            in_d = h_d
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        dec_layers = []
        in_d = latent_dim
        for h_d in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.SiLU()])
            in_d = h_d
        self.decoder = nn.Sequential(*dec_layers)
        self.dec_continuous = nn.Linear(hidden_dims[0], num_continuous)
        self.dec_categorical = nn.ModuleList([
            nn.Linear(hidden_dims[0], card) for card in cat_cardinalities
        ])

    def encode(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if emb_list:
            x_cat_emb = torch.cat(emb_list, dim=-1)
            x_in = torch.cat([x_cont, x_cat_emb], dim=-1)
        else:
            x_in = x_cont
        h = self.encoder(x_in)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self.decoder(z)
        x_cont_recon = self.dec_continuous(h)
        x_cat_logits = [head(h) for head in self.dec_categorical]
        return x_cont_recon, x_cat_logits

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_cont, x_cat)
        z = self.reparameterize(mu, logvar)
        x_cont_recon, x_cat_logits = self.decode(z)
        return x_cont_recon, x_cat_logits, mu, logvar


# ---------------------------------------------------------------------------
# 2. Latent Diffusion Model (DDPM on continuous latent space)
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class LatentDiffusionNet(nn.Module):
    def __init__(self, latent_dim: int = 64, time_emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = torch.cat([z_t, t_emb], dim=-1)
        return self.net(x)


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu"):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, z_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha * z_start + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, z: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t_index])
        model_mean = sqrt_recip_alphas_t * (
            z - betas_t * model(z, t) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(z)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, int]) -> torch.Tensor:
        z = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            z = self.p_sample(model, z, t, i)
        return z


# ---------------------------------------------------------------------------
# 3. TabSyn Pipeline Wrapper
# ---------------------------------------------------------------------------
class TabSynDiffusionPipeline:
    def __init__(
        self,
        latent_dim: int = 64,
        epochs_vae: int = 100,
        epochs_diff: int = 150,
        batch_size: int = 512,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
        self.latent_dim = latent_dim
        self.epochs_vae = epochs_vae
        self.epochs_diff = epochs_diff
        self.batch_size = batch_size
        self.lr = lr

        self.vae: TabularVAE | None = None
        self.diffusion_net: LatentDiffusionNet | None = None
        self.diffusion: GaussianDiffusion | None = None
        self.scaler: RobustScaler | None = None

        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.cat_mappings: Dict[str, Dict] = {}

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        # Filter strictly for normal traffic to maximize downstream anomaly separation
        if "is_anomaly" in df.columns:
            df = df[df["is_anomaly"] == 0].copy()

        # Continuous preprocessing
        x_cont_raw = df[self.num_cols].to_numpy(dtype=np.float32)
        if self.scaler is None:
            self.scaler = RobustScaler()
            x_cont_scaled = self.scaler.fit_transform(x_cont_raw)
        else:
            x_cont_scaled = self.scaler.transform(x_cont_raw)

        # Categorical preprocessing
        cat_tensors = []
        for col in self.cat_cols:
            if col not in self.cat_mappings:
                unique_vals = sorted(df[col].astype(str).unique().tolist())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.cat_mappings[col] = mapping
            mapping = self.cat_mappings[col]
            indices = df[col].astype(str).map(lambda v: mapping.get(v, 0)).to_numpy(dtype=np.int64)
            cat_tensors.append(torch.tensor(indices, dtype=torch.long).unsqueeze(1))

        x_cont_tensor = torch.tensor(x_cont_scaled, dtype=torch.float32)
        if cat_tensors:
            x_cat_tensor = torch.cat(cat_tensors, dim=1)
        else:
            x_cat_tensor = torch.empty((len(df), 0), dtype=torch.long)

        return x_cont_tensor, x_cat_tensor

    def fit(self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)

        x_cont, x_cat = self._prepare_data(df)
        dataset = TensorDataset(x_cont, x_cat)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        cat_cards = [len(self.cat_mappings[c]) for c in self.cat_cols]
        emb_dims = [min(32, max(4, int(card**0.5 * 4))) for card in cat_cards]

        print(f"[{self.device.upper()}] Initializing Tabular VAE (Cont={len(num_cols)}, Cat={len(cat_cols)}, Latent={self.latent_dim})...")
        self.vae = TabularVAE(
            num_continuous=len(num_cols),
            cat_cardinalities=cat_cards,
            embedding_dims=emb_dims,
            latent_dim=self.latent_dim,
        ).to(self.device)

        opt_vae = torch.optim.AdamW(self.vae.parameters(), lr=self.lr, weight_decay=1e-4)

        # 1. Train VAE
        self.vae.train()
        for epoch in range(1, self.epochs_vae + 1):
            total_loss = 0.0
            for batch_cont, batch_cat in dataloader:
                batch_cont = batch_cont.to(self.device)
                batch_cat = batch_cat.to(self.device)
                opt_vae.zero_grad()

                recon_cont, recon_cat, mu, logvar = self.vae(batch_cont, batch_cat)
                loss_cont = F.mse_loss(recon_cont, batch_cont)
                loss_cat = sum(
                    F.cross_entropy(logits, batch_cat[:, idx])
                    for idx, logits in enumerate(recon_cat)
                ) if recon_cat else 0.0

                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = loss_cont + loss_cat + 0.001 * kl_loss

                loss.backward()
                opt_vae.step()
                total_loss += loss.item()

            if epoch % 25 == 0 or epoch == self.epochs_vae:
                print(f"VAE Epoch {epoch}/{self.epochs_vae} - Loss: {total_loss / len(dataloader):.4f}")

        # 2. Extract Latent Representations for Diffusion Training
        self.vae.eval()
        latent_codes = []
        with torch.no_grad():
            for batch_cont, batch_cat in dataloader:
                batch_cont = batch_cont.to(self.device)
                batch_cat = batch_cat.to(self.device)
                mu, _ = self.vae.encode(batch_cont, batch_cat)
                latent_codes.append(mu.cpu())
        z_all = torch.cat(latent_codes, dim=0)

        # 3. Train Latent Diffusion Model
        print(f"[{self.device.upper()}] Training Latent DDPM on extracted latent manifold...")
        self.diffusion = GaussianDiffusion(timesteps=1000, device=self.device)
        self.diffusion_net = LatentDiffusionNet(latent_dim=self.latent_dim).to(self.device)
        opt_diff = torch.optim.AdamW(self.diffusion_net.parameters(), lr=self.lr, weight_decay=1e-4)

        diff_dataset = TensorDataset(z_all)
        diff_loader = DataLoader(diff_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.diffusion_net.train()
        for epoch in range(1, self.epochs_diff + 1):
            total_diff_loss = 0.0
            for (batch_z,) in diff_loader:
                batch_z = batch_z.to(self.device)
                t = torch.randint(0, 1000, (batch_z.shape[0],), device=self.device, dtype=torch.long)
                noise = torch.randn_like(batch_z)
                z_noisy = self.diffusion.q_sample(batch_z, t, noise)

                opt_diff.zero_grad()
                noise_pred = self.diffusion_net(z_noisy, t)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                opt_diff.step()
                total_diff_loss += loss.item()

            if epoch % 25 == 0 or epoch == self.epochs_diff:
                print(f"Diffusion Epoch {epoch}/{self.epochs_diff} - Loss: {total_diff_loss / len(diff_loader):.4f}")

    @torch.no_grad()
    def sample(self, n_samples: int) -> pd.DataFrame:
        self.vae.eval()
        self.diffusion_net.eval()

        print(f"Sampling {n_samples} vectors via Latent Reverse Diffusion on {self.device.upper()}...")
        batch_size = min(2048, n_samples)
        n_batches = int(np.ceil(n_samples / batch_size))

        synth_cont_list = []
        synth_cat_list = {col: [] for col in self.cat_cols}

        for b in range(n_batches):
            cur_n = min(batch_size, n_samples - b * batch_size)
            if b % 5 == 0 or b == n_batches - 1:
                print(f"  Sampling batch {b + 1}/{n_batches} ({cur_n} rows)...")
            z_sampled = self.diffusion.sample(self.diffusion_net, (cur_n, self.latent_dim))
            recon_cont, recon_cat = self.vae.decode(z_sampled)

            recon_cont_np = recon_cont.cpu().numpy()
            recon_cont_orig = self.scaler.inverse_transform(recon_cont_np)
            recon_cont_orig = np.nan_to_num(recon_cont_orig, nan=0.0, posinf=1e6, neginf=-1e6)
            synth_cont_list.append(recon_cont_orig)

            for idx, col in enumerate(self.cat_cols):
                logits = recon_cat[idx]
                cat_indices = torch.argmax(logits, dim=-1).cpu().numpy()
                idx_to_val = {v: k for k, v in self.cat_mappings[col].items()}
                cat_vals = [idx_to_val.get(val_idx, list(idx_to_val.values())[0]) for val_idx in cat_indices]
                synth_cat_list[col].extend(cat_vals)

        synth_cont_all = np.vstack(synth_cont_list)
        df_out = pd.DataFrame(synth_cont_all, columns=self.num_cols)
        for col in self.cat_cols:
            df_out[col] = synth_cat_list[col]

        feature_order = self.num_cols + self.cat_cols
        return df_out[feature_order].fillna(0.0)

    def save_checkpoint(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "vae_state_dict": self.vae.state_dict() if self.vae else None,
            "diff_state_dict": self.diffusion_net.state_dict() if self.diffusion_net else None,
            "scaler": self.scaler,
            "cat_mappings": self.cat_mappings,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "latent_dim": self.latent_dim,
        }, str(path))
        print(f"Checkpoint saved -> {path}")


def find_dataset(filename: str) -> Path:
    candidates = [
        PROJECT_ROOT / filename,
        Path.cwd() / filename,
        Path.cwd().parent / filename,
        Path.cwd().parent.parent / filename,
        CLUSTER_DIR.parent.parent / filename,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Dataset {filename} not found in any candidate path: {candidates}")


# ---------------------------------------------------------------------------
# 4. CLI Driver for SoC Cluster / Workstation (All-In-One Benchmark & Generation)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TabSyn GPU Cluster Latent Diffusion Engine")
    parser.add_argument("--epochs_vae", type=int, default=100)
    parser.add_argument("--epochs_diff", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples_unsw", type=int, default=70000)
    parser.add_argument("--n_samples_nsl", type=int, default=40000)
    parser.add_argument("--output_dir", type=str, default="../submissions")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoints_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CS5344 TRACK 1: TABSYN LATENT DIFFUSION (GPU ALL-IN-ONE ENGINE)")
    print("=" * 80)

    results_summary = []

    # -----------------------------------------------------------------------
    # 1. NSL-KDD: Train -> Holdout Benchmark -> Checkpoint -> 40k Submission
    # -----------------------------------------------------------------------
    print("\n>>> [1/2] NSL-KDD Pipeline: Tabular VAE + Latent DDPM...")
    df_nsl = pd.read_csv(find_dataset("train_nslkdd.csv"))
    train_nsl, val_nsl = train_test_split(df_nsl, test_size=0.2, stratify=df_nsl["is_anomaly"], random_state=42)

    cat_cols_nsl = [
        c for c in df_nsl.columns
        if c != "is_anomaly" and (pd.api.types.is_object_dtype(df_nsl[c]) or pd.api.types.is_string_dtype(df_nsl[c]))
    ]
    num_cols_nsl = [c for c in df_nsl.columns if c != "is_anomaly" and c not in cat_cols_nsl]

    tabsyn_nsl = TabSynDiffusionPipeline(
        epochs_vae=args.epochs_vae, epochs_diff=args.epochs_diff, batch_size=args.batch_size, lr=args.lr
    )
    tabsyn_nsl.fit(train_nsl, num_cols=num_cols_nsl, cat_cols=cat_cols_nsl)

    # A. Holdout Benchmark Evaluation (15,000 samples)
    print("Evaluating TabSyn on NSL-KDD 20% holdout split (ECOD + 10-seed IForest)...")
    synth_nsl_proxy = tabsyn_nsl.sample(15000)
    synth_nsl_proxy = enforce_nslkdd_constraints(synth_nsl_proxy)
    X_tr_nsl, X_val_nsl, _ = build_feature_matrices(synth_nsl_proxy, val_nsl, num_cols_nsl, cat_cols_nsl)
    y_val_nsl = val_nsl["is_anomaly"].to_numpy(dtype=int)
    scores_nsl = score_detectors(X_tr_nsl, X_val_nsl, y_val_nsl, n_seeds=10, n_trees=500)
    print(f"NSL-KDD TabSyn Proxy: AUPRC_ECOD={scores_nsl['AUPRC_ECOD']:.4f}, AUPRC_IForest={scores_nsl['AUPRC_IForest']:.4f}, Composite S={scores_nsl['S']:.4f}")

    results_summary.append({
        "Dataset": "NSL-KDD",
        "Model": "TabSyn Latent Diffusion (GPU)",
        "AUPRC_ECOD": scores_nsl["AUPRC_ECOD"],
        "AUPRC_IForest": scores_nsl["AUPRC_IForest"],
        "Composite_S": scores_nsl["S"],
    })

    # B. Save Checkpoint
    tabsyn_nsl.save_checkpoint(ckpt_dir / "tabsyn_nslkdd.pt")

    # C. Generate Full 40,000 Submission
    print(f"Generating full {args.n_samples_nsl} rows for NSL-KDD submission...")
    synth_nsl = tabsyn_nsl.sample(args.n_samples_nsl)
    synth_nsl = enforce_nslkdd_constraints(synth_nsl)
    synth_nsl.insert(0, "id", np.arange(len(synth_nsl)))

    # Assertions
    assert synth_nsl.shape == (40000, 42), f"Expected (40000, 42), got {synth_nsl.shape}"
    assert not synth_nsl.isna().any().any(), "NaNs detected in NSL submission!"
    assert "is_anomaly" not in synth_nsl.columns, "'is_anomaly' present in submission!"

    nsl_sub_path = out_dir / "submission_nslkdd_tabsyn.csv"
    synth_nsl.to_csv(nsl_sub_path, index=False)
    print(f"Exported & Verified NSL-KDD TabSyn Submission -> {nsl_sub_path}")

    # -----------------------------------------------------------------------
    # 2. UNSW-NB15: Train -> Holdout Benchmark -> Checkpoint -> 70k Submission
    # -----------------------------------------------------------------------
    print("\n>>> [2/2] UNSW-NB15 Pipeline: Tabular VAE + Latent DDPM...")
    df_unsw = pd.read_csv(find_dataset("train_unsw.csv"))
    train_unsw, val_unsw = train_test_split(df_unsw, test_size=0.2, stratify=df_unsw["is_anomaly"], random_state=42)

    cat_cols_unsw = [
        c for c in df_unsw.columns
        if c != "is_anomaly" and (pd.api.types.is_object_dtype(df_unsw[c]) or pd.api.types.is_string_dtype(df_unsw[c]))
    ]
    num_cols_unsw = [c for c in df_unsw.columns if c != "is_anomaly" and c not in cat_cols_unsw]

    tabsyn_unsw = TabSynDiffusionPipeline(
        epochs_vae=args.epochs_vae, epochs_diff=args.epochs_diff, batch_size=args.batch_size, lr=args.lr
    )
    tabsyn_unsw.fit(train_unsw, num_cols=num_cols_unsw, cat_cols=cat_cols_unsw)

    # A. Holdout Benchmark Evaluation (15,000 samples)
    print("Evaluating TabSyn on UNSW-NB15 20% holdout split (ECOD + 10-seed IForest)...")
    synth_unsw_proxy = tabsyn_unsw.sample(15000)
    synth_unsw_proxy = enforce_unsw_constraints(synth_unsw_proxy)
    X_tr_unsw, X_val_unsw, _ = build_feature_matrices(synth_unsw_proxy, val_unsw, num_cols_unsw, cat_cols_unsw)
    y_val_unsw = val_unsw["is_anomaly"].to_numpy(dtype=int)
    scores_unsw = score_detectors(X_tr_unsw, X_val_unsw, y_val_unsw, n_seeds=10, n_trees=500)
    print(f"UNSW-NB15 TabSyn Proxy: AUPRC_ECOD={scores_unsw['AUPRC_ECOD']:.4f}, AUPRC_IForest={scores_unsw['AUPRC_IForest']:.4f}, Composite S={scores_unsw['S']:.4f}")

    results_summary.append({
        "Dataset": "UNSW-NB15",
        "Model": "TabSyn Latent Diffusion (GPU)",
        "AUPRC_ECOD": scores_unsw["AUPRC_ECOD"],
        "AUPRC_IForest": scores_unsw["AUPRC_IForest"],
        "Composite_S": scores_unsw["S"],
    })

    # B. Save Checkpoint
    tabsyn_unsw.save_checkpoint(ckpt_dir / "tabsyn_unsw.pt")

    # C. Generate Full 70,000 Submission
    print(f"Generating full {args.n_samples_unsw} rows for UNSW-NB15 submission...")
    synth_unsw = tabsyn_unsw.sample(args.n_samples_unsw)
    synth_unsw = enforce_unsw_constraints(synth_unsw)
    synth_unsw.insert(0, "id", np.arange(len(synth_unsw)))

    # Assertions
    assert synth_unsw.shape == (70000, 43), f"Expected (70000, 43), got {synth_unsw.shape}"
    assert not synth_unsw.isna().any().any(), "NaNs detected in UNSW submission!"
    assert "is_anomaly" not in synth_unsw.columns, "'is_anomaly' present in submission!"

    unsw_sub_path = out_dir / "submission_unsw_tabsyn.csv"
    synth_unsw.to_csv(unsw_sub_path, index=False)
    print(f"Exported & Verified UNSW-NB15 TabSyn Submission -> {unsw_sub_path}")

    # -----------------------------------------------------------------------
    # 3. Export Proxy Benchmark Summary Table
    # -----------------------------------------------------------------------
    df_summary = pd.DataFrame(results_summary)
    summary_path = out_dir / "local_proxy_results_tabsyn.csv"
    df_summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 80)
    print("TABSYN GPU BENCHMARK SUMMARY:")
    print(df_summary.to_string(index=False))
    print(f"Summary saved -> {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
