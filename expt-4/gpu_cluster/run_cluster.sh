#!/usr/bin/env bash
# ==============================================================================
# Slurm Job Script: End-to-End One-Shot Generative Benchmark & Submission Engine
# Group: CS5344 Group 42 (Aayush Majumdar & Vadiraj Rao)
# Target Datasets: UNSW-NB15 & NSL-KDD
# Hardware Target: 1x GPU (NVIDIA A100 / V100 / RTX 3090 / L40S) + 8 CPUs + 32GB RAM
# ==============================================================================

#SBATCH --job-name=cs5344_expt4_all
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=aayush_majumdar@u.nus.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

notify_exit() {
    local rc=$?
    # Ensure logs folder exists and archive slurm logs
    [ ! -d "logs" ] && mkdir -p logs 2>/dev/null || true
    if [ -n "$SLURM_JOB_ID" ]; then
        cp "slurm_${SLURM_JOB_ID}.out" logs/ 2>/dev/null || true
        cp "slurm_${SLURM_JOB_ID}.err" logs/ 2>/dev/null || true
    fi
    if [ $rc -eq 0 ]; then
        echo -e "\n[SUCCESS] Slurm Job ${SLURM_JOB_ID:-LOCAL} finished successfully with exit code 0 at $(date)."
    else
        echo -e "\n[FAILURE] Slurm Job ${SLURM_JOB_ID:-LOCAL} failed with exit code $rc at $(date)."
    fi
}
trap notify_exit EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "CS5344 EXPERIMENT 4: ALL-IN-ONE BENCHMARK & SUBMISSION ENGINE"
echo "================================================================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $SCRIPT_DIR"

# Defensive, idempotent directory creation (never trips set -e)
for target_dir in "logs" "checkpoints" "../submissions" "../../submissions"; do
    if [ ! -d "$target_dir" ]; then
        mkdir -p "$target_dir" 2>/dev/null || true
    fi
done

# 1. Environment Detection & Dependency Setup
echo -e "\n>>> [Stage 0/4] Environment & Hardware Inspection..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Notice: nvidia-smi not in PATH, checking torch CUDA availability..."
fi

if command -v conda &> /dev/null; then
    echo "Activating conda environment 'cs5344' if available..."
    eval "$(conda shell.bash hook)"
    conda activate cs5344 || true
elif [ -d "../../.venv" ]; then
    echo "Activating project venv..."
    source ../../.venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating local venv..."
    source venv/bin/activate
fi

echo "Verifying Python GPU dependencies..."
pip install -r requirements_gpu.txt --quiet

# 2. Stage 1: Fast Analytical Track (CPB v2 + Contrastive DVS-Copula v2)
echo -e "\n>>> [Stage 1/4] Running Track 2: Contrastive DVS-Copula + CPB v2 (CPU)..."
python ../src/benchmark_expt4.py

# 3. Stage 2: Deep Generative Track (TabSyn Latent DDPM on GPU)
echo -e "\n>>> [Stage 2/4] Running Track 1: TabSyn Latent Diffusion (GPU)..."
python train_tabsyn_diffusion.py \
    --epochs_vae 100 \
    --epochs_diff 150 \
    --batch_size 512 \
    --lr 1e-3 \
    --n_samples_unsw 70000 \
    --n_samples_nsl 40000 \
    --output_dir ../submissions \
    --checkpoints_dir checkpoints

# 4. Stage 3: Master Benchmark Compilation & Integrity Verification
echo -e "\n>>> [Stage 3/4] Compiling Master Benchmark Report & Verifying Submissions..."
python - << 'EOF'
import os
import shutil
import pandas as pd
from pathlib import Path

submissions_dir = Path("../submissions").resolve()
root_submissions_dir = Path("../../submissions").resolve()
submissions_dir.mkdir(parents=True, exist_ok=True)
root_submissions_dir.mkdir(parents=True, exist_ok=True)

# 1. Load and merge benchmark results
p_track2 = submissions_dir / "local_proxy_results_expt4.csv"
p_track1 = submissions_dir / "local_proxy_results_tabsyn.csv"

dfs = []
if p_track2.exists():
    dfs.append(pd.read_csv(p_track2))
if p_track1.exists():
    dfs.append(pd.read_csv(p_track1))

if dfs:
    master_df = pd.concat(dfs, ignore_index=True)
    master_path = submissions_dir / "master_benchmark_summary.csv"
    master_df.to_csv(master_path, index=False)
    print("\n" + "=" * 80)
    print("MASTER BENCHMARK COMPARISON TABLE:")
    print("=" * 80)
    print(master_df.to_string(index=False))
    print("=" * 80)

# 2. Comprehensive Integrity Audit for All Submissions
files_to_check = [
    ("NSL-KDD Track 2", submissions_dir / "submission_nslkdd.csv", 40000, 42),
    ("UNSW-NB15 Track 2", submissions_dir / "submission_unsw.csv", 70000, 43),
    ("NSL-KDD TabSyn", submissions_dir / "submission_nslkdd_tabsyn.csv", 40000, 42),
    ("UNSW-NB15 TabSyn", submissions_dir / "submission_unsw_tabsyn.csv", 70000, 43),
]

print("\nRunning submission integrity checks...")
for label, fpath, exp_rows, exp_cols in files_to_check:
    if not fpath.exists():
        print(f"[-] {label}: File not found ({fpath.name})")
        continue
    df = pd.read_csv(fpath)
    assert df.shape == (exp_rows, exp_cols), f"{label} shape mismatch: {df.shape} vs {(exp_rows, exp_cols)}"
    assert df.columns[0] == "id", f"{label} missing 'id' column at pos 0"
    assert "is_anomaly" not in df.columns, f"{label} contains 'is_anomaly'"
    assert not df.isna().any().any(), f"{label} contains NaNs"
    print(f"[+] {label}: PASS ({df.shape[0]} rows, {df.shape[1]} cols, 0 NaNs, id col 0..{df.shape[0]-1})")

# 3. Synchronize Root Submissions
for fname in ["submission_nslkdd.csv", "submission_unsw.csv", "submission_nslkdd_tabsyn.csv", "submission_unsw_tabsyn.csv"]:
    src = submissions_dir / fname
    if src.exists():
        shutil.copy(src, root_submissions_dir / fname)
        print(f"[+] Synchronized -> ../../submissions/{fname}")

EOF

echo -e "\n================================================================================"
echo "EXPERIMENT 4 JOB FINISHED SUCCESSFULLY AT $(date)"
echo "All benchmarks compiled and all submission CSVs verified in ../submissions/"
echo "================================================================================"
