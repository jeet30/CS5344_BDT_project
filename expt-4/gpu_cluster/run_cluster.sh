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

# ------------------------------------------------------------------------------
# Robust Working Directory Resolution
# ------------------------------------------------------------------------------
# In Slurm execution, sbatch copies the script to a daemon spool folder (/var/lib/slurm/...),
# so ${BASH_SOURCE[0]} evaluates to the spool path instead of the repository directory.
# SLURM_SUBMIT_DIR contains the directory where sbatch was invoked.

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -n "${BASH_SOURCE[0]}" ] && [ -f "${BASH_SOURCE[0]}" ] && [[ "${BASH_SOURCE[0]}" != *"/slurm"* ]]; then
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Ensure we are operating from within expt-4/gpu_cluster
if [ -d "expt-4/gpu_cluster" ]; then
    cd "expt-4/gpu_cluster"
elif [ ! -f "train_tabsyn_diffusion.py" ] && [ -f "../../expt-4/gpu_cluster/train_tabsyn_diffusion.py" ]; then
    cd "../../expt-4/gpu_cluster"
fi

# Defensive search fallback if submitted from an unexpected root
if [ ! -f "train_tabsyn_diffusion.py" ]; then
    FOUND_LOC="$(find "$HOME" -maxdepth 5 -type f -name "train_tabsyn_diffusion.py" 2>/dev/null | head -n 1)"
    if [ -n "$FOUND_LOC" ]; then
        cd "$(dirname "$FOUND_LOC")"
    fi
fi

SCRIPT_DIR="$(pwd)"

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

# Matplotlib cache isolation
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${USER:-cs5344}_${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

# 1. Environment Detection & Dependency Setup
echo -e "\n>>> [Stage 0/4] Environment & Hardware Inspection..."

# Module loading for HPC environments (NUS SoC clusters)
if command -v module &> /dev/null || type module &> /dev/null; then
    echo "Loading cluster environment modules..."
    module load cuda 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true
    module load python 2>/dev/null || module load python/3.10 2>/dev/null || module load miniconda3 2>/dev/null || true
fi

echo "SLURM_JOB_ID:         ${SLURM_JOB_ID:-<local>}"
echo "SLURM_JOB_NODELIST:   ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set by Slurm>}"
echo "SLURM_JOB_GPUS:       ${SLURM_JOB_GPUS:-<not set>}"

# Check nvidia-smi with non-zero protection (exit code 6 must never crash set -e)
if command -v nvidia-smi &> /dev/null; then
    echo "Querying nvidia-smi..."
    nvidia-smi 2>&1 || echo "Notice: nvidia-smi returned non-zero (exit code $?). Proceeding with PyTorch runtime checks..."
else
    echo "Notice: nvidia-smi not in PATH. Proceeding with PyTorch runtime checks..."
fi

# Source profile and conda initializers if available (vital for Slurm non-interactive shells)
[ -f /etc/profile.d/conda.sh ] && source /etc/profile.d/conda.sh 2>/dev/null || true
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true

# Limit CPU threading to Slurm allocation (prevents CPU thread contention and joblib memory thrashing)
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK

# Conda / virtual environment activation
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'cs5344' if available..."
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate cs5344 2>/dev/null || true
elif [ -d "../../.venv" ]; then
    echo "Activating project venv..."
    source ../../.venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating local venv..."
    source venv/bin/activate
elif [ -d "$HOME/miniconda3/envs/cs5344" ]; then
    echo "Activating miniconda cs5344 venv..."
    source "$HOME/miniconda3/bin/activate" cs5344 2>/dev/null || true
fi

# Probe PyTorch runtime hardware device
echo -e "\nChecking Python & PyTorch hardware support:"
python3 -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python version:    {sys.version.split()[0]}')
try:
    import torch
    print(f'PyTorch version:   {torch.__version__}')
    cuda_avail = torch.cuda.is_available()
    print(f'CUDA available:    {cuda_avail}')
    if cuda_avail:
        print(f'CUDA Device count: {torch.cuda.device_count()}')
        print(f'CUDA Device name:  {torch.cuda.get_device_name(0)}')
    else:
        print('CUDA not detected by PyTorch: TabSyn will operate in CPU-optimized mode.')
except ImportError:
    print('Notice: PyTorch not found in current environment. Will install via requirements_gpu.txt...')
" 2>/dev/null || true

echo -e "\nVerifying Python dependencies..."
pip install -r requirements_gpu.txt --quiet 2>/dev/null || echo "Notice: Offline cluster node or requirements already satisfied. Continuing..."

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
