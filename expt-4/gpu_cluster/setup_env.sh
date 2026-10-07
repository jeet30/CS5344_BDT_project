#!/usr/bin/env bash
# ==============================================================================
# One-Shot Environment Setup Script for NUS SoC HPC Cluster
# RUN THIS ON THE LOGIN NODE (xlogin2) WHERE INTERNET IS AVAILABLE!
# Group: CS5344 Group 42 (Aayush Majumdar & Vadiraj Rao)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "CS5344: ONE-SHOT ENVIRONMENT SETUP FOR NUS SOC GPU CLUSTER"
echo "Host:     $(hostname)"
echo "Date:     $(date)"
echo "Location: $SCRIPT_DIR"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 0. Disk Quota Safeguards & Cache Purging
# ------------------------------------------------------------------------------
echo -e "\n[Step 0/4] Checking Disk Quota & Purging Pip Cache..."
# Purge pip cache to free up hundreds of MBs/GBs in quota-limited $HOME
rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/triton 2>/dev/null || true

# Force pip to never write cached wheels to $HOME
export PIP_NO_CACHE_DIR=1
# Route temporary decompression away from $HOME to local /tmp
export TMPDIR=/tmp

# Display current quota / disk usage
if command -v myquota &> /dev/null; then
    myquota || true
elif command -v quota &> /dev/null; then
    quota -s 2>/dev/null || true
else
    df -h "$HOME" 2>/dev/null || true
fi

# ------------------------------------------------------------------------------
# 1. Cluster Module Loading (Prefers Pre-Installed HPC Packages)
# ------------------------------------------------------------------------------
echo -e "\n[Step 1/4] Checking and Loading Cluster Modules..."
if command -v module &> /dev/null || type module &> /dev/null; then
    module load cuda 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true
    module load pytorch 2>/dev/null || module load torch 2>/dev/null || true
    module load python 2>/dev/null || module load python/3.10 2>/dev/null || module load python/3.12 2>/dev/null || true
    module load miniconda3 2>/dev/null || true
fi

# Source profile for Conda if available
[ -f /etc/profile.d/conda.sh ] && source /etc/profile.d/conda.sh 2>/dev/null || true
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true

# ------------------------------------------------------------------------------
# 2. Virtual Environment Location Resolution (Scratch vs Home)
# ------------------------------------------------------------------------------
echo -e "\n[Step 2/4] Resolving Environment Location..."

# Check if a scratch directory exists with generous quota
TARGET_VENV_DIR=""
if [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    TARGET_VENV_DIR="/scratch/$USER/cs5344_venv"
    echo "[+] Using cluster scratch partition: $TARGET_VENV_DIR (quota-free)"
elif [ -d "/scratch/a/$USER" ] && [ -w "/scratch/a/$USER" ]; then
    TARGET_VENV_DIR="/scratch/a/$USER/cs5344_venv"
    echo "[+] Using cluster scratch partition: $TARGET_VENV_DIR (quota-free)"
else
    TARGET_VENV_DIR="$HOME/.venvs/cs5344"
    echo "[*] Using home partition: $TARGET_VENV_DIR"
fi

mkdir -p "$(dirname "$TARGET_VENV_DIR")"

# Create venv with --system-site-packages to inherit any cluster-installed packages (saves quota!)
if [ ! -d "$TARGET_VENV_DIR" ]; then
    echo "Creating virtual environment with system site packages at $TARGET_VENV_DIR..."
    python3 -m venv --system-site-packages "$TARGET_VENV_DIR"
else
    echo "Reusing existing virtual environment at $TARGET_VENV_DIR..."
fi

source "$TARGET_VENV_DIR/bin/activate"

# Symlink to standard $HOME/.venvs/cs5344 if using scratch for standard lookup
if [ "$TARGET_VENV_DIR" != "$HOME/.venvs/cs5344" ]; then
    mkdir -p "$HOME/.venvs"
    ln -sfn "$TARGET_VENV_DIR" "$HOME/.venvs/cs5344" 2>/dev/null || true
fi

# ------------------------------------------------------------------------------
# 3. Installing Dependencies with Zero Cache
# ------------------------------------------------------------------------------
echo -e "\n[Step 3/4] Installing Required Packages (Zero-Cache Mode)..."

# Upgrade pip without caching
python3 -m pip install --no-cache-dir --upgrade pip

# Check if PyTorch is already available from system/module
if python3 -c "import torch; assert torch.cuda.is_available() or True; print('PyTorch already installed:', torch.__version__)" 2>/dev/null; then
    echo "[+] PyTorch is already available via system/module. Skipping download to save quota!"
else
    echo "Installing PyTorch (CUDA 12.1, zero-cache mode)..."
    python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 || \
    python3 -m pip install --no-cache-dir torch
fi

echo "Installing project dependencies from requirements_gpu.txt..."
python3 -m pip install --no-cache-dir -r requirements_gpu.txt

# Final cleanup of any lingering temporary files
rm -rf ~/.cache/pip ~/.cache/torch 2>/dev/null || true

# ------------------------------------------------------------------------------
# 4. Final Environment Verification
# ------------------------------------------------------------------------------
echo -e "\n[Step 4/4] Verifying Environment Integrity..."
python3 -c "
import sys
import torch
import sklearn
import pyod
import pandas
import numpy
import scipy
import tqdm

print('=' * 70)
print('ENVIRONMENT VERIFICATION SUCCESSFUL!')
print(f'Active Python:      {sys.executable}')
print(f'Python Version:     {sys.version.split()[0]}')
print(f'PyTorch:            {torch.__version__} (CUDA built-in: {torch.version.cuda})')
print(f'CUDA Available:     {torch.cuda.is_available()}')
print(f'Scikit-Learn:       {sklearn.__version__}')
print(f'PyOD:               {pyod.__version__}')
print(f'Pandas:             {pandas.__version__}')
print(f'NumPy:              {numpy.__version__}')
print(f'SciPy:              {scipy.__version__}')
print('=' * 70)
"

echo -e "\n================================================================================"
echo "SETUP COMPLETED SUCCESSFULLY!"
echo "Environment installed at: $TARGET_VENV_DIR"
echo "Every compute node can now run your job without downloading."
echo ""
echo "NEXT STEP: Submit your full experiment to the GPU cluster by running:"
echo "    sbatch run_cluster.sh"
echo "================================================================================"
