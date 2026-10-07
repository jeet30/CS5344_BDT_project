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
# 0. Identify Scratch Storage (68TB Quota-Free Pool) & Setup Isolation
# ------------------------------------------------------------------------------
echo -e "\n[Step 0/4] Resolving Scratch Storage & Setting Quota Safeguards..."

# Detect user-owned scratch directory on NUS SoC cluster
FIRST_CHAR="${USER:0:1}"
SCRATCH_BASE=""

if [ -d "/mnt/scratch/$FIRST_CHAR/$USER" ] && [ -w "/mnt/scratch/$FIRST_CHAR/$USER" ]; then
    SCRATCH_BASE="/mnt/scratch/$FIRST_CHAR/$USER"
elif [ -d "/mnt/scratch/a/$USER" ] && [ -w "/mnt/scratch/a/$USER" ]; then
    SCRATCH_BASE="/mnt/scratch/a/$USER"
elif [ -d "/mnt/scratch/$USER" ] && [ -w "/mnt/scratch/$USER" ]; then
    SCRATCH_BASE="/mnt/scratch/$USER"
elif [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    SCRATCH_BASE="/scratch/$USER"
else
    SCRATCH_BASE="$HOME/.local_scratch"
    mkdir -p "$SCRATCH_BASE" 2>/dev/null || true
fi

echo "[+] Target Scratch Base: $SCRATCH_BASE (NUS SoC 68TB Shared Storage Pool)"

# Configure dedicated scratch temp, cache, and venv directories
SCRATCH_TMP="$SCRATCH_BASE/tmp"
SCRATCH_CACHE="$SCRATCH_BASE/pip_cache"
TARGET_VENV_DIR="$SCRATCH_BASE/cs5344_venv"

mkdir -p "$SCRATCH_TMP" "$SCRATCH_CACHE" "$TARGET_VENV_DIR"
chmod 700 "$SCRATCH_TMP" "$SCRATCH_CACHE" "$TARGET_VENV_DIR" 2>/dev/null || true
rm -rf "$SCRATCH_TMP"/* "$SCRATCH_BASE/uv_cache" 2>/dev/null || true

# Purge any lingering pip caches or old venvs in $HOME to immediately free quota
echo "Purging legacy cache in \$HOME to prevent quota leaks..."
rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/triton ~/.cache/huggingface 2>/dev/null || true
rm -rf ~/.local/tmp ~/.local/share/pip 2>/dev/null || true
if [ -d "$HOME/.venvs/cs5344" ] && [ ! -L "$HOME/.venvs/cs5344" ]; then
    echo "Removing unlinked ~/.venvs/cs5344 from \$HOME (reclaiming ~14MB quota)..."
    rm -rf "$HOME/.venvs/cs5344" 2>/dev/null || true
fi
mkdir -p "$HOME/.venvs"

# Export Scratch temp & cache paths
export TMPDIR="$SCRATCH_TMP"
export TEMP="$SCRATCH_TMP"
export TMP="$SCRATCH_TMP"
export PIP_CACHE_DIR="$SCRATCH_CACHE"

# Quick write capability test (writes 20MB directly to scratch partition)
TEST_BIN="$SCRATCH_TMP/write_test_$$"
if dd if=/dev/zero of="$TEST_BIN" bs=1M count=20 2>/dev/null; then
    rm -f "$TEST_BIN"
    echo "[+] Scratch write verification PASSED: 20MB test file written and cleaned."
else
    echo "[!] Warning: 20MB test write to $SCRATCH_TMP failed. Continuing..."
fi

# Display user quota
if command -v myquota &> /dev/null; then
    myquota || true
elif command -v quota &> /dev/null; then
    quota -s 2>/dev/null || true
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

# Re-enforce scratch environment variables (in case .bashrc reset them!)
export TMPDIR="$SCRATCH_TMP"
export TEMP="$SCRATCH_TMP"
export TMP="$SCRATCH_TMP"
export PIP_CACHE_DIR="$SCRATCH_CACHE"

# ------------------------------------------------------------------------------
# 2. Virtual Environment Creation in Scratch
# ------------------------------------------------------------------------------
echo -e "\n[Step 2/4] Setting up Virtual Environment in Scratch..."
echo "[+] Virtualenv Location: $TARGET_VENV_DIR"

if [ ! -f "$TARGET_VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment with system site packages at $TARGET_VENV_DIR..."
    python3 -m venv --system-site-packages "$TARGET_VENV_DIR"
else
    echo "Reusing existing virtual environment at $TARGET_VENV_DIR..."
fi

source "$TARGET_VENV_DIR/bin/activate"

# Symlink $HOME/.venvs/cs5344 to scratch venv for standard lookups
ln -sfn "$TARGET_VENV_DIR" "$HOME/.venvs/cs5344" 2>/dev/null || true

# Re-assert environment inside active venv
export TMPDIR="$SCRATCH_TMP"
export TEMP="$SCRATCH_TMP"
export TMP="$SCRATCH_TMP"
export PIP_CACHE_DIR="$SCRATCH_CACHE"

# ------------------------------------------------------------------------------
# 3. Installing Dependencies Using Scratch Storage & Cache
# ------------------------------------------------------------------------------
echo -e "\n[Step 3/4] Installing Required Packages (Scratch Cache Mode)..."

# Upgrade pip using scratch cache and temp dir
python3 -m pip install \
    --cache-dir "$SCRATCH_CACHE" \
    --default-timeout=1000 \
    --no-warn-script-location \
    --upgrade pip

# Check if PyTorch is already available and functional
if python3 -c "import torch; assert hasattr(torch, '__version__'); print('PyTorch is already functional:', torch.__version__)" 2>/dev/null; then
    echo "[+] PyTorch is already functional. Skipping download to save bandwidth and storage."
else
    echo "Installing PyTorch (CUDA 12.1) using 68TB scratch pool..."
    python3 -m pip install \
        --cache-dir "$SCRATCH_CACHE" \
        --default-timeout=1000 \
        --no-warn-script-location \
        torch --index-url https://download.pytorch.org/whl/cu121
fi

echo "Installing project dependencies from requirements_gpu.txt..."
python3 -m pip install \
    --cache-dir "$SCRATCH_CACHE" \
    --default-timeout=1000 \
    --no-warn-script-location \
    -r requirements_gpu.txt

# Final cleanup of any lingering temporary files
rm -rf ~/.cache/pip ~/.cache/torch "$SCRATCH_TMP"/* 2>/dev/null || true

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
