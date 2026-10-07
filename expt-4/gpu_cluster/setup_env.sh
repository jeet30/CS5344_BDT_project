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

# 1. Load system environment modules if present
if command -v module &> /dev/null || type module &> /dev/null; then
    echo "Loading cluster environment modules..."
    module load cuda 2>/dev/null || module load cuda/12.1 2>/dev/null || true
    module load python 2>/dev/null || module load python/3.10 2>/dev/null || true
    module load miniconda3 2>/dev/null || true
fi

# Source common profile locations for conda
[ -f /etc/profile.d/conda.sh ] && source /etc/profile.d/conda.sh 2>/dev/null || true
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true

USE_CONDA=false
if [ "$1" != "--venv" ] && command -v conda &> /dev/null; then
    USE_CONDA=true
fi

if [ "$USE_CONDA" = true ]; then
    echo -e "\n[Step 1/3] Setting up Conda Environment 'cs5344'..."
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if ! conda info --envs | grep -E '^cs5344\s' > /dev/null 2>&1; then
        echo "Creating new conda environment 'cs5344' (Python 3.10)..."
        conda create -y -n cs5344 python=3.10
    else
        echo "Conda environment 'cs5344' already exists. Reusing..."
    fi
    conda activate cs5344
    PYTHON_EXEC="$(which python3)"
else
    VENV_DIR="$HOME/.venvs/cs5344"
    echo -e "\n[Step 1/3] Setting up Python Virtualenv in $VENV_DIR..."
    mkdir -p "$HOME/.venvs"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    else
        echo "Virtual environment already exists in $VENV_DIR. Reusing..."
    fi
    source "$VENV_DIR/bin/activate"
    PYTHON_EXEC="$(which python3)"
fi

echo -e "\n[Step 2/3] Installing Dependencies via Pip..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (compatible with A100 / H100 on NUS SoC)
echo "Installing PyTorch (CUDA 12.1)..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || python3 -m pip install torch torchvision

echo "Installing project dependencies from requirements_gpu.txt..."
python3 -m pip install -r requirements_gpu.txt

echo -e "\n[Step 3/3] Verifying Environment Integrity..."
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
echo "Your Python environment is now installed on the cluster shared storage ($HOME)."
echo "Every compute node (xgph*, xgpg*, etc.) can now run your job without downloading."
echo ""
echo "NEXT STEP: Submit your full experiment to the GPU cluster by running:"
echo "    sbatch run_cluster.sh"
echo "================================================================================"
