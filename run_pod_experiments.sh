#!/bin/bash
################################################################################
# Run All 10 TMLR Experiments on Pod
################################################################################
#
# Usage on RunPod/Lambda Labs:
#   bash run_pod_experiments.sh
#
# This script will:
# 1. Check GPU availability
# 2. Install dependencies
# 3. Run all 10 experiments (~33 hours on A5000)
# 4. Save results to results/final_experiments/
#
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "TMLR Experiment Suite - Pod Runner"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Check if running in tmux (recommended for long runs)
if [ -z "$TMUX" ]; then
    echo "WARNING: Not running in tmux. If SSH disconnects, experiments will stop!"
    echo "Recommended: Run 'tmux' first, then run this script inside tmux."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
else
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

# Check Python
echo "Checking Python..."
python3 --version
echo ""

# Install dependencies
echo "Installing dependencies..."
cd "$(dirname "$0")"

if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found"
    exit 1
fi

pip install -q -r requirements.txt
echo "Dependencies installed."
echo ""

# Verify PyTorch CUDA
echo "Verifying PyTorch CUDA..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"
echo ""

# Change to code directory
cd code

# Run experiments
echo "================================================================================"
echo "Starting all 10 experiments..."
echo "Estimated time: 33 hours on A5000 GPU"
echo "Estimated cost: \$9.24 on RunPod A5000 @ \$0.28/hour"
echo "================================================================================"
echo ""

python3 run_all_10_experiments.py

# Check results
echo ""
echo "================================================================================"
echo "Experiments complete!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: results/final_experiments/"
echo ""
echo "Check results with:"
echo "  ls -lh results/final_experiments/"
echo "  cat results/final_experiments/experiment_summary.json | python3 -m json.tool"
echo ""
echo "================================================================================"
