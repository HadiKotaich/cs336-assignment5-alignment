#!/bin/bash
# Generic wrapper script to run any Python script with proper GPU setup and environment

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_path> [additional args...]"
    echo "Example: $0 ./scripts/evaluate_baseline.py"
    echo "Example: $0 ./scripts/calculate_metrics.py"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Select GPU with most free memory
GPU_ID=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
    awk '{print NR-1 " " $0}' | \
    sort -k2 -nr | \
    head -n1 | \
    awk '{print $1}')

echo "Memory free on all GPUs:"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
echo "Using GPU $GPU_ID"

export CUDA_VISIBLE_DEVICES=$GPU_ID
export LD_PRELOAD="$PROJECT_DIR/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12"

source "$PROJECT_DIR/.venv/bin/activate"
with-proxy python "$@"
