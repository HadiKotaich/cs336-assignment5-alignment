#!/bin/bash
# Wrapper script to run evaluate_baseline.py with proper LD_PRELOAD for vllm/cublas

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
with-proxy python "$SCRIPT_DIR/evaluate_baseline.py" "$@"
