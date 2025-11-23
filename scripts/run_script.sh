#!/bin/bash
# Generic wrapper script to run any Python script with proper GPU setup and environment

if [ $# -eq 0 ]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=<gpu_ids> $0 <script_path> [additional args...]"
    echo "Example: CUDA_VISIBLE_DEVICES=0 $0 ./scripts/evaluate_baseline.py"
    echo "Example: CUDA_VISIBLE_DEVICES=0,1 $0 ./scripts/calculate_metrics.py"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# CUDA_VISIBLE_DEVICES should be set by the caller
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Warning: CUDA_VISIBLE_DEVICES not set. Using all available GPUs."
fi
export LD_PRELOAD="$PROJECT_DIR/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12"

# Add project root to PYTHONPATH so scripts.* and tests.* imports work
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

source "$PROJECT_DIR/.venv/bin/activate"

with-proxy python "$@"
