#!/bin/bash
# Wrapper script to run evaluate_baseline.py with proper LD_PRELOAD for vllm/cublas

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

export LD_PRELOAD="$PROJECT_DIR/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12"

source "$PROJECT_DIR/.venv/bin/activate"
with-proxy python "$SCRIPT_DIR/evaluate_baseline.py" "$@"
