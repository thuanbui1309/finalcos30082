#!/bin/bash
# Launch Jupyter Notebook for training (uses GPU card #2 by default)
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
uv run --extra train jupyter notebook --notebook-dir=notebooks "$@"
