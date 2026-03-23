#!/bin/bash
# Launch Jupyter Notebook for training
set -e
cd "$(dirname "$0")/.."
uv run --extra train jupyter notebook --notebook-dir=notebooks "$@"
