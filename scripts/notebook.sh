#!/bin/bash
# Launch Jupyter Notebook for training (uses GPU card #2 by default)
# Usage:
#   ./scripts/notebook.sh          # foreground (normal)
#   ./scripts/notebook.sh --bg     # background via nohup (survives SSH disconnect)
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [[ "$1" == "--bg" ]]; then
    nohup uv run --extra train jupyter notebook \
        --notebook-dir=notebooks \
        --no-browser \
        --port=8889 \
        > logs/jupyter.log 2>&1 &
    echo "Jupyter running in background (PID $!)"
    echo "Logs: logs/jupyter.log"
    echo ""
    echo "Get token: grep 'token=' logs/jupyter.log | tail -1"
else
    uv run --extra train jupyter notebook --notebook-dir=notebooks "$@"
fi
