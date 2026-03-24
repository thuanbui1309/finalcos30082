#!/bin/bash
# Run all training notebooks sequentially and save outputs in-place.
# Usage:
#   ./scripts/train_all.sh           # foreground
#   ./scripts/train_all.sh --bg      # background (survives SSH disconnect)
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

if [[ "$1" == "--bg" ]]; then
    nohup bash "$(realpath "$0")" --run > logs/train_all.log 2>&1 &
    echo "Training running in background (PID $!)"
    echo "Follow progress: tail -f logs/train_all.log"
    exit 0
fi

# --run or foreground: execute notebooks
echo "=== Starting training pipeline (GPU $CUDA_VISIBLE_DEVICES) ==="

declare -A NB_WEIGHTS=(
    ["notebooks/02_face_classification.ipynb"]="weights/face_classification.pth"
    ["notebooks/03_face_metric_learning.ipynb"]="weights/face_metric_learning.pth"
)

for nb in \
    "notebooks/02_face_classification.ipynb" \
    "notebooks/03_face_metric_learning.ipynb"
do
    weight="${NB_WEIGHTS[$nb]}"
    if [[ -f "$weight" ]]; then
        echo ""
        echo ">>> Skipping $nb (weights already exist: $weight)"
        continue
    fi

    echo ""
    echo ">>> Running $nb ..."
    uv run --extra train jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=86400 \
        "$nb"
    echo ">>> Done: $nb"
done

echo ""
echo "=== All notebooks finished. Weights saved to weights/ ==="
