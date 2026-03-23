#!/bin/bash
# Install all dependencies including training extras
set -e
cd "$(dirname "$0")/.."
uv sync --extra train
echo "Training setup complete."
