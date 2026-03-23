#!/bin/bash
# Install dependencies (inference only)
set -e
cd "$(dirname "$0")/.."
uv sync
echo "Setup complete."
