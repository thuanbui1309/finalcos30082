#!/bin/bash
# Launch Streamlit GUI
set -e
cd "$(dirname "$0")/.."
uv run streamlit run src/ui/app.py "$@"
