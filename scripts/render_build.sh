#!/usr/bin/env bash
# Render build step for both `emotion-api` and `emotion-worker`.
#
# Render invokes this from the repo root as ./Project/scripts/render_build.sh,
# but paths here resolve from the script's own location so it behaves the same
# when run by hand from anywhere.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"

echo "render_build: installing backend dependencies"
# backend/requirements.txt already pins torch, transformers and shap to the same
# versions as ml/requirements.txt, so the training-time artifact loads
# identically here. Installing ml/requirements.txt too would pull datasets and
# accelerate, which neither the API nor the worker imports.
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r "$project_dir/backend/requirements.txt"

echo "render_build: fetching model artifact"
"$script_dir/fetch_model.sh"

echo "render_build: done"
