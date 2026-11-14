#!/usr/bin/env bash
# Clone tau2-bench at the pinned commit, install the project, and download the Qwen3 MLX models.
set -euo pipefail
cd "$(dirname "$0")/.."

TAU2_COMMIT=c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4
if [ ! -d vendor/tau2-bench ]; then
  git clone https://github.com/sierra-research/tau2-bench.git vendor/tau2-bench
fi
git -C vendor/tau2-bench fetch --depth 1 origin "$TAU2_COMMIT"
git -C vendor/tau2-bench checkout --detach "$TAU2_COMMIT"

uv venv --python 3.12 .venv 2>/dev/null || true
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python -m plvr.models download
