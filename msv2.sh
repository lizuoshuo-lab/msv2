#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MSV2_ROOT="${MSV2_ROOT:-$HOME/msv2}"
if [[ -f "$MSV2_ROOT/runtime.env" ]]; then
  # Created by this deployment, outside the Git tree.
  declare -A supplied_values=()
  for variable in MSV2_PYTHON MSV2_SOURCE_DIR MSV2_SWINIR_SOURCE MSV2_SWINIR_WEIGHT MSV2_EXTENSIONS_DIR MSV2_HEAD_DETECTOR_WEIGHT MSV2_HEAD_PARSER_SOURCE MSV2_HEAD_PARSER_WEIGHT; do
    if [[ -v "$variable" ]]; then supplied_values["$variable"]="${!variable}"; fi
  done
  source "$MSV2_ROOT/runtime.env"
  for variable in "${!supplied_values[@]}"; do export "$variable=${supplied_values[$variable]}"; done
fi
MSV2_PYTHON="${MSV2_PYTHON:-$MSV2_ROOT/.venv/bin/python}"
[[ -x "$MSV2_PYTHON" ]] || { echo 'MSV2 environment missing. Run bash deploy.sh first.' >&2; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$APP_DIR${MSV2_EXTENSIONS_DIR:+:$MSV2_EXTENSIONS_DIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ulimit -n 16384 2>/dev/null || true
exec "$MSV2_PYTHON" -s -m msv2 "$@"
