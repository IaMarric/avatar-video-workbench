#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/avatar-video-workbench-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x ".venv/bin/avw" ]]; then
  "$PYTHON_BIN" -m venv .venv
  .venv/bin/pip install -e ".[test]"
fi

.venv/bin/avw smoke-demo --out-dir "$OUT_DIR" --force
.venv/bin/avw preflight-vertex --job-yaml "$OUT_DIR/compiled/jobs/vertex-custom-job.yaml"
.venv/bin/avw scan-publication .

printf '\nSynthetic smoke demo complete. Output stayed outside git: %s\n' "$OUT_DIR"
