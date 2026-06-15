#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${AVW_SMOKE_OUT_DIR:-/tmp/avatar-video-workbench-smoke}"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${AVW_DEMO_VENV:-${REPO_ROOT}/.venv}"
AVW_BIN="${VENV_DIR}/bin/avw"

cd "${REPO_ROOT}"

echo "==> Preparing local demo environment"
if [[ ! -x "${AVW_BIN}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install -e ".[test]"
fi

echo "==> Running synthetic smoke demo into ${OUT_DIR}"
"${AVW_BIN}" smoke-demo --out-dir "${OUT_DIR}" --force

echo "==> Running Vertex preflight against generated YAML"
"${AVW_BIN}" preflight-vertex \
  --job-yaml "${OUT_DIR}/compiled/jobs/vertex-custom-job.yaml"

echo "==> Running publication scan on the checkout"
SCAN_OUTPUT="$("${AVW_BIN}" scan-publication .)"
printf '%s\n' "${SCAN_OUTPUT}"

if [[ "${SCAN_OUTPUT}" != "[]" ]]; then
  echo "Expected publication scan to return []" >&2
  exit 1
fi

echo "==> Demo complete"
echo "Generated assets stayed under ${OUT_DIR}; do not commit them."
