#!/usr/bin/env bash
set -euo pipefail

ROOT="${NICME_ROOT:-/mnt/storage/github/NICME}"
OUT="${PMI20_NICME_GRID7_LR5E5_OUT:-${ROOT}/results/pmi20_nicme_alpha_lambda_grid7_lr5e5_seed42_20260506}"
SEEDS="${PMI20_NICME_GRID7_SEEDS:-42}"
SPLIT_DIR="${ROOT}/data/prepared/pmi_pills/splits/balanced"
PAPER_FIGURES="${NICME_PAPER_FIGURES_OUT:-${ROOT}/paper_figures}"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

CHAIN_DIR="${OUT}/chain"
LOG="${CHAIN_DIR}/pmi20_nicme_alpha_lambda_grid7_lr5e5_chain.log"
STATUS="${CHAIN_DIR}/status.jsonl"

mkdir -p "${CHAIN_DIR}" "${PAPER_FIGURES}/data"

status_line() {
  local phase="$1"
  local status="$2"
  printf '{"time":"%s","phase":"%s","status":"%s"}\n' "$(date -Is)" "${phase}" "${status}" >> "${STATUS}"
}

run_phase() {
  local phase="$1"
  shift
  local command=("$@")
  status_line "${phase}" "started"
  {
    printf '\n[%s] START %s\n' "$(date -Is)" "${phase}"
    printf '[%s] CMD' "$(date -Is)"
    for arg in "${command[@]}"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
  } >> "${LOG}"
  if "${command[@]}" >> "${LOG}" 2>&1; then
    printf '[%s] DONE %s\n' "$(date -Is)" "${phase}" >> "${LOG}"
    status_line "${phase}" "completed"
  else
    local rc=$?
    printf '[%s] FAILED %s rc=%s\n' "$(date -Is)" "${phase}" "${rc}" >> "${LOG}"
    status_line "${phase}" "failed"
    status_line "chain" "failed"
    exit "${rc}"
  fi
}

refuse_active_training() {
  local active
  active="$(pgrep -af 'scripts/train.py' | grep -v 'pgrep -af' || true)"
  if [[ -n "${active}" ]]; then
    printf 'Refusing to start because active training processes were found:\n%s\n' "${active}" >&2
    status_line "preflight" "failed"
    status_line "chain" "failed"
    exit 1
  fi
}

runner_cmd() {
  local phase="$1"
  shift
  micromamba run -n ml python scripts/run_pmi20_nicme_alpha_lambda_grid7_lr5e5.py \
    --phase "${phase}" \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --seeds "${SEEDS}" \
    "$@"
}

cd "${ROOT}"
: > "${LOG}"
: > "${STATUS}"
status_line "chain" "started"
refuse_active_training

run_phase "gpu-check" \
  nvidia-smi

run_phase "unit-tests" \
  micromamba run -n ml pytest -q \
    tests/test_loss_functions.py \
    tests/test_pmi20_camera_ready_runner.py \
    tests/test_pmi20_nicme_alpha_lambda_grid7_runner.py \
    tests/test_paper_figures_builder.py

run_phase "preflight" \
  runner_cmd preflight

run_phase "dry-run" \
  runner_cmd dry-run

run_phase "smoke" \
  runner_cmd smoke --per-run-timeout-minutes 240

run_phase "long-pmi20-nicme-alpha-lambda-grid7" \
  runner_cmd run --per-run-timeout-minutes 960

run_phase "analysis" \
  runner_cmd analyze

run_phase "paper-figures" \
  micromamba run -n ml python scripts/build_paper_figures.py \
    --phase all \
    --output-root "${PAPER_FIGURES}" \
    --pmi20-alpha-grid "${OUT}/analysis/validation_grid.csv"

status_line "chain" "completed"
