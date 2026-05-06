#!/usr/bin/env bash
set -euo pipefail

ROOT="${NICME_ROOT:-/mnt/storage/github/NICME}"
OUT="${BINARY_CAMERA_READY_COST8_7_OUT:-${ROOT}/results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505}"
SEEDS="${BINARY_CAMERA_READY_SEEDS:-42,43,44}"
DATASETS="${BINARY_CAMERA_READY_DATASETS:-spider,breakhis}"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

CHAIN_DIR="${OUT}/chain"
LOG="${CHAIN_DIR}/binary_camera_ready_cost8_7_lr5e5_chain.log"
STATUS="${CHAIN_DIR}/status.jsonl"

mkdir -p "${CHAIN_DIR}"

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
  active="$(ps -eo pid=,args= | awk '/[p]ython/ && /scripts\/train\.py/ {print}')"
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
  micromamba run -n ml python scripts/run_binary_camera_ready_cost8_7_lr5e5.py \
    --phase "${phase}" \
    --output-root "${OUT}" \
    --seeds "${SEEDS}" \
    --datasets "${DATASETS}" \
    "$@"
}

cd "${ROOT}"
: > "${LOG}"
: > "${STATUS}"
status_line "chain" "started"
refuse_active_training

run_phase "gpu-check" nvidia-smi

run_phase "unit-tests" \
  micromamba run -n ml pytest -q \
    tests/test_loss_functions.py \
    tests/test_costs.py \
    tests/test_dataset_profiles.py \
    tests/test_sota_baselines_runner.py \
    tests/test_pmi20_camera_ready_runner.py \
    tests/test_binary_camera_ready_runner.py

run_phase "preflight" \
  runner_cmd preflight

run_phase "dry-run" \
  runner_cmd dry-run

run_phase "smoke" \
  runner_cmd smoke --per-run-timeout-minutes 240

run_phase "long-binary-camera-ready-chain" \
  runner_cmd run --per-run-timeout-minutes 960

run_phase "analysis" \
  runner_cmd analyze

status_line "chain" "completed"
