#!/usr/bin/env bash
set -euo pipefail

ROOT="${NICME_ROOT:-/mnt/storage/github/NICME}"
OUT_5E5="${PMI10_SOTA_PRETTY_LR5E5_OUT:-${ROOT}/results/pmi10_sota_pretty_balanced_lr5e5_20260503}"
OUT_1E4="${PMI10_SOTA_PRETTY_LR1E4_OUT:-${ROOT}/results/pmi10_sota_pretty_balanced_lr1e4_20260503}"
COMBINED_OUT="${PMI10_SOTA_PRETTY_DUAL_LR_OUT:-${ROOT}/results/pmi10_sota_pretty_balanced_dual_lr_20260503}"
SPLIT_DIR="${ROOT}/data/prepared/pmi_pills_10_no_cal/splits/balanced"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

CHAIN_DIR="${COMBINED_OUT}/chain"
LOG="${CHAIN_DIR}/dual_lr_sota_chain.log"
STATUS="${COMBINED_OUT}/status.jsonl"

mkdir -p "${CHAIN_DIR}" "${OUT_5E5}/chain" "${OUT_1E4}/chain"

status_line() {
  local status_file="$1"
  local phase="$2"
  local status="$3"
  printf '{"time":"%s","phase":"%s","status":"%s"}\n' "$(date -Is)" "${phase}" "${status}" >> "${status_file}"
}

all_status() {
  local phase="$1"
  local status="$2"
  status_line "${STATUS}" "${phase}" "${status}"
}

profile_status() {
  local profile="$1"
  local phase="$2"
  local status="$3"
  local root
  if [[ "${profile}" == "lr5e5" ]]; then
    root="${OUT_5E5}"
  else
    root="${OUT_1E4}"
  fi
  status_line "${root}/chain/status.jsonl" "${phase}" "${status}"
}

run_phase() {
  local phase="$1"
  shift
  local command=("$@")
  all_status "${phase}" "started"
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
    all_status "${phase}" "completed"
  else
    local rc=$?
    printf '[%s] FAILED %s rc=%s\n' "$(date -Is)" "${phase}" "${rc}" >> "${LOG}"
    all_status "${phase}" "failed"
    all_status "chain" "failed"
    exit "${rc}"
  fi
}

run_profile_phase() {
  local profile="$1"
  local root="$2"
  local phase="$3"
  shift 3
  local command=("$@")
  profile_status "${profile}" "${phase}" "started"
  run_phase "${profile}-${phase}" "${command[@]}"
  profile_status "${profile}" "${phase}" "completed"
}

refuse_active_training() {
  local active
  active="$(pgrep -af 'scripts/train.py' | grep -v 'pgrep -af' || true)"
  if [[ -n "${active}" ]]; then
    printf 'Refusing to start because active training processes were found:\n%s\n' "${active}" >&2
    all_status "preflight" "failed"
    all_status "chain" "failed"
    exit 1
  fi
}

runner_cmd() {
  local phase="$1"
  local profile="$2"
  local output_root="$3"
  shift 3
  micromamba run -n ml python scripts/run_pmi10_sota_baselines.py \
    --phase "${phase}" \
    --comparison-profile pretty_balanced \
    --learning-rate-profile "${profile}" \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${output_root}" \
    --seed 42 \
    "$@"
}

cd "${ROOT}"
: > "${LOG}"
: > "${STATUS}"
: > "${OUT_5E5}/chain/status.jsonl"
: > "${OUT_1E4}/chain/status.jsonl"
all_status "chain" "started"
profile_status "lr5e5" "chain" "started"
profile_status "lr1e4" "chain" "started"
refuse_active_training

run_phase "prepare-balanced-split" \
  micromamba run -n ml python scripts/prepare_data.py \
    --dataset pmi_pills_10_no_cal \
    --input-dir data/raw/pmi \
    --output-dir data/prepared/pmi_pills_10_no_cal \
    --seed 42

run_phase "readiness" \
  micromamba run -n ml python scripts/check_multiclass_readiness.py \
    --datasets pmi_pills_10_no_cal \
    --variants balanced \
    --raw-root data/raw \
    --prepared-root data/prepared \
    --output-dir "${COMBINED_OUT}/readiness"

run_phase "unit-tests" \
  micromamba run -n ml pytest -q \
    tests/test_loss_functions.py \
    tests/test_sota_baselines_runner.py \
    tests/test_data_prep.py \
    tests/test_multiclass_readiness.py

run_profile_phase "lr5e5" "${OUT_5E5}" "preflight" \
  runner_cmd preflight lr5e5 "${OUT_5E5}"
run_profile_phase "lr1e4" "${OUT_1E4}" "preflight" \
  runner_cmd preflight lr1e4 "${OUT_1E4}"

run_profile_phase "lr5e5" "${OUT_5E5}" "dry-run-plan" \
  runner_cmd dry-run lr5e5 "${OUT_5E5}"
run_profile_phase "lr1e4" "${OUT_1E4}" "dry-run-plan" \
  runner_cmd dry-run lr1e4 "${OUT_1E4}"

run_profile_phase "lr5e5" "${OUT_5E5}" "smoke" \
  runner_cmd smoke lr5e5 "${OUT_5E5}" --per-run-timeout-minutes 240
run_profile_phase "lr1e4" "${OUT_1E4}" "smoke" \
  runner_cmd smoke lr1e4 "${OUT_1E4}" --per-run-timeout-minutes 240

run_profile_phase "lr5e5" "${OUT_5E5}" "long-baseline-chain" \
  runner_cmd run lr5e5 "${OUT_5E5}" --per-run-timeout-minutes 960
run_profile_phase "lr5e5" "${OUT_5E5}" "analysis" \
  runner_cmd analyze lr5e5 "${OUT_5E5}"

run_profile_phase "lr1e4" "${OUT_1E4}" "long-baseline-chain" \
  runner_cmd run lr1e4 "${OUT_1E4}" --per-run-timeout-minutes 960
run_profile_phase "lr1e4" "${OUT_1E4}" "analysis" \
  runner_cmd analyze lr1e4 "${OUT_1E4}"

run_phase "combined-analysis" \
  micromamba run -n ml python scripts/run_pmi10_sota_baselines.py \
    --phase combined-analyze \
    --comparison-profile pretty_balanced \
    --combined-output-root "${COMBINED_OUT}" \
    --lr5e5-output-root "${OUT_5E5}" \
    --lr1e4-output-root "${OUT_1E4}"

profile_status "lr5e5" "chain" "completed"
profile_status "lr1e4" "chain" "completed"
all_status "chain" "completed"
