#!/usr/bin/env bash
set -euo pipefail

ROOT="${NICME_ROOT:-/mnt/storage/github/NICME}"
OUT_1E5="${PMI10_SOTA_PRETTY_LR1E5_OUT:-${ROOT}/results/pmi10_sota_pretty_balanced_lr1e5_20260503}"
OUT_5E5="${PMI10_SOTA_PRETTY_LR5E5_OUT:-${ROOT}/results/pmi10_sota_pretty_balanced_lr5e5_20260503}"
OUT_1E4="${PMI10_SOTA_PRETTY_LR1E4_OUT:-${ROOT}/results/pmi10_sota_pretty_balanced_lr1e4_20260503}"
COMBINED_OUT="${PMI10_SOTA_PRETTY_TRIPLE_LR_OUT:-${ROOT}/results/pmi10_sota_pretty_balanced_triple_lr_20260503}"
SPLIT_DIR="${ROOT}/data/prepared/pmi_pills_10_no_cal/splits/balanced"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

CHAIN_DIR="${COMBINED_OUT}/chain"
LOG="${CHAIN_DIR}/lr1e5_sota_chain.log"
STATUS="${COMBINED_OUT}/status.jsonl"

mkdir -p "${CHAIN_DIR}" "${OUT_1E5}/chain"

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
  local phase="$1"
  local status="$2"
  status_line "${OUT_1E5}/chain/status.jsonl" "${phase}" "${status}"
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
    profile_status "chain" "failed"
    exit "${rc}"
  fi
}

run_lr1e5_phase() {
  local phase="$1"
  shift
  local command=("$@")
  profile_status "${phase}" "started"
  run_phase "lr1e5-${phase}" "${command[@]}"
  profile_status "${phase}" "completed"
}

refuse_active_training() {
  local active
  active="$(pgrep -af 'scripts/train.py' | grep -v 'pgrep -af' || true)"
  if [[ -n "${active}" ]]; then
    printf 'Refusing to start because active training processes were found:\n%s\n' "${active}" >&2
    all_status "preflight" "failed"
    all_status "chain" "failed"
    profile_status "chain" "failed"
    exit 1
  fi
}

runner_cmd() {
  local phase="$1"
  shift
  micromamba run -n ml python scripts/run_pmi10_sota_baselines.py \
    --phase "${phase}" \
    --comparison-profile pretty_balanced \
    --learning-rate-profile lr1e5 \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT_1E5}" \
    --seed 42 \
    "$@"
}

cd "${ROOT}"
: > "${LOG}"
: > "${STATUS}"
: > "${OUT_1E5}/chain/status.jsonl"
all_status "chain" "started"
profile_status "chain" "started"
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

run_lr1e5_phase "preflight" \
  runner_cmd preflight

run_lr1e5_phase "dry-run-plan" \
  runner_cmd dry-run

run_lr1e5_phase "smoke" \
  runner_cmd smoke --per-run-timeout-minutes 240

run_lr1e5_phase "long-baseline-chain" \
  runner_cmd run --per-run-timeout-minutes 960

run_lr1e5_phase "analysis" \
  runner_cmd analyze

run_phase "triple-combined-analysis" \
  micromamba run -n ml python scripts/run_pmi10_sota_baselines.py \
    --phase combined-analyze \
    --comparison-profile pretty_balanced \
    --combined-output-root "${COMBINED_OUT}" \
    --lr1e5-output-root "${OUT_1E5}" \
    --lr5e5-output-root "${OUT_5E5}" \
    --lr1e4-output-root "${OUT_1E4}"

profile_status "chain" "completed"
all_status "chain" "completed"
