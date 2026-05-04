#!/usr/bin/env bash
set -euo pipefail

ROOT="${NICME_ROOT:-/mnt/storage/github/NICME}"
OUT="${PMI10_V3_OUT:-${ROOT}/results/pmi10_v3_balanced_convnext_base_20260502}"
SPLIT_DIR="${ROOT}/data/prepared/pmi_pills_10_no_cal/splits/balanced"
PROFILE="convnext_base_v3_balanced"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

CHAIN_DIR="${OUT}/chain"
LOG="${CHAIN_DIR}/phase3_chain.log"
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
    exit "${rc}"
  fi
}

refuse_active_training() {
  local active
  active="$(pgrep -af 'scripts/train.py' | grep -v 'pgrep -af' || true)"
  if [[ -n "${active}" ]]; then
    printf 'Refusing to start because active training processes were found:\n%s\n' "${active}" >&2
    status_line "preflight" "failed"
    exit 1
  fi
}

cd "${ROOT}"
: > "${LOG}"
: > "${STATUS}"
status_line "chain" "started"
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
    --output-dir "${OUT}/readiness"

run_phase "unit-tests" \
  micromamba run -n ml pytest -q \
    tests/test_loss_functions.py \
    tests/test_data_prep.py \
    tests/test_pmi10_no_cal_runner.py \
    tests/test_multiclass_readiness.py \
    tests/test_modeling.py

run_phase "preflight" \
  micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
    --phase preflight \
    --variant balanced \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --hpo-profile "${PROFILE}" \
    --seed 42

run_phase "dry-run-hpo-v3-plan" \
  micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
    --phase hpo-v3 \
    --variant balanced \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --hpo-profile "${PROFILE}" \
    --seed 42 \
    --trials 64

run_phase "smoke" \
  micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
    --phase smoke \
    --variant balanced \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --models convnext_base.fb_in22k_ft_in1k \
    --losses ce,nicme_hybrid,nicme_v3_hybrid \
    --epochs 2 \
    --seed 42 \
    --per-run-timeout-minutes 240 \
    --execute

run_phase "hpo-v3" \
  micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
    --phase hpo-v3 \
    --variant balanced \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --hpo-profile "${PROFILE}" \
    --seed 42 \
    --trials 64 \
    --per-run-timeout-minutes 720 \
    --execute

run_phase "hpo-ce-matched" \
  micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
    --phase hpo-ce-matched \
    --variant balanced \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --hpo-profile "${PROFILE}" \
    --seed 42

run_phase "hpo-ce" \
  micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
    --phase hpo-ce \
    --variant balanced \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --hpo-profile "${PROFILE}" \
    --seed 42 \
    --trials 16 \
    --per-run-timeout-minutes 720 \
    --execute

run_phase "hpo-v2-matched" \
  micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
    --phase hpo-v2-matched \
    --variant balanced \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --hpo-profile "${PROFILE}" \
    --seed 42

run_phase "final" \
  micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
    --phase final \
    --variant balanced \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --hpo-profile "${PROFILE}" \
    --seed 42 \
    --per-run-timeout-minutes 720 \
    --execute

run_phase "tpt-stress" \
  micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
    --phase tpt-stress \
    --variant balanced \
    --split-dir "${SPLIT_DIR}" \
    --output-root "${OUT}" \
    --hpo-profile "${PROFILE}" \
    --seed 42 \
    --epochs 64 \
    --per-run-timeout-minutes 1200 \
    --execute

status_line "chain" "completed"
