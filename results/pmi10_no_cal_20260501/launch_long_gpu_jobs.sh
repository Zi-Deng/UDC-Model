#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/storage/github/NICME"
RESULT_ROOT="${REPO_ROOT}/results/pmi10_no_cal_20260501"
CHAIN_DIR="${RESULT_ROOT}/chain"
LOG_PATH="${LOG_PATH:-${CHAIN_DIR}/long_gpu_jobs.log}"
STATUS_PATH="${STATUS_PATH:-${CHAIN_DIR}/status.jsonl}"

mkdir -p "${CHAIN_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

record_status() {
  local phase="$1"
  local status="$2"
  printf '{"time":"%s","phase":"%s","status":"%s"}\n' "$(date --iso-8601=seconds)" "$phase" "$status" >> "$STATUS_PATH"
}

run_step() {
  local phase="$1"
  shift
  log "START ${phase}: $*"
  record_status "${phase}" "started"
  "$@"
  record_status "${phase}" "completed"
  log "DONE ${phase}"
}

{
  log "Focused PMI-10 no-calibration long GPU chain launched"
  log "Repo: ${REPO_ROOT}"
  log "Results: ${RESULT_ROOT}"
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi || true

  run_step smoke \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase smoke \
      --output-root "${RESULT_ROOT}" \
      --seed 42 \
      --per-run-timeout-minutes 240 \
      --execute

  run_step screen \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase screen \
      --output-root "${RESULT_ROOT}" \
      --seed 42 \
      --per-run-timeout-minutes 480 \
      --execute

  run_step select \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase select \
      --output-root "${RESULT_ROOT}" \
      --top-k 3

  run_step hpo-nicme \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase hpo-nicme \
      --output-root "${RESULT_ROOT}" \
      --seed 42 \
      --trials 100 \
      --per-run-timeout-minutes 720 \
      --execute

  run_step hpo-ce \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase hpo-ce \
      --output-root "${RESULT_ROOT}" \
      --seed 42 \
      --trials 30 \
      --per-run-timeout-minutes 720 \
      --execute

  run_step final \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase final \
      --output-root "${RESULT_ROOT}" \
      --seed 42 \
      --per-run-timeout-minutes 720 \
      --execute

  log "Focused PMI-10 no-calibration long GPU chain completed"
} >> "${LOG_PATH}" 2>&1
