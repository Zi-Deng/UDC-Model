#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/storage/github/NICME"
RESULT_ROOT="${REPO_ROOT}/results/pmi10_no_cal_convnext_base_20260501"
SMOKE_RESULT_ROOT="${RESULT_ROOT}/hpo_smoke"
ARCHIVE_ROOT="${REPO_ROOT}/archive/failed_experiment_runs/pmi10_mixed_backbone_hpo_20260501"
CHAIN_DIR="${RESULT_ROOT}/chain"
LOG_PATH="${LOG_PATH:-${CHAIN_DIR}/convnext_base_hpo_chain.log}"
STATUS_PATH="${STATUS_PATH:-${CHAIN_DIR}/status.jsonl}"
HPO_PROFILE="convnext_base_fast"

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
  printf '{"time":"%s","phase":"%s","status":"%s"}\n' "$(date --iso-8601=seconds)" "$phase" "$status" >> "${STATUS_PATH}"
}

run_step() {
  local phase="$1"
  shift
  log "START ${phase}: $*"
  record_status "${phase}" "started"
  if "$@"; then
    record_status "${phase}" "completed"
    log "DONE ${phase}"
  else
    local code=$?
    record_status "${phase}" "failed"
    log "FAILED ${phase} exit=${code}"
    exit "${code}"
  fi
}

archive_old_hpo() {
  mkdir -p "${ARCHIVE_ROOT}/results/pmi10_no_cal_20260501"
  local manifest="${ARCHIVE_ROOT}/MANIFEST.md"
  if [[ ! -f "${manifest}" ]]; then
    printf '# Failed Mixed-Backbone PMI-10 HPO Archive\n\n' > "${manifest}"
    printf 'Created by `launch_convnext_base_hpo_chain.sh`.\n\n' >> "${manifest}"
  fi
  local rel
  for rel in \
    "results/pmi10_no_cal_20260501/hpo-nicme" \
    "results/pmi10_no_cal_20260501/hpo-ce"; do
    local src="${REPO_ROOT}/${rel}"
    if [[ -e "${src}" ]]; then
      local dst="${ARCHIVE_ROOT}/${rel}"
      if [[ -e "${dst}" ]]; then
        dst="${dst}_$(date +%Y%m%d_%H%M%S)"
      fi
      mkdir -p "$(dirname "${dst}")"
      mv "${src}" "${dst}"
      printf -- '- `%s` -> `%s`\n' "${rel}" "${dst#${REPO_ROOT}/}" >> "${manifest}"
    fi
  done
}

{
  log "ConvNeXt-base-only PMI-10 no-calibration recovery chain launched"
  log "Repo: ${REPO_ROOT}"
  log "Results: ${RESULT_ROOT}"
  log "HPO profile: ${HPO_PROFILE}"
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi || true

  run_step preflight \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase preflight \
      --output-root "${RESULT_ROOT}" \
      --hpo-profile "${HPO_PROFILE}" \
      --seed 42

  run_step archive-old-hpo archive_old_hpo

  run_step hpo-nicme-plan \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase hpo-nicme \
      --output-root "${RESULT_ROOT}" \
      --hpo-profile "${HPO_PROFILE}" \
      --seed 42 \
      --trials 36

  run_step hpo-nicme-smoke \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase hpo-nicme \
      --output-root "${SMOKE_RESULT_ROOT}" \
      --hpo-profile "${HPO_PROFILE}" \
      --seed 42 \
      --trials 1 \
      --per-run-timeout-minutes 360 \
      --execute

  run_step hpo-nicme \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase hpo-nicme \
      --output-root "${RESULT_ROOT}" \
      --hpo-profile "${HPO_PROFILE}" \
      --seed 42 \
      --trials 36 \
      --per-run-timeout-minutes 720 \
      --execute

  run_step hpo-ce-matched \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase hpo-ce-matched \
      --output-root "${RESULT_ROOT}" \
      --hpo-profile "${HPO_PROFILE}" \
      --seed 42

  run_step hpo-ce \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase hpo-ce \
      --output-root "${RESULT_ROOT}" \
      --hpo-profile "${HPO_PROFILE}" \
      --seed 42 \
      --trials 16 \
      --per-run-timeout-minutes 720 \
      --execute

  FINAL_CONFIGS="${RESULT_ROOT}/hpo-nicme/best_config.json,${RESULT_ROOT}/hpo-ce-matched/best_config.json,${RESULT_ROOT}/hpo-ce/best_config.json"
  run_step final \
    micromamba run -n ml python scripts/run_pmi10_no_cal_experiments.py \
      --phase final \
      --output-root "${RESULT_ROOT}" \
      --hpo-profile "${HPO_PROFILE}" \
      --configs "${FINAL_CONFIGS}" \
      --seed 42 \
      --per-run-timeout-minutes 720 \
      --execute

  log "ConvNeXt-base-only PMI-10 no-calibration recovery chain completed"
} >> "${LOG_PATH}" 2>&1
