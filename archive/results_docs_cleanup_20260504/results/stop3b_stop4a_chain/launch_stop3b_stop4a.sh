#!/usr/bin/env bash
set -euo pipefail
set -x
trap 'printf "[%s] launcher exit code: %s\n" "$(date -Is)" "$?"' EXIT

cd /mnt/storage/github/NICME
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

printf "[%s] Starting Stop 3B imbalance decoupling\n" "$(date -Is)"
micromamba run -n ml python scripts/run_stop3_main.py \
  --phase stop3b_imbalance_decoupling \
  --output-root results/stop3b_imbalance_decoupling \
  --datasets spider_target_minority,spider_target_majority,breakhis_natural \
  --models vit,timm_dinov3_vit_lora \
  --methods ce,ce_calibrated_cost_min,menon_logit_adjusted,cs_regularized_ce,nicme_logit_adjustment,nicme_hybrid \
  --seeds 42,43,44 \
  --time-budget-hours 8 \
  --cleanup-checkpoints \
  --execute

printf "[%s] Stop 3B completed; starting Stop 4A backbone ablation\n" "$(date -Is)"
micromamba run -n ml python scripts/run_stop3_main.py \
  --phase stop4a_backbone_ablation \
  --output-root results/stop4a_backbone_ablation \
  --datasets spider_balanced,breakhis_balanced \
  --models convnext,timm_dinov3_convnext_lora \
  --methods ce_calibrated_cost_min,nicme_logit_adjustment,nicme_hybrid \
  --seeds 42,43,44 \
  --time-budget-hours 6 \
  --cleanup-checkpoints \
  --execute

printf "[%s] Stop 4A completed\n" "$(date -Is)"
