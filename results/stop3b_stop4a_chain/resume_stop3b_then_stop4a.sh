#!/usr/bin/env bash
set -euo pipefail
set -x
trap 'printf "[%s] resume launcher exit code: %s\n" "$(date -Is)" "$?"' EXIT

cd /home/zi/Work/github/UDC-Model

printf "[%s] Resuming Stop 3B until all manifest rows are complete\n" "$(date -Is)"
micromamba run -n ml python scripts/resume_stop_queue.py \
  --output-root results/stop3b_imbalance_decoupling \
  --cleanup-checkpoints

printf "[%s] Stop 3B resume completed; starting Stop 4A backbone ablation\n" "$(date -Is)"
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
