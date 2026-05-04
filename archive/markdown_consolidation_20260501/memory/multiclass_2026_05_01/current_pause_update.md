# Current Multiclass Pause Update - 2026-05-01

This note supersedes the older pre-Facebook-DINOv3 blocking status in this
memory folder.

Current source of truth:

- Progress summary:
  `results/multiclass_pause_summary_20260501/multiclass_progress_summary.md`
- Current readiness:
  `results/multiclass_pause_summary_20260501/mc0_readiness_current/mc0_readiness.md`
- MC-FB2 stability:
  `results/multiclass_mc_fb2_lora_stability_20260501/mc1/mc_fb2_lora_stability_summary.md`
- MC2 official DINOv3 LoRA:
  `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/mc2/mc2_official_dinov3_lora_balanced_summary.md`
- MC3 ledger:
  `results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/mc3/run_ledger.csv`

State:

- MC0 readiness is now complete for EyePACS DR and PMI Pills, balanced and
  natural variants.
- Official Facebook DINOv3 LoRA integration is active and has passed the
  MC-FB2 stability gate.
- MC2 completed 20/20 official DINOv3 LoRA balanced runs with 0 final failures
  and 1 retry recovery.
- MC3 has been paused at user request after 6 completed rows. The paused row is
  EyePACS DR cost-sensitive CE seed 43 and is marked `paused_by_user`.
- No active MC3 processes remain after the pause.
- Resume should not require a reboot unless a fresh health check reports
  D-state user processes, GPU taint, or CUDA canary failure.

Scientific note:

- MC2 selected official DINOv3 ConvNeXt LoRA configurations for MC3.
- Early evidence does not yet show NICME winning the recall-first objective.
  CE/cost-sensitive CE currently lead cared-class recall in MC2. NICME remains
  important for MC3, particularly for ATC/cost behavior, but conclusions must
  remain fair and data-driven.
