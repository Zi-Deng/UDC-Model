# NICME Results Index

Updated: 2026-05-04

This file is the live index for `results/`. It separates current evidence from retained historical material so old probes and superseded plans do not look like active work.

## Current Canonical Results

| Area | Status | Primary paths |
|---|---|---|
| PMI-10 NICME v3 HPO | Complete, current best HPO record | `pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/README.md`, `pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md` |
| PMI-10 SOTA/baseline comparison | Complete, pre-HPO baseline suite retained for comparison | `pmi10_sota_pretty_balanced_lr5e5_20260503/`, `pmi10_sota_pretty_balanced_lr1e5_20260503/`, `pmi10_sota_pretty_balanced_lr1e4_20260503/`, `pmi10_sota_pretty_balanced_triple_lr_20260503/` |
| PMI-10 baseline raw metrics | Retained because comparison ledgers point here | `pmi10_sota_convnext_base_test/` |
| Binary Stop 3/4 evidence | Complete final binary experiment sequence | `stop3a_balanced_primary/`, `stop3b_imbalance_decoupling/`, `stop4a_backbone_ablation/`, `stop4b_cost_ratio_sensitivity/` |
| Multiclass MC2/MC3 | MC2 complete, MC3 paused | `multiclass_mc2_official_dinov3_lora_balanced_20260501/`, `multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/` |
| Official DINOv3 multiclass raw metrics | Retained because MC2/MC3 and smoke ledgers point here | `facebook_dinov3_vit_lora_test/`, `facebook_dinov3_convnext_lora_test/` |

## Historical Retained Results

These folders are not current guidance, but they contain completed ledgers, logs, raw metrics, or older analyses that may still be useful for audit trails:

- Earlier binary probes and tuning: `stop1_smoke/`, `stop1_smoke_timm/`, `stop1_smoke_timm_lora/`, `stop2_prototype/`, `stop2a_nicme_tuning/`, `stop2b_threshold_tuning/`, `stop2c_nicme_frontier/`
- Earlier binary sweep/raw metric trees: `sweep_cost_*`, `sweep_comparison*`, `resnet_test/`, `convnext_test/`, `vit_test/`, `timm_dinov3_*_test/`
- Earlier multiclass runs with ledgers or raw metric outputs: `multiclass*/`, except planning/preflight folders moved to archive.
- Earlier PMI-10 screens and HPO attempts: `pmi10_no_cal_20260501/`, `pmi10_no_cal_convnext_base_20260501/`, `pmi10_v3_balanced_convnext_base_20260502/`, and backbone-screen metric folders.
- First PMI-10 balanced SOTA run: `pmi10_sota_baselines_balanced_20260502/`, retained as historical ledgered baseline material.

## Archived Cleanup Material

Small superseded probes, dry-runs, launcher-only folders, planning/preflight outputs, the old two-class HPO folder, stale multiclass memory state, and the redundant PMI-10 dual-LR comparison were moved to:

- `../archive/results_docs_cleanup_20260504/MANIFEST.md`

The archive preserves original paths under `archive/results_docs_cleanup_20260504/`; no experiment history was deleted.

## Reading Order

For current interpretation, start with:

1. `../docs/current_status.md`
2. `../docs/pmi10_hpo_sota_summary.md`
3. `pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md`
4. `../docs/binary_experiment_summary.md`
5. `../docs/multiclass_experiment_summary.md`
