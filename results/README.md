# NICME Results Index

Updated: 2026-05-04

This file is the live index for `results/`. It separates current evidence from retained historical material so old probes and superseded plans do not look like active work.

## Current Canonical Results

| Area | Status | Primary paths |
|---|---|---|
| PMI-20 NICME paper SOTA | Complete, current paper-facing result | `pmi20_nicme_sota_lr5e5_multiseed_20260504/README.md`, `pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/pmi20_sota_table.md` |
| PMI-20 source suites | Complete, retained as provenance for the consolidated paper table | `pmi20_camera_ready_lr5e5_multiseed_20260504/`, `pmi20_nicme_alpha_lambda6_lr5e5_multiseed_20260504/` |
| PMI-10 NICME HPO and multi-seed support | Complete, supporting sensitivity/robustness evidence | `pmi10_camera_ready_lr5e5_multiseed_20260504/`, `pmi10_nicme_top5_alpha_lambda_lr5e5_multiseed_20260504/`, historical single-seed HPO result root |
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
- Historical PMI-10 generated result folders may retain versioned names. They are provenance only; current paper-facing language should use `NICME`.

## Archived Cleanup Material

Small superseded probes, dry-runs, launcher-only folders, planning/preflight outputs, the old two-class HPO folder, stale multiclass memory state, and the redundant PMI-10 dual-LR comparison were moved to:

- `../archive/results_docs_cleanup_20260504/MANIFEST.md`
- `../archive/paper_position_cleanup_20260504/MANIFEST.md`

The archive preserves original paths under `archive/results_docs_cleanup_20260504/`; no experiment history was deleted.

## Reading Order

For current interpretation, start with:

1. `../docs/current_status.md`
2. `../docs/paper_results_summary.md`
3. `pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/pmi20_sota_table.md`
4. `../docs/pmi10_hpo_sota_summary.md`
5. `../docs/binary_experiment_summary.md`
6. `../docs/multiclass_experiment_summary.md`
