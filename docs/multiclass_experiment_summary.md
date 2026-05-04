# Multiclass Experiment Summary

Updated: 2026-05-01

The multiclass extension adds EyePACS diabetic retinopathy and PMI Pills experiments with fixed dataset cost matrices, multiclass metrics, official Facebook DINOv3 LoRA models, and MC-style experiment stops.

## Implemented Infrastructure

- Dataset profiles for `eyepacs_dr` and `pmi_pills`, including class names, cared classes, cost matrices, metric metadata, and cost-matrix hashes.
- Multiclass cost validation and decision behavior using `C[true_label][predicted_label]`.
- EyePACS and PMI preparation with balanced and natural split variants.
- Multiclass runner for MC1 through MC4-style stops, including ledgers, per-run configs, logs, fresh checkpoint namespaces, resume behavior, storage-safe cache environment, GPU/D-state/CUDA preflights, and retry-once handling for the observed PyTorch/PEFT traversal failure.
- Multiclass metrics including ATC, normalized ATC, target recall, macro recall, balanced accuracy, macro F1, ECE, NLL, confusion matrices, cost-weighted confusion matrices, DR QWK, and calibrated cost-min decision reports.

## Data Readiness

MC0 readiness is complete for EyePACS DR and PMI Pills, for both balanced and natural variants.

Current balanced paper-run splits:

| Dataset | Split root | Counts | Cared classes |
|---|---|---|---|
| EyePACS DR | `data/prepared/eyepacs_dr/splits/balanced` | train 2485, validation 335, calibration 350, test 370 | `DR4` |
| PMI Pills | `data/prepared/pmi_pills/splits/balanced` | train 1940, validation 320, calibration 300, test 640 | `50111-0434`, `53489-0156`, `53746-0544`, `68382-0227` |

Both balanced audits reported zero missing images and zero patient overlap across checked split pairs.

## Stop Status

| Stop | Status | Result |
|---|---|---|
| MC0 | Complete | EyePACS DR and PMI Pills ready for balanced and natural variants |
| MC-FB2 | Complete | Official Facebook DINOv3 LoRA stability gate passed, 8/8 runs |
| MC2 | Complete | 20/20 official DINOv3 LoRA balanced runs, 0 final failures, 1 retry recovery |
| MC3 | Paused | 6 completed rows, 1 user-paused row, 24 full runs remaining |

MC2 selected official DINOv3 ConvNeXt LoRA candidates for MC3:

- EyePACS DR: `ce`, `cs_regularized_ce`, `nicme_hybrid`
- PMI Pills: `ce`, `cs_regularized_ce`, `nicme_logit_adjustment`

## Current Scientific Read

MC2 does not support claiming that NICME is already the top recall method. On EyePACS, ConvNeXt LoRA cost-sensitive CE had the best target recall among MC2 candidates. On PMI, CE and cost-sensitive CE tied on target recall, while `nicme_logit_adjustment` lowered normalized ATC and improved balanced accuracy but had lower target recall.

MC3 is incomplete. The partial MC3 signal is:

| Dataset | Method | Completed seeds | Target recall mean | Normalized ATC mean | Balanced accuracy mean |
|---|---|---:|---:|---:|---:|
| EyePACS DR | CE | 5 | 0.7459 | 0.1476 | 0.3746 |
| EyePACS DR | cost-sensitive CE | 1 | 0.7838 | 0.1182 | 0.4108 |

The one-seed cost-sensitive CE row is a progress signal, not a stable conclusion.

## Active Artifacts

- MC3 ledger: `results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/mc3/run_ledger.csv`
- MC2 selection JSON: `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc2_selection.json`
- MC3 launch commands: `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc3_launch_commands.sh`
- Archived multiclass reports: [archive/markdown_consolidation_20260501/results/](../archive/markdown_consolidation_20260501/results/)

## Resume Guidance

Resume MC3 only after a fresh health check. The paused row is EyePACS DR cost-sensitive CE seed 43; the runner should rerun it because only completed rows are skipped.
