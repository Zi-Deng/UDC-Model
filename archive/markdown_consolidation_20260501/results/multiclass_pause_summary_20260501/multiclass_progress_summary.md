# NICME Multiclass Progress Summary - 2026-05-01

## Pause State

The active MC3 campaign was paused at user request. No MC3 launcher or
training processes remain active after the pause.

- Active output root: `results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501`
- Ledger: `results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/mc3/run_ledger.csv`
- Completed MC3 rows so far: 6
- Paused row: `0002_eyepacs_dr_balanced_facebook_dinov3_convnext_lora_cs_regularized_ce_seed43`
- Paused row elapsed time: 321.722 seconds
- Paused row status: `paused_by_user`
- Resume behavior: the runner only skips rows whose status is `completed`, so this paused seed will be re-run when MC3 resumes.

## Current Data Readiness

Fresh readiness check:
`results/multiclass_pause_summary_20260501/mc0_readiness_current/mc0_readiness.md`

- Current MC0 readiness: all ready.
- EyePACS DR raw data exists under `data/raw/eyepacs`.
- PMI raw data exists under `data/raw/pmi`.
- Both `balanced` and `natural` prepared variants exist for both datasets.
- The current paper-run path is using the balanced variants, per user request.

EyePACS balanced audit:

- Split root: `data/prepared/eyepacs_dr/splits/balanced`
- Cost matrix SHA256: `2ddb9a77faab698cd29f1475048e7f5438e89d3c7420eaa8607238053746d412`
- Cost convention: `C[true_label][predicted_label]`
- Cared class: `DR4`
- Counts: train 2485, validation 335, calibration 350, test 370
- Balanced class counts: train 497/class, validation 67/class, calibration 70/class, test 74/class
- Missing images: 0
- Patient overlap: 0 across checked split pairs

PMI balanced audit:

- Split root: `data/prepared/pmi_pills/splits/balanced`
- Cost matrix SHA256: `af92dbd367c299056f5f41dab4f3b4330a6171b856e8822bac0357fdfe4289d2`
- Cost convention: `C[true_label][predicted_label]`
- Cared classes: `50111-0434`, `53489-0156`, `53746-0544`, `68382-0227`
- Counts: train 1940, validation 320, calibration 300, test 640
- Balanced class counts: train 97/class, validation 16/class, calibration 15/class, test 32/class
- Missing images: 0
- Patient overlap: 0 across checked split pairs
- PMI archive SHA1: `df15ceb6aa039d71d23774258e28acc89bd31b91`

## Implemented Multiclass Infrastructure

The repository now has working multiclass infrastructure for the two target
datasets and fixed source cost matrices.

- Dataset profiles for `eyepacs_dr` and `pmi_pills`, including class names,
  cared classes, cost matrices, metric metadata, and cost-matrix hashes.
- Multiclass cost validation with the repository's `C[true][pred]` convention.
- EyePACS and PMI data preparation with balanced/natural split variants and
  audit reports.
- Multiclass experiment runner for MC1/MC2/MC3/MC4-style stops.
- Run ledgers, per-run config export, per-run stdout/stderr logs, fresh
  checkpoint namespaces, and resume behavior.
- Multiclass metrics: ATC, normalized ATC, per-class recall, target recall,
  macro recall/balanced accuracy, macro F1, accuracy, ECE, NLL, confusion
  matrices, cost-weighted confusion matrices, DR QWK, and calibrated cost-min
  decision reports.
- Official Facebook DINOv3 integration through Hugging Face model IDs, with
  LoRA retained as part of the active paper plan.
- Storage-safe environment propagation for large artifacts:
  `HF_HOME=/mnt/storage/huggingface`,
  `HF_HUB_CACHE=/mnt/storage/huggingface/hub`,
  `HF_DATASETS_CACHE=/mnt/storage/huggingface/datasets`,
  `TORCH_HOME=/mnt/storage/torch`,
  `TMPDIR=/mnt/storage/tmp/nicme`,
  `XDG_CACHE_HOME=/mnt/storage/.cache`,
  `MPLCONFIGDIR=/mnt/storage/.cache/matplotlib`.
- GPU/D-state/CUDA canary preflights before paper-run training jobs.
- Retry-once handling for a narrow PyTorch/PEFT module traversal failure, with
  retry attempts placed in fresh checkpoint namespaces.

## Validation Completed

### MC-FB2 Stability Gate

Artifact:
`results/multiclass_mc_fb2_lora_stability_20260501/mc1/mc_fb2_lora_stability_summary.md`

- Status: passed.
- Planned runs: 8
- Completed runs: 8
- Failed runs: 0
- Retry attempts used: 0
- Datasets: EyePACS DR and PMI Pills
- Backbones: official Facebook DINOv3 ViT LoRA and ConvNeXt LoRA
- Methods: CE and NICME hybrid
- Post-run D-state check: clean
- Post-run GPU taint check: clean
- Post-run CUDA canary: clean
- No hard reboot required for MC-FB2.

### MC2 Prototype/HPO Selection

Artifact:
`results/multiclass_mc2_official_dinov3_lora_balanced_20260501/mc2/mc2_official_dinov3_lora_balanced_summary.md`

- Status: complete.
- Planned runs: 20
- Completed runs: 20
- Final failed runs: 0
- Retry recoveries: 1
- Regression tests after runner patch: 49 passed
- Post-run D-state/GPU/CUDA checks: clean

MC2 selected candidates for MC3:

- EyePACS DR: official DINOv3 ConvNeXt LoRA with `ce`,
  `cs_regularized_ce`, and `nicme_hybrid`.
- PMI Pills: official DINOv3 ConvNeXt LoRA with `ce`,
  `cs_regularized_ce`, and `nicme_logit_adjustment`.

Important MC2 scientific interpretation:

- MC2 does not support claiming NICME is already best on the recall-first
  objective.
- On EyePACS, ConvNeXt LoRA cost-sensitive CE had the best target recall
  among MC2 candidates: target recall 0.7568, normalized ATC 0.1174.
- On PMI, ConvNeXt LoRA CE and cost-sensitive CE tied on target recall
  at 0.4062, while `nicme_logit_adjustment` lowered normalized ATC to 0.0348
  and improved balanced accuracy to 0.7562, but had lower target recall
  at 0.3750.
- NICME remains scientifically important to carry into MC3, especially for
  cost/ATC behavior, but the final paper claim must be determined by MC3 and
  reported fairly.

## MC3 Progress Before Pause

Active MC3 artifact root:
`results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/mc3`

Planned MC3 campaign:

- 6 selected dataset/method blocks
- 5 seeds per block, seeds 42-46
- 30 total full runs

Completed before pause:

- EyePACS DR, official DINOv3 ConvNeXt LoRA, CE: 5/5 seeds complete.
- EyePACS DR, official DINOv3 ConvNeXt LoRA, cost-sensitive CE: seed 42
  complete; seed 43 was paused and is not counted as a scientific result.

Paused/pending:

- EyePACS cost-sensitive CE: seeds 43-46 still need full completed runs.
- EyePACS NICME hybrid: 5 seeds remain.
- PMI CE: 5 seeds remain.
- PMI cost-sensitive CE: 5 seeds remain.
- PMI NICME logit adjustment: 5 seeds remain.
- Total remaining full MC3 runs: 24.

MC3 completed metrics so far, argmax decision mode:

| Dataset | Method | Seeds | Target recall mean | Target recall sd | Norm ATC mean | Norm ATC sd | Balanced acc mean | Macro F1 mean | QWK mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EyePACS DR | CE | 5 | 0.7459 | 0.0644 | 0.1476 | 0.0850 | 0.3746 | 0.3413 | 0.4689 |
| EyePACS DR | cost-sensitive CE | 1 | 0.7838 | 0.0000 | 0.1182 | 0.0000 | 0.4108 | 0.3960 | 0.5641 |

The cost-sensitive CE row is only one completed seed, so it is a progress
signal, not a stable MC3 conclusion.

## Reboot-Avoidance State

The current pause was a normal user-requested interrupt, not a crash, CUDA
taint, D-state failure, or hard-reboot condition.

Hardening already implemented:

- Run-level preflight checks before GPU training.
- CUDA canary before paper-run training jobs.
- Conservative D-state process detection that ignores kernel threads.
- Completed runs are skipped before preflight/canary on resume.
- Retry-once for the specific PyTorch/PEFT module traversal failure observed
  in MC2, with a fresh checkpoint namespace.
- Single-GPU sequential training; no parallel GPU jobs.
- Storage-safe cache environment passed to subprocesses.

Recommended resume posture:

- Run a fresh health check before resuming MC3.
- Resume the selected MC3 launcher; completed rows should skip, and the
  paused cost-sensitive seed should re-run.
- Do not reboot unless the health check reports D-state user processes,
  GPU-taint signatures, or a failed CUDA canary that persists after normal
  process cleanup.

## Current Caveats

- Current paper-run testing is balanced-only. Natural splits are prepared but
  have not yet been included in the active MC3 campaign.
- MC3 is incomplete; final statistical comparisons, confidence intervals, and
  paired tests are not ready.
- Early MC2 evidence favors CE/cost-sensitive CE on recall, while NICME may
  offer cost/ATC advantages in some settings. The paper should report this
  without forcing a NICME-win interpretation if MC3 confirms it.
- The old memory snapshot in `memory/multiclass_2026_05_01` contains pre-FB
  DINOv3 recovery context and is partially superseded by the official
  Facebook DINOv3 LoRA MC-FB2 and MC2 artifacts.
