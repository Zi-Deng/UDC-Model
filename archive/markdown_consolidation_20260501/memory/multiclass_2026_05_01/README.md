# NICME Multiclass State Snapshot - 2026-05-01

This folder preserves the current state of the multiclass NICME extension after
MC0/MC1 completion, MC2 launch attempts, failure analysis, and runner
hardening. It is intended as handoff memory for future Codex sessions.

The most important scientific constraint is that **LoRA remains a key component
of the paper claim**. The temporary non-PEFT full-finetuning fallback that was
patched during emergency debugging must not silently replace the LoRA-centered
paper plan without explicit user approval. The correct next engineering step is
to fix or replace the PEFT/timm LoRA integration, then rerun a multi-epoch LoRA
stability gate before restarting MC2.

## Current Status

- Repository target: `/mnt/storage/github/NICME`.
- MC0 data/cost readiness: complete for EyePACS DR and PMI Pills.
- MC1 scientific smoke: complete for both target datasets before MC2.
- MC2: not complete. Attempts were stopped due to infrastructure/model-wrapper
  failures, not scientific underperformance.
- MC3: not started. It must wait for completed MC2 selection.
- Current GPU state at last check: one Python training process was stuck in
  kernel `D` state after the PEFT/timm LoRA failure. A reboot or equivalent
  driver reset is required before further GPU training.

## Implemented Multiclass Infrastructure

Implemented or substantially extended:

- `nicme/dataset_profiles.py`
  - Adds fixed multiclass profiles for `eyepacs_dr` and `pmi_pills`.
  - Stores class names, cost matrices, cared classes, metric profile metadata,
    critical PMI pairs, balanced/natural split selection rules, and cost-matrix
    SHA-256 hashes.
- `nicme/costs.py`
  - Multiclass cost validation and `C[true][pred]` orientation handling.
- `nicme/data_prep.py`
  - EyePACS and PMI preparation, auditing, balanced split generation, and cost
    matrix propagation.
- `nicme/cli.py`
  - Adds/extends `nicme-prepare-data` support for multiclass profiles.
- `scripts/check_multiclass_readiness.py`
  - Readiness checks for prepared data, cost matrices, and audit artifacts.
- `scripts/run_multiclass_experiments.py`
  - MC1/MC2/MC3/MC4 runner with per-run configs, ledgers, logs, retries,
    fresh checkpoint namespaces, GPU preflight, and fail-closed behavior.
- `scripts/select_mc2_configs.py`
  - MC2 selector for recall-first model choice with normalized ATC tie-break.
- `scripts/train.py`
  - Multiclass metric selection support, dataset max-sample limits, optional
    cuDNN disabling, and contiguous float logits before custom losses.
- `utils/utils.py`
  - Multiclass metrics, calibration/cost-min outputs, optional visualization
    skipping, safer confusion-matrix handling, and artifact export additions.
- Tests:
  - Multiclass cost validation, calibration/cost-min behavior, dataset
    profiles, data-prep checks, and related smoke/unit coverage.

## Data State

EyePACS:

- Primary source: Kaggle diabetic-retinopathy-detection training data.
- Prepared under `data/prepared/eyepacs_dr`.
- Uses ordinal grades `0..4`.
- Fixed cost matrix from Galdran-style squared ordinal distance:
  `[[0,1,4,9,16],[1,0,1,4,9],[4,1,0,1,4],[9,4,1,0,1],[16,9,4,1,0]]`.
- Primary cared class: `DR4`; secondary severe set: `DR3,DR4`.
- Balanced split used for current testing.

PMI Pills:

- Manual archive provided at `/mnt/storage/medication_images.zip`.
- README provided at `/mnt/storage/image_dataset_readme.txt`.
- Hard-linked/copied into `data/raw/pmi`.
- SHA1 verified for `medication_images.zip`:
  `df15ceb6aa039d71d23774258e28acc89bd31b91`.
- Extracted official `NLM20/{train,valid,test}` under `data/raw/pmi/NLM20`.
- Prepared under `data/prepared/pmi_pills`.
- Balanced split counts: train 1940, validation 320, calibration 300, test 640.
- PMI cost matrix:
  - diagonal `0`;
  - non-critical off-diagonal `1`;
  - critical overrides:
    - `50111-0434 -> 00591-0461 = 10`
    - `53489-0156 -> 68382-0227 = 10`
    - `53746-0544 -> 00378-0208 = 10`
    - `68382-0227 -> 53489-0156 = 8`

Important readiness artifacts:

- `results/multiclass_pmi_ready_20260430/mc0/mc0_readiness.md`
- `results/multiclass_pmi_ready_20260430/pmi_preparation_summary.md`
- `results/multiclass_mc1_hardened_v2_20260430/mc1/mc1_smoke_summary.md`
- `results/multiclass_mc1_pmi_hardened_v2_20260430/mc1/mc1_pmi_smoke_summary.md`

## MC1 Status

MC1 passed for both target datasets.

EyePACS MC1:

- Summary:
  `results/multiclass_mc1_hardened_v2_20260430/mc1/mc1_smoke_summary.md`
- Ledger:
  `results/multiclass_mc1_hardened_v2_20260430/mc1/run_ledger.csv`

PMI MC1:

- Summary:
  `results/multiclass_mc1_pmi_hardened_v2_20260430/mc1/mc1_pmi_smoke_summary.md`
- Ledger:
  `results/multiclass_mc1_pmi_hardened_v2_20260430/mc1/run_ledger.csv`
- All 6 scientific smoke configs completed after retry hardening.

## MC2 Attempt History

First MC2 attempt:

- Output root: `results/multiclass_mc2_20260430`.
- Included HF `facebook/convnext-tiny-224`.
- Failed with a PyTorch autograd internal assert:
  `isDifferentiableType(grad.scalar_type()) INTERNAL ASSERT FAILED`.
- A focused HF ConvNeXt repro then froze while holding CUDA memory.
- HF ConvNeXt was quarantined.
- Note:
  `results/multiclass_mc2_convnext_ce_repro_20260430/mc2/hf_convnext_quarantine_note.md`.

Second MC2 attempt:

- Output root: `results/multiclass_mc2_v2_20260430`.
- Used timm DINOv3 LoRA backbones:
  `timm_dinov3_vit_lora`, `timm_dinov3_convnext_lora`.
- Before reboot, failed with:
  `CUDA error: invalid resource handle`.
- Cause: GPU/session contamination from the stuck HF ConvNeXt process.
- Reboot cleared the original stuck processes.

Post-reboot LoRA attempt:

- Re-ran `timm_dinov3_vit_lora`.
- The run passed early checkpoints/evals but then failed around epoch/checkpoint
  cycling with:
  `TypeError: cannot unpack non-iterable Linear object`
  inside `torch.nn.Module.train()` after PEFT/timm wrapping.
- Stopping this run left a new Python training process in kernel `D` state.
- Current status note:
  `results/multiclass_mc2_v2_20260430/mc2/mc2_environment_stop_summary.md`.

## Current Important Caveat

During emergency hardening, MC2/MC3/MC4 defaults were temporarily switched to
non-PEFT timm DINOv3 full-finetuning:

- `timm_dinov3_vit`
- `timm_dinov3_convnext`

This is **not scientifically acceptable as a replacement for LoRA** unless the
user explicitly approves changing the paper claim. Treat it as a diagnostic
fallback only. The correct paper-grade path is to preserve LoRA and fix the
PEFT/timm integration.

## Recommended Next Plan

1. Reboot or reset the GPU/driver before any additional GPU work.

2. Restore LoRA as the primary MC2/MC3 path:
   - MC2/MC3 defaults should use:
     - `timm_dinov3_vit_lora`
     - `timm_dinov3_convnext_lora`
   - Keep non-PEFT `timm_dinov3_vit` and `timm_dinov3_convnext` as ablation or
     diagnostic baselines, not as the main paper path.

3. Fix LoRA implementation:
   - Investigate `nicme/modeling.py::apply_lora_if_requested`.
   - Current PEFT wrapping applies LoRA to the outer
     `TimmForImageClassification` wrapper and uses
     `peft_modules_to_save="model.head"`.
   - Likely safer design: apply PEFT to the underlying timm model before
     wrapping, avoid fragile dotted `modules_to_save` on the wrapper, and keep
     the classification head trainable manually.
   - Add tests that call repeated `model.train()`, `model.eval()`, checkpoint
     save/reload if possible, and forward/backward on CPU for timm LoRA models.

4. Add a new LoRA stability gate before MC2:
   - A one-epoch MC1 smoke is insufficient; the PEFT bug appears after repeated
     train/eval/checkpoint cycles.
   - Run a 4-5 epoch small/subset LoRA stability smoke for both DINOv3 timm LoRA
     backbones, at least on EyePACS and preferably PMI too.
   - Require no D-state processes, no CUDA taint, successful train/eval/export,
     and finite metrics before MC2.

5. Relaunch MC2 only after the LoRA stability gate passes:

```bash
micromamba run -n ml python scripts/run_multiclass_experiments.py \
  --stop mc2 \
  --datasets eyepacs_dr,pmi_pills \
  --variants balanced \
  --models timm_dinov3_vit_lora,timm_dinov3_convnext_lora \
  --output-dir results/multiclass_mc2_lora_fixed_20260501 \
  --execute \
  --per-run-timeout-minutes 60
```

6. Run MC2 selection:

```bash
micromamba run -n ml python scripts/select_mc2_configs.py \
  --ledger results/multiclass_mc2_lora_fixed_20260501/mc2/run_ledger.csv \
  --output-dir results/multiclass_mc2_lora_fixed_20260501/selection
```

7. Launch MC3 only from selected MC2 configs.

## Guardrails To Preserve

- Do not optimize dataset cost matrices.
- Use fixed cost-matrix source `dataset`.
- Keep `C[true][pred]` orientation.
- Report both argmax and calibrated cost-min decision behavior where relevant.
- Reject binary-only calibrated-threshold paths for multiclass.
- Always use balanced classes for current MC testing unless the user changes
  that instruction.
- Abort MC2/MC3 after any failed run unless explicitly doing exploratory work.
- Always run GPU preflight before training:
  - no D-state processes;
  - `nvidia-smi` responsive;
  - low idle GPU memory;
  - no active train/runners.

## Verification Already Run

Before the latest interruption:

- `ruff check scripts/run_multiclass_experiments.py scripts/train.py` passed.
- Targeted tests passed:
  `tests/test_costs.py tests/test_calibration.py tests/test_dataset_profiles.py`.
- A full test pass was previously achieved after PMI prep:
  `36 passed`.

## Current Blocker

Do not launch more GPU training in the current boot if a Python process remains
in kernel `D` state. At last check after the LoRA failure, one process like
`PID 14433 python D` remained. Reboot/reset before continuing.
