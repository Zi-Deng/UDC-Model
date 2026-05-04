# MC2 v2 Environment Stop Summary

MC2 v2 was launched with the stable MC1 backbones only:

- `timm_dinov3_vit_lora`
- `timm_dinov3_convnext_lora`

The first run, `0001_eyepacs_dr_balanced_timm_dinov3_vit_lora_ce_seed42`,
trained through epoch 4 and then failed with:

```text
torch.AcceleratorError: CUDA error: invalid resource handle
```

This happened after an earlier HF ConvNeXt repro left an unkillable Python
process in kernel `D` state while holding CUDA memory. A second timm run was
started by the sequential runner, but the campaign was stopped immediately
because the GPU context was no longer clean enough for paper-grade results.

Current blocker:

- GPU reset via `nvidia-smi --gpu-reset -i 0` was refused because the RTX 5090
  is the primary GPU.
- The D-state training processes cannot be killed from user space.

Required next action before restarting MC2:

- Reboot the host, restart the GPU driver, or otherwise clear the stuck
  D-state CUDA processes.

Hardening added after this stop:

- MC2/MC4 defaults now exclude the unstable HF ConvNeXt backend. It remains
  available only by explicitly passing `--models convnext`.
- The multiclass runner now performs a conservative GPU preflight before each
  training job. It refuses to start when D-state processes or a hung `nvidia-smi`
  probe indicate a tainted GPU/session.
- The runner now aborts the campaign after CUDA taint signatures such as
  `invalid resource handle` instead of continuing into subsequent configs.
- Training subprocesses are launched in their own process group, and timeouts
  terminate the process group before the next run is considered.

Post-reboot update:

- The reboot cleared the original HF ConvNeXt D-state processes and reduced GPU
  memory use back to idle levels.
- Re-running MC2 with PEFT LoRA backbones exposed a separate PEFT/timm failure:
  after epoch/eval/checkpoint cycling, `model.train()` failed with
  `TypeError: cannot unpack non-iterable Linear object`.
- Stopping that run left one Python training process in kernel `D` state
  (`PID 14433` at the time of this note), so GPU work is again blocked until a
  reboot or equivalent driver reset.
- To avoid both the HF ConvNeXt backend instability and the PEFT LoRA
  module-tree failure, MC2/MC3/MC4 defaults now use full fine-tuning for the
  non-PEFT timm DINOv3 backbones:
  `timm_dinov3_vit` and `timm_dinov3_convnext`.
- The paper-grade runner now aborts after any failed run by default. Passing
  `--continue-after-failure` is available for exploratory work only.

After the hardware state is clean, resume MC2 using:

```bash
micromamba run -n ml python scripts/run_multiclass_experiments.py \
  --stop mc2 \
  --datasets eyepacs_dr,pmi_pills \
  --variants balanced \
  --models timm_dinov3_vit,timm_dinov3_convnext \
  --output-dir results/multiclass_mc2_fulltune_20260501 \
  --execute \
  --per-run-timeout-minutes 60
```

Use a fresh output directory for the full-finetune campaign so the failed LoRA
rows remain auditable but are not mixed with the new stable plan.
