# LoRA Recovery Plan For Multiclass MC2/MC3

LoRA must remain part of the main paper-grade plan. The current non-PEFT
full-finetuning path is a diagnostic fallback only.

## Failure To Fix

The post-reboot LoRA run used `timm_dinov3_vit_lora` and failed after repeated
epoch/eval/checkpoint cycling with:

```text
TypeError: cannot unpack non-iterable Linear object
```

The traceback entered `torch.nn.Module.train()` through the PEFT-wrapped timm
model. This suggests the PEFT wrapping/modules-to-save interaction corrupted or
confused the module tree after checkpoint/eval/save cycling.

## Current Suspect

`nicme/modeling.py::apply_lora_if_requested` currently applies PEFT to the
outer `TimmForImageClassification` wrapper. The presets use:

```text
peft_target_modules="qkv"
peft_modules_to_save="model.head"
```

For timm models, this dotted `modules_to_save` on a wrapper appears fragile.

## Preferred Fix Direction

1. Refactor timm LoRA construction:
   - create the timm base model;
   - apply PEFT directly to the timm model, not to the outer wrapper;
   - avoid `modules_to_save="model.head"` on the wrapper;
   - keep the timm head trainable explicitly with `requires_grad=True`;
   - then wrap with `TimmForImageClassification`.

2. Add a targeted test:
   - build `timm_dinov3_vit_lora` with `timm_pretrained=False`;
   - verify LoRA parameters are trainable;
   - verify head parameters are trainable;
   - run repeated `model.train(); model.eval(); model.train()` cycles;
   - run a tiny forward/backward pass on CPU;
   - optionally test state dict save/load if feasible.

3. Add a checkpoint-cycle smoke:
   - small balanced subset;
   - at least 4-5 epochs;
   - both `timm_dinov3_vit_lora` and `timm_dinov3_convnext_lora`;
   - at least CE and NICME hybrid;
   - checkpoint every epoch, evaluate every epoch.

4. Only after the LoRA stability smoke passes:
   - run MC2 with LoRA models;
   - run MC2 selection;
   - launch MC3.

## Avoid For Now

- Do not make full-finetuning the paper-grade replacement for LoRA without
  explicit user approval.
- Do not re-enable HF ConvNeXt as a default MC2/MC3 model.
- Do not launch GPU training while any process is in kernel `D` state.

## Possible Stable Command After Fix

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
