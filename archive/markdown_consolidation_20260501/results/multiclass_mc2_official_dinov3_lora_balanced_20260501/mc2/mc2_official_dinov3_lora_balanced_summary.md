# MC2 Official DINOv3 LoRA Balanced Summary

Date: 2026-05-01 UTC

## Status

MC2 completed for balanced EyePACS DR and balanced PMI Pills using official
Facebook DINOv3 LoRA backbones.

- Planned runs: 20
- Completed runs: 20
- Failed final runs: 0
- Retry recoveries: 1
- Storage preflight: passed, all large cache paths under `/mnt/storage`
- Post-run D-state check: clean
- Post-run GPU taint check: clean
- Post-run CUDA canary: clean
- Regression tests after runner patch: `49 passed`

## No-Reboot Recovery

Run `0017_pmi_pills_balanced_facebook_dinov3_convnext_lora_menon_logit_adjusted_seed42`
failed once with a PyTorch/PEFT module traversal traceback:

`ValueError: not enough values to unpack (expected 2, got 1)`

This was not a CUDA taint or D-state failure. A synthetic reproduction using
the same official DINOv3 ConvNeXt LoRA model and Menon loss completed cleanly.
The runner was patched to retry this narrow traceback once with a fresh
checkpoint namespace. The resumed MC2 run recovered row 17 on retry attempt 2
and completed rows 18-20 without rebooting.

## MC2 Metrics

| Dataset | Model | Method | Status | Attempts | Target recall | Norm ATC | Balanced acc | Macro F1 | Expected cost |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| eyepacs_dr | facebook_dinov3_vit_lora | ce | completed | 1 | 0.4730 | 0.4162 | 0.1622 | 0.1065 | 6.6595 |
| eyepacs_dr | facebook_dinov3_vit_lora | menon_logit_adjusted | completed | 1 | 0.4730 | 0.4162 | 0.1622 | 0.1065 | 6.6595 |
| eyepacs_dr | facebook_dinov3_vit_lora | cs_regularized_ce | completed | 1 | 0.4730 | 0.4176 | 0.1541 | 0.1006 | 6.6811 |
| eyepacs_dr | facebook_dinov3_vit_lora | nicme_logit_adjustment | completed | 1 | 0.2297 | 0.1503 | 0.2541 | 0.2107 | 2.4054 |
| eyepacs_dr | facebook_dinov3_vit_lora | nicme_hybrid | completed | 1 | 0.2568 | 0.3563 | 0.1946 | 0.1595 | 5.7000 |
| eyepacs_dr | facebook_dinov3_convnext_lora | ce | completed | 1 | 0.7432 | 0.1199 | 0.4243 | 0.4020 | 1.9189 |
| eyepacs_dr | facebook_dinov3_convnext_lora | menon_logit_adjusted | completed | 1 | 0.7432 | 0.1199 | 0.4243 | 0.4020 | 1.9189 |
| eyepacs_dr | facebook_dinov3_convnext_lora | cs_regularized_ce | completed | 1 | 0.7568 | 0.1174 | 0.4432 | 0.4226 | 1.8784 |
| eyepacs_dr | facebook_dinov3_convnext_lora | nicme_logit_adjustment | completed | 1 | 0.5811 | 0.1034 | 0.4216 | 0.4204 | 1.6541 |
| eyepacs_dr | facebook_dinov3_convnext_lora | nicme_hybrid | completed | 1 | 0.6892 | 0.1225 | 0.4108 | 0.4087 | 1.9595 |
| pmi_pills | facebook_dinov3_vit_lora | ce | completed | 1 | 0.0000 | 0.0917 | 0.0828 | 0.0256 | 0.9172 |
| pmi_pills | facebook_dinov3_vit_lora | menon_logit_adjusted | completed | 1 | 0.0000 | 0.0917 | 0.0828 | 0.0256 | 0.9172 |
| pmi_pills | facebook_dinov3_vit_lora | cs_regularized_ce | completed | 1 | 0.0000 | 0.0917 | 0.0828 | 0.0256 | 0.9172 |
| pmi_pills | facebook_dinov3_vit_lora | nicme_logit_adjustment | completed | 1 | 0.0000 | 0.0923 | 0.0766 | 0.0229 | 0.9234 |
| pmi_pills | facebook_dinov3_vit_lora | nicme_hybrid | completed | 1 | 0.0000 | 0.0944 | 0.0703 | 0.0245 | 0.9437 |
| pmi_pills | facebook_dinov3_convnext_lora | ce | completed | 1 | 0.4062 | 0.0505 | 0.7078 | 0.7027 | 0.5047 |
| pmi_pills | facebook_dinov3_convnext_lora | menon_logit_adjusted | completed | 2 | 0.4062 | 0.0505 | 0.7078 | 0.7027 | 0.5047 |
| pmi_pills | facebook_dinov3_convnext_lora | cs_regularized_ce | completed | 1 | 0.4062 | 0.0503 | 0.7094 | 0.7039 | 0.5031 |
| pmi_pills | facebook_dinov3_convnext_lora | nicme_logit_adjustment | completed | 1 | 0.3750 | 0.0348 | 0.7562 | 0.7513 | 0.3484 |
| pmi_pills | facebook_dinov3_convnext_lora | nicme_hybrid | completed | 1 | 0.0000 | 0.0958 | 0.1406 | 0.0956 | 0.9578 |

## Selection

Selection artifacts:

- `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc2_selection.md`
- `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc2_selection.json`

Selected MC3 candidates:

- EyePACS DR: ConvNeXt LoRA `ce`, `cs_regularized_ce`, `nicme_hybrid`
- PMI Pills: ConvNeXt LoRA `ce`, `cs_regularized_ce`, `nicme_logit_adjustment`

## Interpretation

MC2 does not support claiming that NICME is already the top recall method in
this balanced prototype. On both datasets, ConvNeXt LoRA CE or
cost-sensitive CE leads the recall-first objective. NICME logit adjustment is
still scientifically worth carrying into MC3 for PMI because it substantially
reduces ATC while keeping competitive, though lower, cared-class recall and
higher balanced accuracy. For EyePACS, NICME hybrid is the strongest NICME
candidate but trails the CE/cost-sensitive CE recall baselines.
