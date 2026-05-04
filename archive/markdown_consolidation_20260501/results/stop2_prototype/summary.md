# Stop 2 Prototype Aggregate Summary

- Run status: 32/32 successful rows.
- Total clean-run wall time recorded by runner: 982.04 seconds (16.37 minutes).
- Mean successful-row time: 30.69 seconds.
- Prototype splits: 256 train / 64 validation / 64 calibration / 64 test per dataset, class-balanced in every split.
- BreaKHis prototype remained patient-disjoint with zero missing images.
- Official Meta `facebook/dinov3-*` checkpoints remain shelved pending Hugging Face gated approval; this run used accessible ConvNeXt, ViT, timm DINOv3 ViT LoRA, and timm DINOv3 ConvNeXt MLP-LoRA.

## Best Rows By Selection Score

| dataset | model | method | mode | nATC | target recall | acc | bal acc | macro-F1 | selection | AUROC | AUPRC | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breakhis_balanced_prototype | convnext | ce | argmax | 0.0203 | 0.9688 | 0.9375 | 0.9375 | 0.9374 | 0.0203 | 0.9541 | 0.9527 | 0.2577 |  |
| breakhis_balanced_prototype | convnext | ce_calibrated_cost_min | argmax | 0.0203 | 0.9688 | 0.9375 | 0.9375 | 0.9374 | 0.0203 | 0.9541 | 0.9527 | 0.2577 |  |
| breakhis_balanced_prototype | convnext | menon_logit_adjusted | argmax | 0.0203 | 0.9688 | 0.9375 | 0.9375 | 0.9374 | 0.0203 | 0.9541 | 0.9527 | 0.2577 |  |
| breakhis_balanced_prototype | vit | nicme_hybrid | argmax | 0.0234 | 0.9688 | 0.9062 | 0.9062 | 0.9059 | 0.0234 | 0.9756 | 0.9845 | 0.1706 |  |
| breakhis_balanced_prototype | vit | menon_logit_adjusted | argmax | 0.0469 | 0.9062 | 0.9531 | 0.9531 | 0.9530 | 0.0631 | 0.9619 | 0.9770 | 0.2855 |  |
| breakhis_balanced_prototype | vit | ce | argmax | 0.0469 | 0.9062 | 0.9531 | 0.9531 | 0.9530 | 0.0631 | 0.9619 | 0.9770 | 0.2855 |  |
| spider_balanced_prototype | vit | ce_calibrated_cost_min | calibrated_cost_min | 0.0422 | 1.0000 | 0.5781 | 0.5781 | 0.4868 | 0.0914 | 0.9443 | 0.9571 | 0.1213 | 0.5000 |
| spider_balanced_prototype | timm_dinov3_convnext_lora | menon_logit_adjusted | calibrated_cost_min | 0.0563 | 0.9688 | 0.5781 | 0.5781 | 0.5022 | 0.1055 | 0.7686 | 0.7734 | 0.1672 | 0.7000 |
| spider_balanced_prototype | timm_dinov3_convnext_lora | ce | calibrated_cost_min | 0.0563 | 0.9688 | 0.5781 | 0.5781 | 0.5022 | 0.1055 | 0.7686 | 0.7734 | 0.1672 | 0.7000 |
| spider_balanced_prototype | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_cost_min | 0.0563 | 0.9688 | 0.5781 | 0.5781 | 0.5022 | 0.1055 | 0.7686 | 0.7734 | 0.1672 | 0.7000 |
| spider_balanced_prototype | vit | menon_logit_adjusted | calibrated_cost_min | 0.0453 | 1.0000 | 0.5469 | 0.5469 | 0.4298 | 0.1094 | 0.9424 | 0.9548 | 0.1296 | 0.5000 |
| spider_balanced_prototype | timm_dinov3_convnext_lora | nicme_hybrid | argmax | 0.0594 | 0.9688 | 0.5469 | 0.5469 | 0.4488 | 0.1234 | 0.6357 | 0.6551 | 0.2522 |  |

## Best Row Per Dataset And Model

| dataset | model | method | mode | nATC | target recall | acc | bal acc | macro-F1 | selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breakhis_balanced_prototype | convnext | ce | argmax | 0.0203 | 0.9688 | 0.9375 | 0.9375 | 0.9374 | 0.0203 |
| breakhis_balanced_prototype | vit | nicme_hybrid | argmax | 0.0234 | 0.9688 | 0.9062 | 0.9062 | 0.9059 | 0.0234 |
| spider_balanced_prototype | vit | ce_calibrated_cost_min | calibrated_cost_min | 0.0422 | 1.0000 | 0.5781 | 0.5781 | 0.4868 | 0.0914 |
| spider_balanced_prototype | timm_dinov3_convnext_lora | menon_logit_adjusted | calibrated_cost_min | 0.0563 | 0.9688 | 0.5781 | 0.5781 | 0.5022 | 0.1055 |
| spider_balanced_prototype | convnext | menon_logit_adjusted | calibrated_cost_min | 0.0500 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 |
| spider_balanced_prototype | timm_dinov3_vit_lora | ce | calibrated_cost_min | 0.0500 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_cost_min | 0.0578 | 0.9688 | 0.5625 | 0.5625 | 0.4760 | 0.1405 |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | nicme_hybrid | calibrated_cost_min | 0.0500 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 |

## Best Available Row Compared With CE Argmax For The Same Dataset And Model

| dataset | model | method | mode | nATC | delta nATC | target recall | delta recall | acc | delta acc | selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breakhis_balanced_prototype | convnext | ce | argmax | 0.0203 | 0.0000 | 0.9688 | 0.0000 | 0.9375 | 0.0000 | 0.0203 |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | ce | calibrated_cost_min | 0.0578 | -0.0781 | 0.9688 | 0.2188 | 0.5625 | -0.2031 | 0.1405 |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | ce | calibrated_cost_min | 0.0500 | -0.1172 | 1.0000 | 0.3125 | 0.5000 | -0.2344 | 0.1725 |
| breakhis_balanced_prototype | vit | nicme_hybrid | argmax | 0.0234 | -0.0234 | 0.9688 | 0.0625 | 0.9062 | -0.0469 | 0.0234 |
| spider_balanced_prototype | convnext | ce | calibrated_cost_min | 0.0500 | -0.1281 | 1.0000 | 0.3438 | 0.5000 | -0.2656 | 0.1400 |
| spider_balanced_prototype | timm_dinov3_convnext_lora | ce | calibrated_cost_min | 0.0563 | -0.1703 | 0.9688 | 0.4062 | 0.5781 | -0.1250 | 0.1055 |
| spider_balanced_prototype | timm_dinov3_vit_lora | ce | calibrated_cost_min | 0.0500 | -0.0875 | 1.0000 | 0.2500 | 0.5000 | -0.2500 | 0.1400 |
| spider_balanced_prototype | vit | ce_calibrated_cost_min | calibrated_cost_min | 0.0422 | -0.1031 | 1.0000 | 0.2812 | 0.5781 | -0.2344 | 0.0914 |

## Interpretation Notes

- Operationally, Stop 2 passed: all 32 configured prototype runs completed and exported metrics under the one-hour target.
- Scientifically, the signal is mixed. Cost-sensitive calibrated-cost-min inference often improves cared-class recall and normalized ATC, but several prototype rows do so by reducing accuracy toward roughly 0.50 to 0.58.
- BreaKHis ConvNeXt CE argmax was already strong on this small balanced prototype subset, so improvements there should be judged against a high CE baseline rather than against a weak model.
- `vit` + `nicme_hybrid` on BreaKHis is the cleanest prototype example of a training-time NICME gain with reasonable accuracy.
- timm DINOv3 LoRA paths are operationally viable for Stop 3, but their prototype metrics do not yet justify treating them as uniformly superior.
- Stop 3 should therefore preserve CE argmax and CE calibrated-cost-min as strong baselines, include all six methods, and make accuracy-floor violations explicit in the paper tables.
