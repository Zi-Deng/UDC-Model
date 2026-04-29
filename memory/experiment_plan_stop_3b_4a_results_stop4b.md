# Stop 3B And Stop 4A Results, Interpretation, And Revised Stop 4B Plan

Generated: 2026-04-28

Stop 3B and Stop 4A completed successfully:

- Stop 3B imbalance decoupling: `108/108` runs complete, `0` failures, `328.3` min logged training time.
- Stop 4A backbone ablation: `36/36` runs complete, `0` failures, `100.3` min logged training time.
- Disk remained healthy with about `127G` free and `checkpoints/` around `3.1G`.

## Key Scientific Read

Balanced-dataset evidence is now stronger than after Stop 3A. Stop 4A produced strict all-seed NICME winners for both balanced applications:

- `spider_balanced`: best overall row is `convnext + nicme_hybrid + calibrated_threshold`, with selection `0.0183`, normalized ATC `0.0183`, target recall `0.9911`, and accuracy `0.8567`.
- `breakhis_balanced`: best overall row is `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold`, with selection `0.0145`, normalized ATC `0.0145`, malignant recall `0.9965`, and accuracy `0.8712`.

Stop 3B also completed cleanly:

- `breakhis_natural`: best overall and best NICME row is `timm_dinov3_vit_lora + nicme_hybrid + argmax`, selection `0.0153`, normalized ATC `0.0153`, malignant recall `0.9907`, selected balanced accuracy `0.8742`.
- `spider_target_majority`: best overall row is Menon, but best NICME row is `vit + nicme_logit_adjustment + calibrated_threshold`, selection `0.0170`, normalized ATC `0.0162`, recall `0.9933`, selected balanced accuracy `0.7733`.
- `spider_target_minority`: best overall row is Menon, but best NICME row is `vit + nicme_logit_adjustment + calibrated_threshold`, selection `0.0220`, normalized ATC `0.0220`, recall `0.9867`, selected balanced accuracy `0.8689`.

Interpretation:

- NICME-family methods now support the central balanced-data claim: explicit costs can help even when class imbalance is removed.
- Menon-style logit adjustment remains strong under controlled imbalance, which should be reported honestly as an imbalance-baseline result.
- `calibrated_threshold` remains the most useful binary deployment mode for optimizing the actual recall/ATC/accuracy objective.
- The proposed method should be framed as the NICME family, not only `nicme_hybrid`: Spider favors hybrid with ConvNeXt, while BreaKHis favors logit adjustment with timm DINOv3 ConvNeXt LoRA.

## Revised Stop 4B

Stop 4B Phase 1 should be focused rather than all-cross-product:

- Spider model: `convnext`
- BreaKHis model: `timm_dinov3_convnext_lora`
- Methods: `ce_calibrated_cost_min`, `nicme_logit_adjustment`, `nicme_hybrid`
- Ratios: `1,2,5,10,20`
- Seeds: `42,43,44`
- Run count: `90`
- Expected runtime: around `4` to `5` hours on the RTX 5090 with checkpoint cleanup.

Launch roots:

- `results/stop4b_cost_ratio_sensitivity/spider_convnext`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora`

Stop 4B Phase 1 was launched after this analysis:

- tmux session: `nicme_stop4b_cost_ratio`
- chain stdout: `results/stop4b_cost_ratio_sensitivity/chain.stdout.log`
- chain stderr: `results/stop4b_cost_ratio_sensitivity/chain.stderr.log`
- first active run: `stop4b_cost_ratio_spider_convnext_spider_balanced_costr1_convnext_ce_calibrated_cost_min_seed42`

The corresponding docs artifact is:

- `docs/experiment_plans/STOP_3B_4A_results_and_stop4B_plan.md`
