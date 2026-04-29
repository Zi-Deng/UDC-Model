# Stop 3B And Stop 4A Results, Interpretation, And Revised Stop 4B Plan

Generated: 2026-04-28

## Status

Stop 3B and Stop 4A completed successfully.

| phase | planned | completed | failures | elapsed training time | primary artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| Stop 3B imbalance decoupling | 108 | 108 | 0 | 328.3 min | `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_scientific_read.md` |
| Stop 4A backbone ablation | 36 | 36 | 0 | 100.3 min | `results/stop4a_backbone_ablation/stop4a_backbone_ablation_scientific_read.md` |

Storage remained healthy after checkpoint cleanup:

| location | size/free |
| --- | ---: |
| filesystem free space | 127G |
| `checkpoints/` | 3.1G |
| `results/stop3b_imbalance_decoupling/` | 11M |
| `results/stop4a_backbone_ablation/` | 2.8M |

## Result Summary

Lower `selection_score` and `normalized_ATC` are better. `selected acc` is the configured selection accuracy metric: plain accuracy for balanced datasets and balanced accuracy for imbalanced/natural datasets. Floors are strict when all three seeds pass both target-recall and selected-accuracy floors.

| phase | dataset | best overall row | floors | selection | nATC | target recall | selected acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Stop 3B | `breakhis_natural` | `timm_dinov3_vit_lora + nicme_hybrid + argmax` | all | 0.0153 | 0.0153 | 0.9907 | 0.8742 |
| Stop 3B | `spider_target_majority` | `vit + menon_logit_adjusted + calibrated_cost_min` | 67% | 0.0156 | 0.0155 | 0.9956 | 0.7544 |
| Stop 3B | `spider_target_minority` | `vit + menon_logit_adjusted + calibrated_threshold` | all | 0.0152 | 0.0152 | 0.9867 | 0.9144 |
| Stop 4A | `breakhis_balanced` | `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` | all | 0.0145 | 0.0145 | 0.9965 | 0.8712 |
| Stop 4A | `spider_balanced` | `convnext + nicme_hybrid + calibrated_threshold` | all | 0.0183 | 0.0183 | 0.9911 | 0.8567 |

Best NICME rows:

| phase | dataset | best NICME row | floors | selection | nATC | target recall | selected acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Stop 3B | `breakhis_natural` | `timm_dinov3_vit_lora + nicme_hybrid + argmax` | all | 0.0153 | 0.0153 | 0.9907 | 0.8742 |
| Stop 3B | `spider_target_majority` | `vit + nicme_logit_adjustment + calibrated_threshold` | 67% | 0.0170 | 0.0162 | 0.9933 | 0.7733 |
| Stop 3B | `spider_target_minority` | `vit + nicme_logit_adjustment + calibrated_threshold` | all | 0.0220 | 0.0220 | 0.9867 | 0.8689 |
| Stop 4A | `breakhis_balanced` | `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` | all | 0.0145 | 0.0145 | 0.9965 | 0.8712 |
| Stop 4A | `spider_balanced` | `convnext + nicme_hybrid + calibrated_threshold` | all | 0.0183 | 0.0183 | 0.9911 | 0.8567 |

Best CE-family comparison rows:

| phase | dataset | best CE-family row | floors | selection | nATC | target recall | selected acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Stop 3B | `breakhis_natural` | `vit + ce + calibrated_threshold` | all | 0.0154 | 0.0154 | 0.9977 | 0.8221 |
| Stop 3B | `spider_target_majority` | `vit + ce_calibrated_cost_min + calibrated_threshold` | 67% | 0.0219 | 0.0213 | 0.9867 | 0.7667 |
| Stop 3B | `spider_target_minority` | `vit + ce + calibrated_cost_min` | 33% | 0.0267 | 0.0253 | 0.9467 | 0.8933 |
| Stop 4A | `breakhis_balanced` | `timm_dinov3_convnext_lora + ce_calibrated_cost_min + calibrated_cost_min` | 33% | 0.0180 | 0.0170 | 0.9976 | 0.8410 |
| Stop 4A | `spider_balanced` | `convnext + ce_calibrated_cost_min + calibrated_cost_min` | all | 0.0184 | 0.0184 | 0.9956 | 0.8356 |

## Scientific Interpretation

### Alignment With The Goal

The updated evidence is favorable to the project goal, with important nuance.

On the balanced datasets, which are the central evidence for decoupling cost sensitivity from class imbalance, Stop 4A produced strict all-seed NICME winners for both applications:

- `spider_balanced`: `convnext + nicme_hybrid + calibrated_threshold` is the best overall row, with mean target recall `0.9911`, mean normalized ATC `0.0183`, and mean accuracy `0.8567`.
- `breakhis_balanced`: `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` is the best overall row, with mean malignant recall `0.9965`, mean normalized ATC `0.0145`, and mean accuracy `0.8712`.

This materially strengthens the paper claim compared with Stop 2 and Stop 3A. Stop 3A already showed tuned NICME was competitive on balanced data, but BreaKHis lacked strict all-seed floor stability. Stop 4A resolves that for the tested ConvNeXt-family backbone.

### What Is Strong

- NICME is no longer merely competitive on balanced data. In Stop 4A, NICME is the best overall family on both balanced applications.
- `calibrated_threshold` remains the most reliable binary operating mode for this research objective. It directly optimizes the same selection score on the calibration split, whereas raw `calibrated_cost_min` sometimes overcorrects toward the cared class and loses accuracy.
- The best NICME member is application-dependent. Spider favors `nicme_hybrid` with standard ConvNeXt, while BreaKHis favors `nicme_logit_adjustment` with timm DINOv3 ConvNeXt LoRA.

### What Is Weak Or Needs Care

- The best imbalanced Spider rows are Menon-style logit adjustment rows. That is not a contradiction of the proposal because Menon is an imbalance/long-tail baseline and the central claim depends on balanced data. It does mean the paper should be honest: on controlled class imbalance, class-prior methods can be very strong.
- `nicme_hybrid` is not globally the best proposed method. It is excellent for balanced Spider and natural BreaKHis with timm DINOv3 ViT LoRA, but `nicme_logit_adjustment` is stronger on balanced BreaKHis and controlled imbalanced Spider.
- The cost-min Bayes decision baseline is competitive but can be too aggressive under a 10:1 matrix. This supports reporting it as a baseline, not replacing threshold-tuned deployment evaluation with it.

### Current Paper-Facing Read

The cleanest current claim is:

> Under balanced data, where class-frequency imbalance is removed as an explanation, NICME-family methods achieve the best recall/ATC tradeoff on both tested applications. The strongest NICME variant is dataset and backbone dependent: hybrid training is best for balanced Spider with ConvNeXt, while logit adjustment is best for balanced BreaKHis with timm DINOv3 ConvNeXt LoRA.

The claim should not be:

> NICME hybrid is universally best.

The evidence does not support that stronger wording yet.

## Revised Stop 4B Plan

Stop 4B should now test cost-ratio sensitivity using the best observed application-specific backbone rather than spending the next several hours on every backbone cross-product.

### Rationale

The original Stop 4B plan left model choice open until Stop 3B and Stop 4A completed. Stop 4A supplied the missing backbone evidence:

- Spider best balanced model: `convnext`
- BreaKHis best balanced model: `timm_dinov3_convnext_lora`

Therefore, Stop 4B Phase 1 will focus on those application-specific winners. This keeps the run under the one-day RTX 5090 target while directly testing whether NICME responds smoothly to explicit user-defined costs.

### Stop 4B Phase 1 Scope

Datasets:

- `spider_balanced`
- `breakhis_balanced`

Application-specific model assignment:

- Spider: `convnext`
- BreaKHis: `timm_dinov3_convnext_lora`

Methods:

- `ce_calibrated_cost_min`
- `nicme_logit_adjustment`
- `nicme_hybrid`

Cost ratios:

- `1:1`
- `2:1`
- `5:1`
- `10:1`
- `20:1`

Seeds:

- `42,43,44`

Run count:

- `2 application/model pairs x 3 methods x 5 ratios x 3 seeds = 90 runs`

Expected runtime:

- Stop 4A took `100.3` minutes for 36 runs.
- The focused Stop 4B Phase 1 queue is projected around `4` to `5` hours, with checkpoint cleanup enabled.

### Planned Launch

Spider queue:

```bash
micromamba run -n ml python scripts/run_stop4b_cost_ratio.py \
  --phase stop4b_cost_ratio_spider_convnext \
  --output-root results/stop4b_cost_ratio_sensitivity/spider_convnext \
  --datasets spider_balanced \
  --models convnext \
  --methods ce_calibrated_cost_min,nicme_logit_adjustment,nicme_hybrid \
  --ratios 1,2,5,10,20 \
  --seeds 42,43,44 \
  --cleanup-checkpoints \
  --execute
```

BreaKHis queue:

```bash
micromamba run -n ml python scripts/run_stop4b_cost_ratio.py \
  --phase stop4b_cost_ratio_breakhis_dinov3_convnext_lora \
  --output-root results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora \
  --datasets breakhis_balanced \
  --models timm_dinov3_convnext_lora \
  --methods ce_calibrated_cost_min,nicme_logit_adjustment,nicme_hybrid \
  --ratios 1,2,5,10,20 \
  --seeds 42,43,44 \
  --cleanup-checkpoints \
  --execute
```

These will be launched sequentially in one tmux session so the GPU remains occupied without running competing training processes.

### Live Execution Note

Stop 4B Phase 1 was launched in tmux after this analysis.

- session: `nicme_stop4b_cost_ratio`
- chain stdout: `results/stop4b_cost_ratio_sensitivity/chain.stdout.log`
- chain stderr: `results/stop4b_cost_ratio_sensitivity/chain.stderr.log`
- spider output root: `results/stop4b_cost_ratio_sensitivity/spider_convnext`
- BreaKHis output root: `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora`
- first active run: `stop4b_cost_ratio_spider_convnext_spider_balanced_costr1_convnext_ce_calibrated_cost_min_seed42`

### Stop 4B Success Criteria

For each application, Stop 4B should answer:

- Does NICME remain best or tied near-best at the original `10:1` ratio?
- Does NICME improve monotonically or at least sensibly as the cared-class false-negative cost increases?
- At what ratio does the method begin to trade too much accuracy for recall?
- Is `nicme_hybrid` or `nicme_logit_adjustment` more stable across ratios?
- Does `ce_calibrated_cost_min` close the gap when the cost matrix alone is enough to set a good deployment threshold?

## Source Artifacts

This report uses local generated artifacts only:

- `results/stop3a_balanced_primary/stop3a_aggregate_summary.csv`
- `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_aggregate_summary.csv`
- `results/stop4a_backbone_ablation/stop4a_backbone_ablation_aggregate_summary.csv`
- `docs/experiment_plans/STOP_3A_balanced_primary_scientific_read.md`
- `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_scientific_read.md`
- `results/stop4a_backbone_ablation/stop4a_backbone_ablation_scientific_read.md`

No new external literature claims are introduced in this memo.
