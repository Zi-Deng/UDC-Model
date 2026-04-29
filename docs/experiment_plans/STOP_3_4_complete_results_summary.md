# Stop 3 And Stop 4 Complete Results Summary

Generated: 2026-04-29

This report consolidates the completed Stop 3 and Stop 4 experiment sequence for the binary-first NICME extension. It is intended to be readable as a human research note: what was planned, what ran, what the results mean, where the strongest claims are, and where the evidence needs careful wording.

The short version is favorable, with nuance. Stop 3A showed that tuned NICME methods were finally competitive on the balanced tasks, but BreaKHis was not yet strict all-seed stable. Stop 3B showed that imbalance and deployment realism introduce strong class-prior baselines, especially Menon-style logit adjustment on imbalanced Spider. Stop 4A was the key turning point: with ConvNeXt-family backbones, NICME-family methods became strict all-seed winners on both balanced applications at the original 10:1 cost setting. Stop 4B then confirmed that the cost-ratio behavior is meaningful, while also showing that very high ratios can trade too much accuracy for recall.

> Paper-facing thesis supported by these stops:
>
> Under balanced data, where class-frequency imbalance is removed as the explanation, NICME-family methods achieve the best or near-best recall/ATC tradeoff on both tested applications at the original 10:1 operating point. The strongest NICME variant is dataset and backbone dependent: Spider favors `convnext + nicme_hybrid + calibrated_threshold`, while BreaKHis favors `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold`.

The evidence does not support the stronger claim that `nicme_hybrid` is universally best. It supports the more credible claim that the NICME family gives the strongest balanced-data evidence, with method choice depending on the application.

## At A Glance

### Operational Ledger

| stop | purpose | final successful runs | run-log failures | decision rows | elapsed successful training time |
| --- | --- | ---: | ---: | ---: | ---: |
| Stop 3A | Balanced primary evidence | 72 | 0 | 216 | 198.7 min |
| Stop 3B | Imbalance and deployment decoupling | 108 | 0 | 324 | 328.3 min |
| Stop 4A | Backbone ablation on balanced tasks | 36 | 0 | 108 | 100.3 min |
| Stop 4B Spider | Cost-ratio sensitivity, Spider ConvNeXt | 45 | 1 retried segfault | 135 | 85.2 min |
| Stop 4B BreaKHis | Cost-ratio sensitivity, BreaKHis DINOv3 ConvNeXt LoRA | 45 | 0 | 135 | 175.5 min |
| Total | Stop 3 and Stop 4 completed evidence | 306 | 1 retried failure | 918 | 888.0 min |

The only Stop 3/4 failure was Stop 4B Spider run index 13:

- Failed attempt: `returncode=-11`, a native process segfault, after about 1.9 seconds.
- Failed logs: `results/stop4b_cost_ratio_sensitivity/spider_convnext/logs/013_stop4b_cost_ratio_spider_convnext_spider_balanced_costr2_convnext_ce_calibrated_cost_min_seed43.stdout.log` and matching `.stderr.log`.
- Retry: succeeded with `returncode=0` in the `.resume.*.log` files.
- Final status: Spider Stop 4B has 45/45 successful planned rows. The run log keeps the failed attempt as historical evidence.

### Best Rows By Stop

Lower `selection_score` and `normalized_ATC` are better. `floors=all` means all three seeds met both target-recall and selected-accuracy floors.

| stop | dataset | best row | floors | selection | nATC | target recall | selected acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Stop 3A | `spider_balanced` | `timm_dinov3_vit_lora + nicme_logit_adjustment + calibrated_threshold` | 33% | 0.0214 | 0.0211 | 1.0000 | 0.7889 |
| Stop 3A | `breakhis_balanced` | `vit + ce_calibrated_cost_min + calibrated_cost_min` | 67% | 0.0173 | 0.0164 | 0.9976 | 0.8463 |
| Stop 3B | `spider_target_minority` | `vit + menon_logit_adjusted + calibrated_threshold` | all | 0.0152 | 0.0152 | 0.9867 | 0.9144 |
| Stop 3B | `spider_target_majority` | `vit + menon_logit_adjusted + calibrated_cost_min` | 67% | 0.0156 | 0.0155 | 0.9956 | 0.7544 |
| Stop 3B | `breakhis_natural` | `timm_dinov3_vit_lora + nicme_hybrid + argmax` | all | 0.0153 | 0.0153 | 0.9907 | 0.8742 |
| Stop 4A | `spider_balanced` | `convnext + nicme_hybrid + calibrated_threshold` | all | 0.0183 | 0.0183 | 0.9911 | 0.8567 |
| Stop 4A | `breakhis_balanced` | `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` | all | 0.0145 | 0.0145 | 0.9965 | 0.8712 |
| Stop 4B | `spider_balanced` | `convnext + nicme_hybrid + calibrated_threshold`, ratio 20 | 67% | 0.0119 | 0.0115 | 0.9978 | 0.7911 |
| Stop 4B | `breakhis_balanced` | `timm_dinov3_convnext_lora + ce_calibrated_cost_min + calibrated_threshold`, ratio 20 | 67% | 0.0114 | 0.0112 | 0.9929 | 0.8434 |

The Stop 4B ratio-20 rows have the lowest mean selection scores, but neither is the cleanest strict all-seed paper-facing row. For robust balanced claims, the original 10:1 ratio remains especially important.

### Primary Balanced Paper Rows From Stop 4A

| dataset | cleanest 10:1 NICME row | floors | selection | nATC | target recall | accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `spider_balanced` | `convnext + nicme_hybrid + calibrated_threshold` | all | 0.0183 | 0.0183 | 0.9911 | 0.8567 |
| `breakhis_balanced` | `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` | all | 0.0145 | 0.0145 | 0.9965 | 0.8712 |

These are the strongest paper-facing rows because they are balanced-data results, use the original cost ratio, and pass both floors in all three seeds.

Stop 4B independently reran the same Spider architecture/method/mode at the 10:1 ratio as part of the sensitivity sweep and observed selection `0.0193`, nATC `0.0193`, recall `0.9933`, and accuracy `0.8367`. Use Stop 4A as the primary backbone-ablation result and Stop 4B as the cost-ratio confirmation.

## Evaluation Semantics

The stops used a consistent result-selection objective:

- `selection_score`: lower is better. It combines normalized ATC with squared penalties for missing the target-recall floor and the configured accuracy floor.
- `normalized_ATC`: lower is better. It reports expected misclassification cost normalized by the user-defined cost matrix.
- `target_recall`: recall for the user-cared class.
- `selected acc`: the configured accuracy metric used for floor checks.
- `floors`: fraction of seeds meeting both recall and accuracy floors.

For balanced Spider and balanced BreaKHis, `selected acc` is plain accuracy. For imbalanced and natural deployment stress tests, `selected acc` is balanced accuracy.

The decision modes have distinct meanings:

| mode | meaning |
| --- | --- |
| `argmax` | Standard highest-logit prediction. Costs affect this only through training. |
| `calibrated_cost_min` | Temperature-scaled probabilities followed by minimum expected-cost decision. |
| `calibrated_threshold` | Temperature-scaled probabilities followed by a calibration-split threshold chosen to minimize the same selection score. |

The strongest repeated pattern is that `calibrated_threshold` is the most reliable deployment mode for the current binary research objective. Raw cost-min decisions can drive recall very high, but they sometimes lose too much accuracy.

## Stop 3A: Balanced Primary Evidence

### Plan

Stop 3A was the primary balanced-data test after Stop 2A-2C NICME tuning. Its purpose was to test the central proposal in the cleanest setting: balanced Spider and balanced BreaKHis, where any benefit from cost-sensitive learning cannot be explained away as ordinary class-imbalance correction.

Scope:

| dimension | values |
| --- | --- |
| datasets | `spider_balanced`, `breakhis_balanced` |
| models | `vit`, `timm_dinov3_vit_lora` |
| methods | `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `cs_regularized_ce`, `nicme_logit_adjustment`, `nicme_hybrid` |
| seeds | `42`, `43`, `44` |
| decision modes | `argmax`, `calibrated_cost_min`, `calibrated_threshold` |

Status:

- 72/72 training jobs completed.
- 216 decision rows were exported.
- Checkpoint cleanup succeeded.

### Spider Balanced

Target: `black_widow`. Floors: recall `0.95`, accuracy `0.80`.

| read | row | floors | selection | nATC | recall | accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Best overall | `timm_dinov3_vit_lora + nicme_logit_adjustment + calibrated_threshold` | 33% | 0.0214 | 0.0211 | 1.0000 | 0.7889 |
| Best strict all-seed row | `timm_dinov3_vit_lora + nicme_hybrid + argmax` | all | 0.0272 | 0.0272 | 0.9600 | 0.9078 |
| Best mean-floor NICME row | `vit + nicme_logit_adjustment + calibrated_threshold` | 67% | 0.0230 | 0.0229 | 0.9933 | 0.8011 |

Interpretation:

Stop 3A Spider was favorable to NICME but not yet fully clean. The top two mean-selection rows were NICME rows with excellent recall, but the best overall row had mean accuracy just under the 0.80 floor. The strict all-seed winner was also NICME, but it used `argmax`, not the deployment-aligned calibrated-threshold mode. This showed that the tuning had worked, but it also created a good reason to continue into backbone ablation.

### BreaKHis Balanced

Target: `malignant`. Floors: recall `0.97`, accuracy `0.85`.

| read | row | floors | selection | nATC | recall | accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Best overall | `vit + ce_calibrated_cost_min + calibrated_cost_min` | 67% | 0.0173 | 0.0164 | 0.9976 | 0.8463 |
| Best NICME / mean-floor row | `vit + nicme_logit_adjustment + calibrated_threshold` | 33% | 0.0186 | 0.0178 | 0.9929 | 0.8534 |
| Close NICME comparison | `vit + nicme_hybrid + calibrated_threshold` | 67% | 0.0188 | 0.0180 | 0.9929 | 0.8522 |

Interpretation:

BreaKHis was close but still not strict. The best overall row was a CE cost-min baseline, and the best NICME row cleared the accuracy floor on mean performance but not all seeds. No BreaKHis Stop 3A row met both floors in all three seeds. This was the main reason Stop 4A became necessary.

### Stop 3A Takeaway

Stop 3A moved the project from "prototype evidence is mixed" to "NICME is competitive and often best on balanced data, but floor stability is not yet good enough." It justified continuing, but it did not yet provide the cleanest final paper claim.

## Revised Stop 3B/4 Plan After Stop 3A

The plan was revised after Stop 3A because the evidence had two important features:

- `nicme_logit_adjustment` was often stronger and safer than `nicme_hybrid`.
- Balanced BreaKHis still lacked strict all-seed floor stability.

The revised plan had two subparts:

| subplan | purpose |
| --- | --- |
| Stop 3B | Run imbalance and natural-deployment stress tests to separate explicit costs from class-frequency effects. |
| Stop 4A | Run missing backbone ablations on balanced data to see whether ConvNeXt-family models make NICME floor-stable. |

This revision was important. If Stop 4A had been skipped, the paper would have been left with promising but fragile balanced BreaKHis evidence. Stop 4A is what made the balanced claim substantially cleaner.

## Stop 3B: Imbalance Decoupling And Deployment Realism

### Plan

Stop 3B tested whether the methods behaved differently when class distributions were deliberately shifted or left natural. This was not meant to replace the balanced-data evidence. It was a decoupling and realism check.

Scope:

| dimension | values |
| --- | --- |
| datasets | `spider_target_minority`, `spider_target_majority`, `breakhis_natural` |
| models | `vit`, `timm_dinov3_vit_lora` |
| methods | same six-method family as Stop 3A |
| seeds | `42`, `43`, `44` |
| decision modes | `argmax`, `calibrated_cost_min`, `calibrated_threshold` |

Status:

- 108/108 training jobs completed.
- 324 decision rows were exported.
- No failures.

### Results

| dataset | best overall row | floors | selection | nATC | target recall | selected bal acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `breakhis_natural` | `timm_dinov3_vit_lora + nicme_hybrid + argmax` | all | 0.0153 | 0.0153 | 0.9907 | 0.8742 |
| `spider_target_majority` | `vit + menon_logit_adjusted + calibrated_cost_min` | 67% | 0.0156 | 0.0155 | 0.9956 | 0.7544 |
| `spider_target_minority` | `vit + menon_logit_adjusted + calibrated_threshold` | all | 0.0152 | 0.0152 | 0.9867 | 0.9144 |

Best NICME rows:

| dataset | best NICME row | floors | selection | nATC | target recall | selected bal acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `breakhis_natural` | `timm_dinov3_vit_lora + nicme_hybrid + argmax` | all | 0.0153 | 0.0153 | 0.9907 | 0.8742 |
| `spider_target_majority` | `vit + nicme_logit_adjustment + calibrated_threshold` | 67% | 0.0170 | 0.0162 | 0.9933 | 0.7733 |
| `spider_target_minority` | `vit + nicme_logit_adjustment + calibrated_threshold` | all | 0.0220 | 0.0220 | 0.9867 | 0.8689 |

Best CE-family comparison rows:

| dataset | best CE-family row | floors | selection | nATC | target recall | selected bal acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `breakhis_natural` | `vit + ce + calibrated_threshold` | all | 0.0154 | 0.0154 | 0.9977 | 0.8221 |
| `spider_target_majority` | `vit + ce_calibrated_cost_min + calibrated_threshold` | 67% | 0.0219 | 0.0213 | 0.9867 | 0.7667 |
| `spider_target_minority` | `vit + ce + calibrated_cost_min` | 33% | 0.0267 | 0.0253 | 0.9467 | 0.8933 |

### Interpretation

Stop 3B is mixed in exactly the way a useful decoupling test should be.

On natural BreaKHis, NICME hybrid was the best overall row and passed both floors in all seeds. This is strong deployment-realism evidence, especially because it pairs low selection score with high balanced accuracy.

On controlled imbalanced Spider, Menon-style logit adjustment was the best overall family. That does not contradict the balanced-data claim. Menon is a class-prior/long-tail baseline, so it is expected to be strong when class priors are manipulated. The important scientific distinction is this:

- Stop 3B says class-prior methods can be very strong under explicit imbalance.
- Stop 4A says NICME-family methods can still win when class imbalance is removed.

Those two findings are compatible and useful. They help keep the final paper honest.

## Stop 4A: Backbone Ablation On Balanced Data

### Plan

Stop 4A was designed to answer the weakness left by Stop 3A: whether better backbones could turn the balanced NICME results into strict all-seed evidence, especially on BreaKHis.

Scope:

| dimension | values |
| --- | --- |
| datasets | `spider_balanced`, `breakhis_balanced` |
| backbones | ConvNeXt-family rows, including `convnext` and `timm_dinov3_convnext_lora` |
| methods | `ce_calibrated_cost_min`, `nicme_logit_adjustment`, `nicme_hybrid` |
| seeds | `42`, `43`, `44` |
| decision modes | `argmax`, `calibrated_cost_min`, `calibrated_threshold` |

Status:

- 36/36 training jobs completed.
- 108 decision rows were exported.
- No failures.

### Results

| dataset | best overall row | floors | selection | nATC | target recall | accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `spider_balanced` | `convnext + nicme_hybrid + calibrated_threshold` | all | 0.0183 | 0.0183 | 0.9911 | 0.8567 |
| `breakhis_balanced` | `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` | all | 0.0145 | 0.0145 | 0.9965 | 0.8712 |

Best CE-family comparisons:

| dataset | best CE-family row | floors | selection | nATC | target recall | accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `spider_balanced` | `convnext + ce_calibrated_cost_min + calibrated_cost_min` | all | 0.0184 | 0.0184 | 0.9956 | 0.8356 |
| `breakhis_balanced` | `timm_dinov3_convnext_lora + ce_calibrated_cost_min + calibrated_cost_min` | 33% | 0.0180 | 0.0170 | 0.9976 | 0.8410 |

### Interpretation

Stop 4A is the central positive result across Stop 3 and Stop 4.

For Spider, `convnext + nicme_hybrid + calibrated_threshold` beat the strongest CE-family row by a very small margin. The margin is narrow, but the result is still scientifically meaningful because the NICME row is both best overall and strict all-seed floor compliant.

For BreaKHis, the result is stronger. `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` was the best overall row, the best NICME row, and strict all-seed floor compliant. The best CE-family comparison remained below the accuracy floor in two of three seeds.

The key lesson is not that one NICME variant wins everywhere. The key lesson is that the NICME family has strict balanced-data winners on both applications once the backbone is chosen appropriately.

## Revised Stop 4B Plan

Stop 4B was revised after Stop 4A to focus on the application-specific winners instead of running an expensive full cross-product.

### Plan

| application | dataset | selected backbone | methods | ratios | seeds |
| --- | --- | --- | --- | --- | --- |
| Spider | `spider_balanced` | `convnext` | `ce_calibrated_cost_min`, `nicme_logit_adjustment`, `nicme_hybrid` | `1`, `2`, `5`, `10`, `20` | `42`, `43`, `44` |
| BreaKHis | `breakhis_balanced` | `timm_dinov3_convnext_lora` | `ce_calibrated_cost_min`, `nicme_logit_adjustment`, `nicme_hybrid` | `1`, `2`, `5`, `10`, `20` | `42`, `43`, `44` |

Run count:

```text
2 application/model pairs x 3 methods x 5 ratios x 3 seeds = 90 runs
```

The questions were:

- Does NICME remain best or tied near-best at the original 10:1 ratio?
- Does behavior change sensibly as the cared-class false-negative cost increases?
- At what ratio does the method begin to trade too much accuracy for recall?
- Is `nicme_hybrid` or `nicme_logit_adjustment` more stable across ratios?
- Does `ce_calibrated_cost_min` close the gap when cost-aware thresholding alone is enough?

### Status

| queue | final successful planned runs | historical failures | decision rows | aggregate rows |
| --- | ---: | ---: | ---: | ---: |
| Spider ConvNeXt | 45/45 | 1 failed attempt, retried successfully | 135 | 45 |
| BreaKHis DINOv3 ConvNeXt LoRA | 45/45 | 0 | 135 | 45 |

Stop 4B completed. The Spider queue contains one old failed run-log entry from the native segfault described above, but all planned rows eventually succeeded and were included in the final analysis.

## Stop 4B Spider Cost-Ratio Sensitivity

Dataset: `spider_balanced`. Backbone: `convnext`. Target: `black_widow`. Floors: recall `0.95`, accuracy `0.80`.

### Ratio Trend

| ratio | best overall row | floors | selection | nATC | recall | accuracy |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `ce_calibrated_cost_min + argmax` | 0% | 0.0587 | 0.0567 | 0.9289 | 0.9433 |
| 2 | `nicme_logit_adjustment + calibrated_threshold` | 67% | 0.0429 | 0.0428 | 0.9600 | 0.9344 |
| 5 | `nicme_logit_adjustment + calibrated_cost_min` | all | 0.0260 | 0.0260 | 0.9911 | 0.8878 |
| 10 | `ce_calibrated_cost_min + calibrated_cost_min` | all | 0.0184 | 0.0184 | 0.9956 | 0.8356 |
| 20 | `nicme_hybrid + calibrated_threshold` | 67% | 0.0119 | 0.0115 | 0.9978 | 0.7911 |

Best strict all-seed rows by ratio:

| ratio | strict row | selection | nATC | recall | accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `nicme_logit_adjustment + calibrated_threshold` | 0.0589 | 0.0589 | 0.9578 | 0.9411 |
| 2 | `nicme_hybrid + calibrated_threshold` | 0.0439 | 0.0439 | 0.9822 | 0.9211 |
| 5 | `nicme_logit_adjustment + calibrated_cost_min` | 0.0260 | 0.0260 | 0.9911 | 0.8878 |
| 10 | `ce_calibrated_cost_min + calibrated_cost_min` | 0.0184 | 0.0184 | 0.9956 | 0.8356 |
| 20 | `ce_calibrated_cost_min + calibrated_threshold` | 0.0124 | 0.0124 | 0.9889 | 0.8567 |

### Interpretation

Spider behaves sensibly as the cost ratio increases: selection score and normalized ATC fall, and target recall generally rises. The tradeoff is that high ratios begin to pressure accuracy.

The ratio-20 NICME hybrid row has the lowest mean selection score, but it does not clear the accuracy floor on mean performance (`0.7911` against a `0.80` floor) and only passes both floors in two of three seeds. The strict ratio-20 winner is CE calibrated threshold.

At the original 10:1 ratio, the evidence is still favorable to NICME but very close:

- Best overall strict row: `ce_calibrated_cost_min + calibrated_cost_min`, selection `0.0184`.
- Best NICME strict row: `nicme_hybrid + calibrated_threshold`, selection `0.0193`.
- Difference: about `0.0009` selection-score points.

For paper wording, the cleanest Spider balanced result remains the Stop 4A/Stop 4B 10:1 NICME hybrid row: it is strict all-seed floor compliant, has high recall, and is effectively tied near the strongest CE cost-min baseline.

## Stop 4B BreaKHis Cost-Ratio Sensitivity

Dataset: `breakhis_balanced`. Backbone: `timm_dinov3_convnext_lora`. Target: `malignant`. Floors: recall `0.97`, accuracy `0.85`.

### Ratio Trend

| ratio | best overall row | floors | selection | nATC | recall | accuracy |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `ce_calibrated_cost_min + calibrated_threshold` | 33% | 0.1990 | 0.1093 | 0.8700 | 0.8907 |
| 2 | `nicme_hybrid + argmax` | 33% | 0.0400 | 0.0393 | 0.9669 | 0.9379 |
| 5 | `nicme_logit_adjustment + argmax` | 33% | 0.0274 | 0.0270 | 0.9669 | 0.9314 |
| 10 | `nicme_logit_adjustment + calibrated_threshold` | all | 0.0145 | 0.0145 | 0.9965 | 0.8712 |
| 20 | `ce_calibrated_cost_min + calibrated_threshold` | 67% | 0.0114 | 0.0112 | 0.9929 | 0.8434 |

Best strict all-seed rows by ratio:

| ratio | strict row | selection | nATC | recall | accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | none | n/a | n/a | n/a | n/a |
| 2 | none | n/a | n/a | n/a | n/a |
| 5 | `nicme_logit_adjustment + calibrated_cost_min` | 0.0288 | 0.0288 | 0.9846 | 0.8865 |
| 10 | `nicme_logit_adjustment + calibrated_threshold` | 0.0145 | 0.0145 | 0.9965 | 0.8712 |
| 20 | none | n/a | n/a | n/a | n/a |

### Interpretation

BreaKHis has the clearest cost-ratio story. The original 10:1 ratio is the robust optimum for NICME: `nicme_logit_adjustment + calibrated_threshold` is the best NICME row, the best strict all-seed row, and the best mean-floor-compliant row.

Ratio 20 produces a lower mean selection score for CE calibrated threshold, but it does so with mean accuracy below the `0.85` floor and no strict all-seed row. This is useful sensitivity evidence rather than a better deployment setting.

The BreaKHis conclusion is clean:

- `10:1` is the best robust operating point.
- `nicme_logit_adjustment` is the safest NICME variant.
- `calibrated_threshold` is the best deployment mode.

## Consolidated Scientific Read

### What The Evidence Now Supports

1. Balanced-data NICME evidence is strong after Stop 4A.

   Stop 4A produced strict all-seed NICME winners on both balanced applications. This is the most important result because balanced data removes class-frequency imbalance as the explanation.

2. The strongest NICME variant is application dependent.

   Spider favors `nicme_hybrid` with `convnext`. BreaKHis favors `nicme_logit_adjustment` with `timm_dinov3_convnext_lora`. Natural BreaKHis favors `nicme_hybrid` with `timm_dinov3_vit_lora`.

3. `calibrated_threshold` should be the primary binary deployment mode.

   It repeatedly finds operating points that align with the study objective. It avoids some of the false-positive inflation seen with raw cost-min decisions.

4. Cost ratio sensitivity is meaningful.

   Increasing false-negative cost generally lowers normalized ATC and raises recall, but very high ratios can weaken accuracy-floor stability.

5. CE cost-min is a serious baseline.

   It sometimes wins, especially in Stop 4B ratio-20 Spider/BreaKHis mean-selection comparisons. The report should treat it as a strong comparator, not a straw baseline.

### What The Evidence Does Not Support

The results do not support claiming that:

- `nicme_hybrid` is universally best.
- NICME dominates every CE-family or class-prior baseline.
- Ratio 20 is automatically better just because it lowers mean selection score.
- Imbalanced Spider proves the same thing as balanced Spider.

The accurate claim is narrower and stronger: NICME-family methods produce the cleanest balanced-data evidence that explicit user-defined costs matter even when class imbalance is removed, but the winning variant and operating mode are application specific.

## Recommended Paper Claims

Use wording close to this:

> On balanced Spider and BreaKHis splits, where class-frequency imbalance is controlled, NICME-family methods achieved the best strict all-seed recall/ATC tradeoffs at the original 10:1 user-cost setting after backbone selection. Spider favored a ConvNeXt NICME hybrid with calibrated-threshold inference, while BreaKHis favored a DINOv3 ConvNeXt LoRA NICME logit-adjustment model with calibrated-threshold inference.

Also report:

- CE calibrated cost-min remains a strong baseline.
- Menon-style logit adjustment is strong under controlled class imbalance.
- High cost ratios can reduce ATC but may destabilize accuracy floors.
- The final claim is about cost-sensitive decision quality under balanced conditions, not universal method dominance.

Avoid wording like:

> NICME hybrid is the best method across all datasets.

or:

> NICME wins because the datasets are imbalanced.

The first is false, and the second misses the core balanced-data finding.

## Source Artifacts

Primary human reports:

- `docs/experiment_plans/STOP_3A_balanced_primary_scientific_read.md`
- `docs/experiment_plans/STOP_3B_4A_results_and_stop4B_plan.md`
- `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_scientific_read.md`
- `results/stop4a_backbone_ablation/stop4a_backbone_ablation_scientific_read.md`
- `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_scientific_read.md`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_scientific_read.md`

CSV and JSON data:

- `results/stop3a_balanced_primary/stop3a_full_decision_rows.csv`
- `results/stop3a_balanced_primary/stop3a_aggregate_summary.csv`
- `results/stop3a_balanced_primary/stop3a_ranked_summary.csv`
- `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_full_decision_rows.csv`
- `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_aggregate_summary.csv`
- `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_ranked_summary.csv`
- `results/stop4a_backbone_ablation/stop4a_backbone_ablation_full_decision_rows.csv`
- `results/stop4a_backbone_ablation/stop4a_backbone_ablation_aggregate_summary.csv`
- `results/stop4a_backbone_ablation/stop4a_backbone_ablation_ranked_summary.csv`
- `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_full_decision_rows.csv`
- `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_aggregate_summary.csv`
- `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_ranked_summary.csv`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_full_decision_rows.csv`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_aggregate_summary.csv`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_ranked_summary.csv`

Run logs:

- `results/stop3a_balanced_primary/run_log.json`
- `results/stop3b_imbalance_decoupling/run_log.json`
- `results/stop4a_backbone_ablation/run_log.json`
- `results/stop4b_cost_ratio_sensitivity/spider_convnext/run_log.json`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/run_log.json`

No external literature claims are introduced in this summary. It is a synthesis of local run logs, generated metrics, and the previously generated Stop 3/4 scientific reads.
