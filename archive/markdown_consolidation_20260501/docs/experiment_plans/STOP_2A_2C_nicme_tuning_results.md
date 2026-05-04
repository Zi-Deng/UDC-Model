# Stop 2A-2C NICME Tuning Results

Generated: 2026-04-28

## Status

The stop-gated paper plan remains paused after Stop 2. These runs were a calibration/model-adjustment phase intended to improve NICME before any Stop 3 paper-scale experiments.

## Implementation Changes

- Added `nicme_logit_cost_scale`, default `1.0`, to scale raw cost values inside NICME logit-adjustment losses without changing existing configs.
- Added binary `calibrated_threshold` inference mode. This fits temperature on the calibration split, then selects a target-class probability threshold on the calibration split by minimizing the same NICME selection score used elsewhere.
- Kept `argmax` and `calibrated_cost_min` unchanged.
- Removed generated checkpoints after each tuning run; metrics and logs were retained.

## Run Summary

| phase | runs | passed | seconds | minutes |
| --- | --- | --- | --- | --- |
| Stop 2A NICME scale/lambda grid | 84 | 84 | 2722.9230 | 45.3821 |
| Stop 2B threshold calibration rerun | 13 | 13 | 403.6633 | 6.7277 |
| Stop 2C NICME frontier pass | 22 | 22 | 665.8368 | 11.0973 |

Total tuning runtime was 63.21 minutes across three sequential prototype passes. Each individual pass stayed small enough for prototype work, and all generated checkpoints were cleaned after metrics export.

## Best Comparison Across Stop 2 And Tuning Passes

| dataset | candidate | phase | model | method | mode | scale | lambda | threshold | nATC | target recall | acc | selection | floors met | confusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breakhis_balanced_prototype | best_any | stop2a_nicme_tuning | vit | nicme_hybrid | argmax | 0.2000 | 0.2500 |  | 0.0187 | 0.9688 | 0.9531 | 0.0188 | False | [[30, 2], [1, 31]] |
| breakhis_balanced_prototype | best_nicme | stop2c_nicme_frontier | vit | nicme_hybrid | argmax | 0.3000 | 0.2500 |  | 0.0187 | 0.9688 | 0.9531 | 0.0188 | False | [[30, 2], [1, 31]] |
| breakhis_balanced_prototype | best_nicme_argmax | stop2b_threshold_tuning | vit | nicme_hybrid | argmax | 0.0500 | 0.1000 |  | 0.0187 | 0.9688 | 0.9531 | 0.0188 | False | [[30, 2], [1, 31]] |
| breakhis_balanced_prototype | best_nicme_threshold | stop2b_threshold_tuning | vit | nicme_hybrid | calibrated_threshold | 0.0500 | 0.1000 | 0.5076 | 0.0187 | 0.9688 | 0.9531 | 0.0188 | False | [[30, 2], [1, 31]] |
| breakhis_balanced_prototype | best_ce | stop2 | convnext | ce | argmax |  |  |  | 0.0203 | 0.9688 | 0.9375 | 0.0203 |  |  |
| breakhis_balanced_prototype | best_ce_threshold | stop2b_threshold_tuning | convnext | ce | calibrated_threshold | 1.0000 | 10.0000 | 0.2846 | 0.0281 | 0.9688 | 0.8594 | 0.0281 | False | [[24, 8], [1, 31]] |
| spider_balanced_prototype | best_any | stop2b_threshold_tuning | vit | ce | calibrated_threshold | 1.0000 | 10.0000 | 0.3614 | 0.0312 | 0.9688 | 0.8281 | 0.0312 | True | [[31, 1], [10, 22]] |
| spider_balanced_prototype | best_any_floor_met | stop2b_threshold_tuning | vit | ce | calibrated_threshold | 1.0000 | 10.0000 | 0.3614 | 0.0312 | 0.9688 | 0.8281 | 0.0312 | True | [[31, 1], [10, 22]] |
| spider_balanced_prototype | best_nicme | stop2c_nicme_frontier | vit | nicme_logit_adjustment | calibrated_threshold | 0.8000 | 0.0000 | 0.4659 | 0.0406 | 0.9688 | 0.7344 | 0.0449 | False | [[31, 1], [16, 16]] |
| spider_balanced_prototype | best_nicme_argmax | stop2b_threshold_tuning | vit | nicme_logit_adjustment | argmax | 0.5000 | 0.0000 |  | 0.0484 | 0.9375 | 0.7969 | 0.0491 | False | [[30, 2], [11, 21]] |
| spider_balanced_prototype | best_nicme_threshold | stop2c_nicme_frontier | vit | nicme_logit_adjustment | calibrated_threshold | 0.8000 | 0.0000 | 0.4659 | 0.0406 | 0.9688 | 0.7344 | 0.0449 | False | [[31, 1], [16, 16]] |
| spider_balanced_prototype | best_ce | stop2b_threshold_tuning | vit | ce | calibrated_threshold | 1.0000 | 10.0000 | 0.3614 | 0.0312 | 0.9688 | 0.8281 | 0.0312 | True | [[31, 1], [10, 22]] |
| spider_balanced_prototype | best_ce_threshold | stop2b_threshold_tuning | vit | ce | calibrated_threshold | 1.0000 | 10.0000 | 0.3614 | 0.0312 | 0.9688 | 0.8281 | 0.0312 | True | [[31, 1], [10, 22]] |

## Main Findings

### Tumor / BreaKHis

Tuned NICME is currently the best observed model by selection score on the BreaKHis prototype.

Best NICME configuration:

- Model: ViT
- Method: `nicme_hybrid`
- NICME logit cost scale: `0.05` to `0.20` all tied in this prototype
- CS lambda: `0.10` or `0.25`
- Best mode: `argmax` or `calibrated_threshold`, tied
- Confusion matrix: `[[30, 2], [1, 31]]`
- normalized ATC: `0.01875`
- malignant recall: `0.96875`
- accuracy: `0.953125`
- selection score: `0.018756`

This narrowly improves over the best Stop 2 CE/ConvNeXt baseline: normalized ATC `0.020313`, malignant recall `0.96875`, accuracy `0.9375`, selection score `0.020319`.

Caveat: the configured BreaKHis recall floor is `0.97`, and the prototype test split has 32 malignant examples. That means `31/32 = 0.96875` narrowly misses the floor; the next attainable recall is `32/32 = 1.0`. No non-degenerate tuned row reached `32/32` malignant recall while preserving high accuracy.

### Widow / Spider

NICME improved substantially after tuning, but it did **not** become the best Spider model in this prototype.

Best overall Spider row:

- Model: ViT
- Method: CE
- Mode: `calibrated_threshold`
- Threshold: `0.361413`
- Confusion matrix: `[[31, 1], [10, 22]]`
- normalized ATC: `0.03125`
- black-widow recall: `0.96875`
- accuracy: `0.828125`
- selection score: `0.03125`
- floors met: `True`

Best Spider NICME row:

- Model: ViT
- Method: `nicme_logit_adjustment`
- NICME logit cost scale: `0.80`
- Mode: `calibrated_threshold`
- Threshold: `0.465916`
- Confusion matrix: `[[31, 1], [16, 16]]`
- normalized ATC: `0.040625`
- black-widow recall: `0.96875`
- accuracy: `0.734375`
- selection score: `0.044932`
- floors met: `False`

Best Spider NICME argmax row:

- Model: ViT
- Method: `nicme_logit_adjustment`
- NICME logit cost scale: `0.50` or `0.80`
- Confusion matrix: `[[30, 2], [11, 21]]`
- normalized ATC: `0.048438`
- black-widow recall: `0.9375`
- accuracy: `0.796875`
- selection score: `0.049072`

This means the current prototype cannot honestly claim NICME is best for widow classification. The best CE threshold row both beats NICME and satisfies the configured floors.

## Interpretation

Lowering NICME logit adjustment from raw `10.0` to a scale around `0.5-0.8` fixed much of the all-target-class collapse on Spider. For BreaKHis, a softer hybrid setting around scale `0.05-0.20` and lambda `0.10-0.25` improved over the CE baseline.

The Spider failure is not a calibration-only problem. The CE ViT model has a better operating point after threshold calibration than tuned NICME. NICME is moving recall upward, but it is degrading the non-target class too much, which raises false positives and lowers accuracy.

## Recommended Next Adjustment

Do not proceed to Stop 3 yet if the requirement is that NICME be best for both applications.

Next work should focus on Spider-specific NICME improvements:

1. Add a constrained validation objective that can select checkpoints using a calibrated threshold policy, not only argmax validation metrics.
2. Add a less blunt NICME loss variant that penalizes high-cost false negatives without globally over-penalizing the false-widow class; likely candidates are margin-capped logit adjustment, normalized logit adjustment, or asymmetric target-class regularization.
3. Run the next Spider tuning pass on a larger balanced split or multiple seeds, because this 64-image test split is too coarse for one-sample boundary conclusions.
4. Keep the tuned BreaKHis ViT `nicme_hybrid` setting as the current tumor candidate, but retest it on a larger split because it misses the recall floor by exactly one malignant case.

## Artifacts

- `results/stop2a_nicme_tuning/full_results.csv`
- `results/stop2b_threshold_tuning/full_results.csv`
- `results/stop2c_nicme_frontier/full_results.csv`
- `results/stop2c_nicme_frontier/combined_stop2_all_tuning.csv`
- `results/stop2c_nicme_frontier/final_best_comparison.csv`
