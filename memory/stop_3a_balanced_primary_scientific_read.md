# Stop 3A Balanced Primary Results And Scientific Read

Generated: 2026-04-28

## Executive Summary

Stop 3A completed the balanced-dataset primary queue. This is the most important first test of the paper claim because balanced Spider and balanced BreaKHis decouple explicit user-defined misclassification costs from class-frequency imbalance.

- Training jobs: `72/72` completed successfully.
- Metric files: `72`.
- Decision rows: `216` across `argmax`, `calibrated_cost_min`, and `calibrated_threshold`.
- Seeds: `42, 43, 44`.
- Datasets: `spider_balanced`, `breakhis_balanced`.
- Models: full-finetuned ViT and timm DINOv3 ViT-S LoRA.
- Checkpoint cleanup succeeded; generated run checkpoints were removed after metric export.

The short scientific read is mixed but much better than Stop 2:

- On `spider_balanced`, NICME is now the top-ranked family by mean selection score. The top overall row is `timm_dinov3_vit_lora + nicme_logit_adjustment + calibrated_threshold`, but it misses the strict `0.80` accuracy floor on average. The best strict all-seed floor-satisfying row is `timm_dinov3_vit_lora + nicme_hybrid + argmax`; the best mean-floor-compliant calibrated row is `vit + nicme_logit_adjustment + calibrated_threshold`.
- On `breakhis_balanced`, `vit + ce_calibrated_cost_min + calibrated_cost_min` has the lowest mean selection score, but it misses the `0.85` accuracy floor on average. The best mean-floor-compliant row is `vit + nicme_logit_adjustment + calibrated_threshold`; however, no BreaKHis aggregate row met both floors in all three seeds, so this remains promising but not yet strict all-seed evidence.
- The strongest repeated pattern is that calibrated-threshold operating points are better aligned with the project goal than raw Bayes cost-min decisions, because they can preserve recall while controlling false positives and accuracy collapse.

## Artifacts

- Full decision rows: `results/stop3a_balanced_primary/stop3a_full_decision_rows.csv` and `results/stop3a_balanced_primary/stop3a_full_decision_rows.json`
- Aggregate summary: `results/stop3a_balanced_primary/stop3a_aggregate_summary.csv` and `results/stop3a_balanced_primary/stop3a_aggregate_summary.json`
- Ranked summary: `results/stop3a_balanced_primary/stop3a_ranked_summary.csv`
- Runner manifest: `results/stop3a_balanced_primary/manifest.json`
- Runner log: `results/stop3a_balanced_primary/run_log.json`

## Evaluation Semantics

Lower `selection_score` is better. It combines normalized ATC with squared penalties for missing the target recall floor and the configured accuracy or balanced-accuracy floor.

For these balanced Stop 3A datasets, the selected accuracy metric is plain accuracy:

- Spider target class: `black_widow`; cost matrix `[[0,10],[1,0]]`.
- BreaKHis target class: `malignant`; cost matrix `[[0,1],[10,0]]`.
- Spider floors: target recall `0.95`, accuracy `0.80`.
- BreaKHis floors: target recall `0.97`, accuracy `0.85`.

Decision modes mean:

- `argmax`: ordinary highest-logit prediction. Cost affects this mode only through the training loss.
- `calibrated_cost_min`: temperature-scaled probabilities plus minimum expected cost decision.
- `calibrated_threshold`: temperature-scaled probabilities plus a calibration-split threshold chosen to minimize the same selection score. This is binary-only and is the most directly aligned with the current optimization goal.

## breakhis_balanced

Target: `malignant`. Floors: recall `0.97`, accuracy `0.85` using `accuracy`.

### Top Overall Rows

| rank | model | method | mode | floors | selection | nATC | recall | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | ce_calibrated_cost_min | calibrated_cost_min | 67% | 0.0173 +/- 0.0050 | 0.0164 +/- 0.0037 | 0.9976 +/- 0.0041 | 0.8463 +/- 0.0407 | 0.8463 +/- 0.0407 | 0.8421 |
| 2 | vit | ce_calibrated_cost_min | calibrated_threshold | 33% | 0.0186 +/- 0.0027 | 0.0178 +/- 0.0016 | 0.9965 +/- 0.0035 | 0.8375 +/- 0.0319 | 0.8375 +/- 0.0319 | 0.8328 |
| 3 | vit | nicme_logit_adjustment | calibrated_threshold | 33% | 0.0186 +/- 0.0056 | 0.0178 +/- 0.0047 | 0.9929 +/- 0.0035 | 0.8534 +/- 0.0623 | 0.8534 +/- 0.0623 | 0.8493 |
| 4 | vit | nicme_hybrid | calibrated_threshold | 67% | 0.0188 +/- 0.0033 | 0.0180 +/- 0.0018 | 0.9929 +/- 0.0071 | 0.8522 +/- 0.0486 | 0.8522 +/- 0.0486 | 0.8484 |
| 5 | vit | menon_logit_adjusted | calibrated_threshold | 0% | 0.0191 +/- 0.0017 | 0.0185 +/- 0.0010 | 0.9965 +/- 0.0035 | 0.8310 +/- 0.0194 | 0.8310 +/- 0.0194 | 0.8260 |
| 6 | vit | ce | calibrated_threshold | 33% | 0.0198 +/- 0.0023 | 0.0191 +/- 0.0014 | 0.9929 +/- 0.0061 | 0.8404 +/- 0.0371 | 0.8404 +/- 0.0371 | 0.8361 |
| 7 | vit | ce | calibrated_cost_min | 33% | 0.0213 +/- 0.0034 | 0.0196 +/- 0.0020 | 0.9929 +/- 0.0094 | 0.8357 +/- 0.0610 | 0.8357 +/- 0.0610 | 0.8301 |
| 8 | vit | ce | argmax | 33% | 0.0213 +/- 0.0016 | 0.0213 +/- 0.0016 | 0.9787 +/- 0.0094 | 0.8824 +/- 0.0543 | 0.8824 +/- 0.0543 | 0.8806 |
| 9 | vit | ce_calibrated_cost_min | argmax | 67% | 0.0217 +/- 0.0052 | 0.0212 +/- 0.0044 | 0.9693 +/- 0.0168 | 0.9261 +/- 0.0346 | 0.9261 +/- 0.0346 | 0.9258 |
| 10 | vit | cs_regularized_ce | calibrated_threshold | 0% | 0.0220 +/- 0.0008 | 0.0219 +/- 0.0008 | 0.9870 +/- 0.0020 | 0.8398 +/- 0.0020 | 0.8398 +/- 0.0020 | 0.8363 |

### Top NICME Rows

| rank | model | method | mode | floors | selection | nATC | recall | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | vit | nicme_logit_adjustment | calibrated_threshold | 33% | 0.0186 +/- 0.0056 | 0.0178 +/- 0.0047 | 0.9929 +/- 0.0035 | 0.8534 +/- 0.0623 | 0.8534 +/- 0.0623 | 0.8493 |
| 4 | vit | nicme_hybrid | calibrated_threshold | 67% | 0.0188 +/- 0.0033 | 0.0180 +/- 0.0018 | 0.9929 +/- 0.0071 | 0.8522 +/- 0.0486 | 0.8522 +/- 0.0486 | 0.8484 |
| 11 | vit | nicme_logit_adjustment | argmax | 67% | 0.0229 +/- 0.0085 | 0.0228 +/- 0.0085 | 0.9752 +/- 0.0128 | 0.8836 +/- 0.0289 | 0.8836 +/- 0.0289 | 0.8825 |
| 13 | vit | nicme_logit_adjustment | calibrated_cost_min | 0% | 0.0273 +/- 0.0093 | 0.0228 +/- 0.0038 | 0.9953 +/- 0.0054 | 0.7937 +/- 0.0457 | 0.7937 +/- 0.0457 | 0.7839 |
| 16 | vit | nicme_hybrid | argmax | 33% | 0.0457 +/- 0.0464 | 0.0339 +/- 0.0259 | 0.9456 +/- 0.0610 | 0.9054 +/- 0.0504 | 0.9054 +/- 0.0504 | 0.9044 |
| 17 | timm_dinov3_vit_lora | nicme_hybrid | calibrated_threshold | 33% | 0.0476 +/- 0.0275 | 0.0379 +/- 0.0190 | 0.9374 +/- 0.0481 | 0.9031 +/- 0.0269 | 0.9031 +/- 0.0269 | 0.9024 |
| 18 | timm_dinov3_vit_lora | nicme_hybrid | argmax | 0% | 0.0493 +/- 0.0167 | 0.0408 +/- 0.0101 | 0.9279 +/- 0.0228 | 0.9161 +/- 0.0037 | 0.9161 +/- 0.0037 | 0.9160 |
| 19 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_threshold | 0% | 0.0610 +/- 0.0033 | 0.0472 +/- 0.0016 | 0.9113 +/- 0.0035 | 0.9267 +/- 0.0041 | 0.9267 +/- 0.0041 | 0.9267 |

### Floor-Satisfying Rows

No aggregate row met both floors in all seeds.

### Interpretation

BreaKHis remains extremely close. The lowest mean selection score belongs to `vit + ce_calibrated_cost_min + calibrated_cost_min`, but its mean accuracy is `0.8463`, below the configured `0.85` floor. This means it is not the cleanest paper-facing winner under the stated objective.

The best mean-floor-compliant NICME result is `vit + nicme_logit_adjustment + calibrated_threshold`, with mean nATC `0.0178`, mean malignant recall `0.9929`, and mean accuracy `0.8534`. This is a stronger result than the Stop 2 tumor prototype because NICME is near the top and clears the floors on mean performance. The caveat is important: no BreaKHis row met both floors in all three seeds, so this should be treated as promising but not yet strict floor-stable evidence.

## spider_balanced

Target: `black_widow`. Floors: recall `0.95`, accuracy `0.80` using `accuracy`.

### Top Overall Rows

| rank | model | method | mode | floors | selection | nATC | recall | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_threshold | 33% | 0.0214 +/- 0.0020 | 0.0211 +/- 0.0017 | 1.0000 +/- 0.0000 | 0.7889 +/- 0.0168 | 0.7889 +/- 0.0168 | 0.7789 |
| 2 | timm_dinov3_vit_lora | nicme_hybrid | calibrated_threshold | 0% | 0.0218 +/- 0.0007 | 0.0216 +/- 0.0005 | 1.0000 +/- 0.0000 | 0.7844 +/- 0.0051 | 0.7844 +/- 0.0051 | 0.7739 |
| 3 | vit | cs_regularized_ce | calibrated_threshold | 33% | 0.0230 +/- 0.0085 | 0.0221 +/- 0.0079 | 0.9956 +/- 0.0077 | 0.7989 +/- 0.0593 | 0.7989 +/- 0.0593 | 0.7894 |
| 4 | vit | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0230 +/- 0.0047 | 0.0229 +/- 0.0047 | 0.9933 +/- 0.0115 | 0.8011 +/- 0.0184 | 0.8011 +/- 0.0184 | 0.7933 |
| 5 | vit | nicme_hybrid | calibrated_threshold | 67% | 0.0261 +/- 0.0065 | 0.0260 +/- 0.0064 | 0.9867 +/- 0.0115 | 0.8000 +/- 0.0115 | 0.8000 +/- 0.0115 | 0.7928 |
| 6 | timm_dinov3_vit_lora | nicme_hybrid | argmax | all | 0.0272 +/- 0.0033 | 0.0272 +/- 0.0033 | 0.9600 +/- 0.0067 | 0.9078 +/- 0.0038 | 0.9078 +/- 0.0038 | 0.9075 |
| 7 | vit | ce | calibrated_threshold | 33% | 0.0273 +/- 0.0009 | 0.0264 +/- 0.0002 | 0.9844 +/- 0.0154 | 0.8056 +/- 0.0703 | 0.8056 +/- 0.0703 | 0.7967 |
| 8 | vit | menon_logit_adjusted | calibrated_threshold | 33% | 0.0277 +/- 0.0016 | 0.0267 +/- 0.0007 | 0.9911 +/- 0.0038 | 0.7733 +/- 0.0233 | 0.7733 +/- 0.0233 | 0.7617 |
| 9 | timm_dinov3_vit_lora | nicme_logit_adjustment | argmax | 67% | 0.0279 +/- 0.0037 | 0.0279 +/- 0.0037 | 0.9578 +/- 0.0102 | 0.9111 +/- 0.0102 | 0.9111 +/- 0.0102 | 0.9109 |
| 10 | vit | ce_calibrated_cost_min | calibrated_threshold | 0% | 0.0281 +/- 0.0045 | 0.0269 +/- 0.0035 | 0.9911 +/- 0.0038 | 0.7711 +/- 0.0222 | 0.7711 +/- 0.0222 | 0.7592 |

### Top NICME Rows

| rank | model | method | mode | floors | selection | nATC | recall | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_threshold | 33% | 0.0214 +/- 0.0020 | 0.0211 +/- 0.0017 | 1.0000 +/- 0.0000 | 0.7889 +/- 0.0168 | 0.7889 +/- 0.0168 | 0.7789 |
| 2 | timm_dinov3_vit_lora | nicme_hybrid | calibrated_threshold | 0% | 0.0218 +/- 0.0007 | 0.0216 +/- 0.0005 | 1.0000 +/- 0.0000 | 0.7844 +/- 0.0051 | 0.7844 +/- 0.0051 | 0.7739 |
| 4 | vit | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0230 +/- 0.0047 | 0.0229 +/- 0.0047 | 0.9933 +/- 0.0115 | 0.8011 +/- 0.0184 | 0.8011 +/- 0.0184 | 0.7933 |
| 5 | vit | nicme_hybrid | calibrated_threshold | 67% | 0.0261 +/- 0.0065 | 0.0260 +/- 0.0064 | 0.9867 +/- 0.0115 | 0.8000 +/- 0.0115 | 0.8000 +/- 0.0115 | 0.7928 |
| 6 | timm_dinov3_vit_lora | nicme_hybrid | argmax | all | 0.0272 +/- 0.0033 | 0.0272 +/- 0.0033 | 0.9600 +/- 0.0067 | 0.9078 +/- 0.0038 | 0.9078 +/- 0.0038 | 0.9075 |
| 9 | timm_dinov3_vit_lora | nicme_logit_adjustment | argmax | 67% | 0.0279 +/- 0.0037 | 0.0279 +/- 0.0037 | 0.9578 +/- 0.0102 | 0.9111 +/- 0.0102 | 0.9111 +/- 0.0102 | 0.9109 |
| 11 | vit | nicme_logit_adjustment | argmax | all | 0.0299 +/- 0.0070 | 0.0299 +/- 0.0070 | 0.9622 +/- 0.0102 | 0.8711 +/- 0.0250 | 0.8711 +/- 0.0250 | 0.8700 |
| 12 | vit | nicme_hybrid | argmax | all | 0.0323 +/- 0.0023 | 0.0323 +/- 0.0023 | 0.9622 +/- 0.0038 | 0.8467 +/- 0.0186 | 0.8467 +/- 0.0186 | 0.8445 |

### Floor-Satisfying Rows

| rank | model | method | mode | floors | selection | nATC | recall | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | timm_dinov3_vit_lora | nicme_hybrid | argmax | all | 0.0272 +/- 0.0033 | 0.0272 +/- 0.0033 | 0.9600 +/- 0.0067 | 0.9078 +/- 0.0038 | 0.9078 +/- 0.0038 | 0.9075 |
| 11 | vit | nicme_logit_adjustment | argmax | all | 0.0299 +/- 0.0070 | 0.0299 +/- 0.0070 | 0.9622 +/- 0.0102 | 0.8711 +/- 0.0250 | 0.8711 +/- 0.0250 | 0.8700 |
| 12 | vit | nicme_hybrid | argmax | all | 0.0323 +/- 0.0023 | 0.0323 +/- 0.0023 | 0.9622 +/- 0.0038 | 0.8467 +/- 0.0186 | 0.8467 +/- 0.0186 | 0.8445 |

### Interpretation

Spider is the key stress test because Stop 2 showed CE plus thresholding beating NICME. In Stop 3A, the top two rows by mean selection are both timm DINOv3 ViT LoRA NICME rows using `calibrated_threshold`, with perfect mean target recall and normalized ATC near `0.021`. However, their mean accuracy is just below the `0.80` floor, so they should be treated as high-recall/high-cost-control candidates rather than clean floor-satisfying winners.

The strongest mean-floor-compliant calibrated NICME row is `vit + nicme_logit_adjustment + calibrated_threshold`, with mean nATC `0.0229`, mean target recall `0.9933`, and mean accuracy `0.8011`, though only two of three seeds satisfy both floors. The strongest strict all-seed floor-satisfying row is `timm_dinov3_vit_lora + nicme_hybrid + argmax`, with mean nATC `0.0272`, mean target recall `0.9600`, and mean accuracy `0.9078`. That makes the Spider read favorable to NICME, but the exact deployment mode depends on whether the paper prioritizes lowest mean selection score or strict per-seed floor stability.

## Method-Level Scientific Takeaways

### NICME Versus CE

The main favorable result is not that every NICME variant dominates every baseline. It does not. The favorable result is narrower and more scientifically credible: after tuning `nicme_logit_cost_scale`, NICME logit adjustment with calibrated-threshold inference is the best mean-floor-compliant calibrated row on both balanced datasets, while NICME argmax/hybrid rows provide stricter per-seed floor stability on Spider.

This matters because balanced datasets are the central decoupling condition. These results are less vulnerable to the criticism that cost-sensitive gains are just class-imbalance correction.

### NICME Hybrid Versus NICME Logit Adjustment

In Stop 3A, `nicme_logit_adjustment` is generally stronger than `nicme_hybrid`. The hybrid loss remains competitive, especially on Spider with timm DINOv3 LoRA, but the added cost-sensitive regularizer can still push false positives too high. This suggests the current hybrid regularizer is not yet the safest final proposed method unless later Stop 3B/Stop 4 runs reverse the pattern.

### Calibrated Threshold Versus Calibrated Cost-Min

The calibrated-threshold mode is consistently important. Plain `calibrated_cost_min` often drives recall up but can fall just under the accuracy floor. Threshold tuning on the calibration split gives the method a way to optimize the actual study objective rather than blindly applying the Bayes cost rule from the cost matrix alone.

### timm DINOv3 ViT LoRA

The timm DINOv3 ViT LoRA rows are strong on Spider but weaker on BreaKHis than full ViT in this pass. This is a useful paper result rather than a failure: the model/backend comparison should be framed as empirical, not assumed. The official gated Meta DINOv3 checkpoints remain shelved until access is available, so these rows should be labeled as timm DINOv3-weighted ViT-S LoRA rather than official Hugging Face `facebook/dinov3-*` rows.

## Aggregate Ranked Table

This table includes every Stop 3A aggregate row. `floors=all` means all three seeds met both configured floors; otherwise the percentage is the fraction of seeds satisfying both floors.

| rank | model | method | mode | floors | selection | nATC | recall | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | ce_calibrated_cost_min | calibrated_cost_min | 67% | 0.0173 +/- 0.0050 | 0.0164 +/- 0.0037 | 0.9976 +/- 0.0041 | 0.8463 +/- 0.0407 | 0.8463 +/- 0.0407 | 0.8421 |
| 2 | vit | ce_calibrated_cost_min | calibrated_threshold | 33% | 0.0186 +/- 0.0027 | 0.0178 +/- 0.0016 | 0.9965 +/- 0.0035 | 0.8375 +/- 0.0319 | 0.8375 +/- 0.0319 | 0.8328 |
| 3 | vit | nicme_logit_adjustment | calibrated_threshold | 33% | 0.0186 +/- 0.0056 | 0.0178 +/- 0.0047 | 0.9929 +/- 0.0035 | 0.8534 +/- 0.0623 | 0.8534 +/- 0.0623 | 0.8493 |
| 4 | vit | nicme_hybrid | calibrated_threshold | 67% | 0.0188 +/- 0.0033 | 0.0180 +/- 0.0018 | 0.9929 +/- 0.0071 | 0.8522 +/- 0.0486 | 0.8522 +/- 0.0486 | 0.8484 |
| 5 | vit | menon_logit_adjusted | calibrated_threshold | 0% | 0.0191 +/- 0.0017 | 0.0185 +/- 0.0010 | 0.9965 +/- 0.0035 | 0.8310 +/- 0.0194 | 0.8310 +/- 0.0194 | 0.8260 |
| 6 | vit | ce | calibrated_threshold | 33% | 0.0198 +/- 0.0023 | 0.0191 +/- 0.0014 | 0.9929 +/- 0.0061 | 0.8404 +/- 0.0371 | 0.8404 +/- 0.0371 | 0.8361 |
| 7 | vit | ce | calibrated_cost_min | 33% | 0.0213 +/- 0.0034 | 0.0196 +/- 0.0020 | 0.9929 +/- 0.0094 | 0.8357 +/- 0.0610 | 0.8357 +/- 0.0610 | 0.8301 |
| 8 | vit | ce | argmax | 33% | 0.0213 +/- 0.0016 | 0.0213 +/- 0.0016 | 0.9787 +/- 0.0094 | 0.8824 +/- 0.0543 | 0.8824 +/- 0.0543 | 0.8806 |
| 9 | vit | ce_calibrated_cost_min | argmax | 67% | 0.0217 +/- 0.0052 | 0.0212 +/- 0.0044 | 0.9693 +/- 0.0168 | 0.9261 +/- 0.0346 | 0.9261 +/- 0.0346 | 0.9258 |
| 10 | vit | cs_regularized_ce | calibrated_threshold | 0% | 0.0220 +/- 0.0008 | 0.0219 +/- 0.0008 | 0.9870 +/- 0.0020 | 0.8398 +/- 0.0020 | 0.8398 +/- 0.0020 | 0.8363 |
| 11 | vit | nicme_logit_adjustment | argmax | 67% | 0.0229 +/- 0.0085 | 0.0228 +/- 0.0085 | 0.9752 +/- 0.0128 | 0.8836 +/- 0.0289 | 0.8836 +/- 0.0289 | 0.8825 |
| 12 | vit | menon_logit_adjusted | argmax | 67% | 0.0248 +/- 0.0150 | 0.0232 +/- 0.0123 | 0.9728 +/- 0.0322 | 0.8901 +/- 0.0250 | 0.8901 +/- 0.0250 | 0.8889 |
| 13 | vit | nicme_logit_adjustment | calibrated_cost_min | 0% | 0.0273 +/- 0.0093 | 0.0228 +/- 0.0038 | 0.9953 +/- 0.0054 | 0.7937 +/- 0.0457 | 0.7937 +/- 0.0457 | 0.7839 |
| 14 | vit | menon_logit_adjusted | calibrated_cost_min | 0% | 0.0288 +/- 0.0115 | 0.0225 +/- 0.0042 | 0.9988 +/- 0.0020 | 0.7807 +/- 0.0484 | 0.7807 +/- 0.0484 | 0.7685 |
| 15 | vit | cs_regularized_ce | argmax | 33% | 0.0310 +/- 0.0187 | 0.0287 +/- 0.0149 | 0.9669 +/- 0.0329 | 0.8617 +/- 0.0258 | 0.8617 +/- 0.0258 | 0.8599 |
| 16 | vit | nicme_hybrid | argmax | 33% | 0.0457 +/- 0.0464 | 0.0339 +/- 0.0259 | 0.9456 +/- 0.0610 | 0.9054 +/- 0.0504 | 0.9054 +/- 0.0504 | 0.9044 |
| 17 | timm_dinov3_vit_lora | nicme_hybrid | calibrated_threshold | 33% | 0.0476 +/- 0.0275 | 0.0379 +/- 0.0190 | 0.9374 +/- 0.0481 | 0.9031 +/- 0.0269 | 0.9031 +/- 0.0269 | 0.9024 |
| 18 | timm_dinov3_vit_lora | nicme_hybrid | argmax | 0% | 0.0493 +/- 0.0167 | 0.0408 +/- 0.0101 | 0.9279 +/- 0.0228 | 0.9161 +/- 0.0037 | 0.9161 +/- 0.0037 | 0.9160 |
| 19 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_threshold | 0% | 0.0610 +/- 0.0033 | 0.0472 +/- 0.0016 | 0.9113 +/- 0.0035 | 0.9267 +/- 0.0041 | 0.9267 +/- 0.0041 | 0.9267 |
| 20 | vit | cs_regularized_ce | calibrated_cost_min | 0% | 0.0718 +/- 0.0872 | 0.0303 +/- 0.0171 | 0.9965 +/- 0.0035 | 0.7128 +/- 0.1843 | 0.7128 +/- 0.1843 | 0.6534 |
| 21 | timm_dinov3_vit_lora | menon_logit_adjusted | calibrated_threshold | 0% | 0.0900 +/- 0.0310 | 0.0609 +/- 0.0141 | 0.8877 +/- 0.0275 | 0.8960 +/- 0.0181 | 0.8960 +/- 0.0181 | 0.8960 |
| 22 | timm_dinov3_vit_lora | ce | calibrated_threshold | 0% | 0.0902 +/- 0.0309 | 0.0611 +/- 0.0140 | 0.8877 +/- 0.0275 | 0.8948 +/- 0.0178 | 0.8948 +/- 0.0178 | 0.8948 |
| 23 | timm_dinov3_vit_lora | ce_calibrated_cost_min | calibrated_threshold | 0% | 0.0902 +/- 0.0309 | 0.0611 +/- 0.0140 | 0.8877 +/- 0.0275 | 0.8948 +/- 0.0178 | 0.8948 +/- 0.0178 | 0.8948 |
| 24 | timm_dinov3_vit_lora | cs_regularized_ce | calibrated_threshold | 0% | 0.0992 +/- 0.0152 | 0.0652 +/- 0.0075 | 0.8783 +/- 0.0108 | 0.8960 +/- 0.0271 | 0.8960 +/- 0.0271 | 0.8959 |
| 25 | timm_dinov3_vit_lora | nicme_logit_adjustment | argmax | 0% | 0.1011 +/- 0.0191 | 0.0642 +/- 0.0075 | 0.8747 +/- 0.0148 | 0.9220 +/- 0.0081 | 0.9220 +/- 0.0081 | 0.9218 |
| 26 | vit | nicme_hybrid | calibrated_cost_min | 0% | 0.1153 +/- 0.0789 | 0.0393 +/- 0.0159 | 1.0000 +/- 0.0000 | 0.6070 +/- 0.1594 | 0.6070 +/- 0.1594 | 0.5042 |
| 27 | timm_dinov3_vit_lora | cs_regularized_ce | argmax | 0% | 0.1238 +/- 0.0348 | 0.0739 +/- 0.0098 | 0.8605 +/- 0.0266 | 0.8883 +/- 0.0218 | 0.8883 +/- 0.0218 | 0.8881 |
| 28 | timm_dinov3_vit_lora | nicme_hybrid | calibrated_cost_min | 0% | 0.1238 +/- 0.0843 | 0.0409 +/- 0.0158 | 0.9988 +/- 0.0020 | 0.5963 +/- 0.1669 | 0.5963 +/- 0.1669 | 0.4821 |
| 29 | timm_dinov3_vit_lora | cs_regularized_ce | calibrated_cost_min | 0% | 0.1306 +/- 0.0725 | 0.0471 +/- 0.0050 | 0.9775 +/- 0.0389 | 0.6300 +/- 0.2252 | 0.6300 +/- 0.2252 | 0.5188 |
| 30 | timm_dinov3_vit_lora | ce | calibrated_cost_min | 0% | 0.1355 +/- 0.0641 | 0.0492 +/- 0.0013 | 0.9704 +/- 0.0512 | 0.6407 +/- 0.2436 | 0.6407 +/- 0.2436 | 0.5295 |
| 31 | timm_dinov3_vit_lora | ce_calibrated_cost_min | calibrated_cost_min | 0% | 0.1355 +/- 0.0641 | 0.0492 +/- 0.0013 | 0.9704 +/- 0.0512 | 0.6407 +/- 0.2436 | 0.6407 +/- 0.2436 | 0.5295 |
| 32 | timm_dinov3_vit_lora | menon_logit_adjusted | calibrated_cost_min | 0% | 0.1355 +/- 0.0641 | 0.0492 +/- 0.0013 | 0.9704 +/- 0.0512 | 0.6407 +/- 0.2436 | 0.6407 +/- 0.2436 | 0.5295 |
| 33 | timm_dinov3_vit_lora | ce | argmax | 0% | 0.1530 +/- 0.0302 | 0.0824 +/- 0.0087 | 0.8381 +/- 0.0195 | 0.9048 +/- 0.0010 | 0.9048 +/- 0.0010 | 0.9044 |
| 34 | timm_dinov3_vit_lora | ce_calibrated_cost_min | argmax | 0% | 0.1530 +/- 0.0302 | 0.0824 +/- 0.0087 | 0.8381 +/- 0.0195 | 0.9048 +/- 0.0010 | 0.9048 +/- 0.0010 | 0.9044 |
| 35 | timm_dinov3_vit_lora | menon_logit_adjusted | argmax | 0% | 0.1530 +/- 0.0302 | 0.0824 +/- 0.0087 | 0.8381 +/- 0.0195 | 0.9048 +/- 0.0010 | 0.9048 +/- 0.0010 | 0.9044 |
| 36 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_cost_min | 0% | 0.1692 +/- 0.0029 | 0.0496 +/- 0.0004 | 1.0000 +/- 0.0000 | 0.5041 +/- 0.0037 | 0.5041 +/- 0.0037 | 0.3424 |
| 1 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_threshold | 33% | 0.0214 +/- 0.0020 | 0.0211 +/- 0.0017 | 1.0000 +/- 0.0000 | 0.7889 +/- 0.0168 | 0.7889 +/- 0.0168 | 0.7789 |
| 2 | timm_dinov3_vit_lora | nicme_hybrid | calibrated_threshold | 0% | 0.0218 +/- 0.0007 | 0.0216 +/- 0.0005 | 1.0000 +/- 0.0000 | 0.7844 +/- 0.0051 | 0.7844 +/- 0.0051 | 0.7739 |
| 3 | vit | cs_regularized_ce | calibrated_threshold | 33% | 0.0230 +/- 0.0085 | 0.0221 +/- 0.0079 | 0.9956 +/- 0.0077 | 0.7989 +/- 0.0593 | 0.7989 +/- 0.0593 | 0.7894 |
| 4 | vit | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0230 +/- 0.0047 | 0.0229 +/- 0.0047 | 0.9933 +/- 0.0115 | 0.8011 +/- 0.0184 | 0.8011 +/- 0.0184 | 0.7933 |
| 5 | vit | nicme_hybrid | calibrated_threshold | 67% | 0.0261 +/- 0.0065 | 0.0260 +/- 0.0064 | 0.9867 +/- 0.0115 | 0.8000 +/- 0.0115 | 0.8000 +/- 0.0115 | 0.7928 |
| 6 | timm_dinov3_vit_lora | nicme_hybrid | argmax | all | 0.0272 +/- 0.0033 | 0.0272 +/- 0.0033 | 0.9600 +/- 0.0067 | 0.9078 +/- 0.0038 | 0.9078 +/- 0.0038 | 0.9075 |
| 7 | vit | ce | calibrated_threshold | 33% | 0.0273 +/- 0.0009 | 0.0264 +/- 0.0002 | 0.9844 +/- 0.0154 | 0.8056 +/- 0.0703 | 0.8056 +/- 0.0703 | 0.7967 |
| 8 | vit | menon_logit_adjusted | calibrated_threshold | 33% | 0.0277 +/- 0.0016 | 0.0267 +/- 0.0007 | 0.9911 +/- 0.0038 | 0.7733 +/- 0.0233 | 0.7733 +/- 0.0233 | 0.7617 |
| 9 | timm_dinov3_vit_lora | nicme_logit_adjustment | argmax | 67% | 0.0279 +/- 0.0037 | 0.0279 +/- 0.0037 | 0.9578 +/- 0.0102 | 0.9111 +/- 0.0102 | 0.9111 +/- 0.0102 | 0.9109 |
| 10 | vit | ce_calibrated_cost_min | calibrated_threshold | 0% | 0.0281 +/- 0.0045 | 0.0269 +/- 0.0035 | 0.9911 +/- 0.0038 | 0.7711 +/- 0.0222 | 0.7711 +/- 0.0222 | 0.7592 |
| 11 | vit | nicme_logit_adjustment | argmax | all | 0.0299 +/- 0.0070 | 0.0299 +/- 0.0070 | 0.9622 +/- 0.0102 | 0.8711 +/- 0.0250 | 0.8711 +/- 0.0250 | 0.8700 |
| 12 | vit | nicme_hybrid | argmax | all | 0.0323 +/- 0.0023 | 0.0323 +/- 0.0023 | 0.9622 +/- 0.0038 | 0.8467 +/- 0.0186 | 0.8467 +/- 0.0186 | 0.8445 |
| 13 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_cost_min | 0% | 0.0347 +/- 0.0050 | 0.0280 +/- 0.0020 | 1.0000 +/- 0.0000 | 0.7200 +/- 0.0203 | 0.7200 +/- 0.0203 | 0.6959 |
| 14 | vit | cs_regularized_ce | argmax | 33% | 0.0384 +/- 0.0044 | 0.0381 +/- 0.0044 | 0.9444 +/- 0.0077 | 0.8689 +/- 0.0532 | 0.8689 +/- 0.0532 | 0.8676 |
| 15 | vit | ce | calibrated_cost_min | 33% | 0.0387 +/- 0.0213 | 0.0273 +/- 0.0098 | 1.0000 +/- 0.0000 | 0.7267 +/- 0.0985 | 0.7267 +/- 0.0985 | 0.6981 |
| 16 | vit | cs_regularized_ce | calibrated_cost_min | 33% | 0.0390 +/- 0.0203 | 0.0276 +/- 0.0096 | 0.9956 +/- 0.0077 | 0.7444 +/- 0.1300 | 0.7444 +/- 0.1300 | 0.7170 |
| 17 | vit | ce | argmax | 0% | 0.0403 +/- 0.0012 | 0.0398 +/- 0.0007 | 0.9400 +/- 0.0067 | 0.8722 +/- 0.0234 | 0.8722 +/- 0.0234 | 0.8715 |
| 18 | vit | ce_calibrated_cost_min | argmax | 33% | 0.0404 +/- 0.0033 | 0.0401 +/- 0.0030 | 0.9444 +/- 0.0077 | 0.8489 +/- 0.0069 | 0.8489 +/- 0.0069 | 0.8475 |
| 19 | vit | menon_logit_adjusted | argmax | 33% | 0.0415 +/- 0.0049 | 0.0410 +/- 0.0044 | 0.9422 +/- 0.0102 | 0.8500 +/- 0.0145 | 0.8500 +/- 0.0145 | 0.8486 |
| 20 | vit | menon_logit_adjusted | calibrated_cost_min | 0% | 0.0438 +/- 0.0074 | 0.0311 +/- 0.0022 | 1.0000 +/- 0.0000 | 0.6889 +/- 0.0222 | 0.6889 +/- 0.0222 | 0.6551 |
| 21 | vit | ce_calibrated_cost_min | calibrated_cost_min | 0% | 0.0470 +/- 0.0103 | 0.0320 +/- 0.0029 | 1.0000 +/- 0.0000 | 0.6800 +/- 0.0291 | 0.6800 +/- 0.0291 | 0.6427 |
| 22 | timm_dinov3_vit_lora | ce | calibrated_threshold | 0% | 0.0591 +/- 0.0314 | 0.0446 +/- 0.0192 | 0.9689 +/- 0.0269 | 0.6944 +/- 0.0712 | 0.6944 +/- 0.0712 | 0.6677 |
| 23 | timm_dinov3_vit_lora | ce_calibrated_cost_min | calibrated_threshold | 0% | 0.0591 +/- 0.0314 | 0.0446 +/- 0.0192 | 0.9689 +/- 0.0269 | 0.6944 +/- 0.0712 | 0.6944 +/- 0.0712 | 0.6677 |
| 24 | timm_dinov3_vit_lora | menon_logit_adjusted | calibrated_threshold | 0% | 0.0591 +/- 0.0314 | 0.0446 +/- 0.0192 | 0.9689 +/- 0.0269 | 0.6944 +/- 0.0712 | 0.6944 +/- 0.0712 | 0.6677 |
| 25 | vit | nicme_logit_adjustment | calibrated_cost_min | 0% | 0.0664 +/- 0.0619 | 0.0341 +/- 0.0136 | 1.0000 +/- 0.0000 | 0.6589 +/- 0.1362 | 0.6589 +/- 0.1362 | 0.5913 |
| 26 | timm_dinov3_vit_lora | nicme_hybrid | calibrated_cost_min | 0% | 0.0680 +/- 0.0443 | 0.0360 +/- 0.0098 | 1.0000 +/- 0.0000 | 0.6400 +/- 0.0982 | 0.6400 +/- 0.0982 | 0.5745 |
| 27 | timm_dinov3_vit_lora | cs_regularized_ce | calibrated_threshold | 0% | 0.0763 +/- 0.0008 | 0.0554 +/- 0.0002 | 0.9533 +/- 0.0000 | 0.6556 +/- 0.0019 | 0.6556 +/- 0.0019 | 0.6220 |
| 28 | timm_dinov3_vit_lora | ce | calibrated_cost_min | 33% | 0.1010 +/- 0.0675 | 0.0410 +/- 0.0156 | 0.9978 +/- 0.0038 | 0.6000 +/- 0.1732 | 0.6000 +/- 0.1732 | 0.4863 |
| 29 | timm_dinov3_vit_lora | ce_calibrated_cost_min | calibrated_cost_min | 33% | 0.1010 +/- 0.0675 | 0.0410 +/- 0.0156 | 0.9978 +/- 0.0038 | 0.6000 +/- 0.1732 | 0.6000 +/- 0.1732 | 0.4863 |
| 30 | timm_dinov3_vit_lora | menon_logit_adjusted | calibrated_cost_min | 33% | 0.1010 +/- 0.0675 | 0.0410 +/- 0.0156 | 0.9978 +/- 0.0038 | 0.6000 +/- 0.1732 | 0.6000 +/- 0.1732 | 0.4863 |
| 31 | vit | nicme_hybrid | calibrated_cost_min | 0% | 0.1077 +/- 0.0559 | 0.0437 +/- 0.0110 | 1.0000 +/- 0.0000 | 0.5633 +/- 0.1097 | 0.5633 +/- 0.1097 | 0.4412 |
| 32 | timm_dinov3_vit_lora | cs_regularized_ce | calibrated_cost_min | 0% | 0.1400 +/- 0.0000 | 0.0500 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.3333 |
| 33 | timm_dinov3_vit_lora | cs_regularized_ce | argmax | 0% | 0.1468 +/- 0.0051 | 0.0972 +/- 0.0018 | 0.8511 +/- 0.0038 | 0.6978 +/- 0.0019 | 0.6978 +/- 0.0019 | 0.6905 |
| 34 | timm_dinov3_vit_lora | ce | argmax | 0% | 0.1652 +/- 0.0966 | 0.0963 +/- 0.0416 | 0.8378 +/- 0.0654 | 0.7667 +/- 0.1212 | 0.7667 +/- 0.1212 | 0.7645 |
| 35 | timm_dinov3_vit_lora | ce_calibrated_cost_min | argmax | 0% | 0.1652 +/- 0.0966 | 0.0963 +/- 0.0416 | 0.8378 +/- 0.0654 | 0.7667 +/- 0.1212 | 0.7667 +/- 0.1212 | 0.7645 |
| 36 | timm_dinov3_vit_lora | menon_logit_adjusted | argmax | 0% | 0.1652 +/- 0.0966 | 0.0963 +/- 0.0416 | 0.8378 +/- 0.0654 | 0.7667 +/- 0.1212 | 0.7667 +/- 0.1212 | 0.7645 |

## Acceptance Against Paper-Level Criteria

| Criterion | Stop 3A read |
| --- | --- |
| Balanced-dataset evidence required for decoupling claim | Satisfied as an experimental condition: both datasets are balanced. |
| NICME improves cared-class recall and ATC while preserving accuracy floors | Supported on mean-floor-compliant balanced rows; strict all-seed floor stability is strong on Spider but not yet achieved on BreaKHis. |
| NICME hybrid is the best proposed method | Not supported yet. `nicme_logit_adjustment` is stronger than `nicme_hybrid` in Stop 3A. |
| CE calibrated-cost-min baseline is included and competitive | Yes. It is the strongest non-NICME competitor and nearly wins BreaKHis. |
| Menon baseline remains an imbalance/long-tail baseline | Yes. It is included but not treated as a user-defined cost-matrix method. |
| Results justify moving directly to all imbalanced Stop 3B runs | Not automatically. A short review checkpoint is recommended first because Spider still has a tight accuracy-floor margin. |

## Recommended Next Step

Before running Stop 3B, inspect the full decision rows for the best floor-compliant NICME and CE rows. If the user approves, run the imbalanced decoupling queue with the same tuned settings: `spider_target_minority`, `spider_target_majority`, and `breakhis_natural`.

If the goal is specifically to make `nicme_hybrid` the named strongest method, do not proceed blindly. Instead, run a focused Stop 3A-Hybrid tuning pass with lower Spider `cs_lambda` values, possibly `0.025`, `0.05`, and `0.075`, and BreaKHis `cs_lambda` values around `0.10` and `0.15`. The current evidence favors NICME logit adjustment over the hybrid.

## Provenance

This report was generated from local Stop 3A artifacts only. No new external literature claims are made here. The scientific framing follows the already saved literature-search and stop-gated plan artifacts.
