# Stop 2 Calibration And Model Adjustment Analysis

Generated: 2026-04-28

## Scope

The stop-gated paper plan is intentionally paused after Stop 2. This artifact explains the inference-mode semantics and records every Stop 2 result row with its cost matrix so calibration and NICME adjustments can be planned without losing provenance.

Source result files:

- `results/stop2_prototype/summary.csv`
- `results/stop2_prototype/full_results_with_cost_matrix.csv`
- `results/stop2_prototype/full_results_with_cost_matrix.json`

## Inference Mode Semantics

### `argmax`

`argmax` is the ordinary deployment rule for a classifier. The trained model produces logits for each class, probabilities are computed for metric/calibration reporting, and the predicted class is the largest logit/probability:

`prediction = argmax_k logit_k`

In this mode, the cost matrix does **not** directly change the final decision rule. It affects results only if the training method used it while learning, such as `nicme_hybrid`, `nicme_logit_adjustment`, or `cs_regularized_ce`. For `ce`, `ce_calibrated_cost_min`, and `menon_logit_adjusted`, the `argmax` row is the model's normal top-class prediction.

### `calibrated_cost_min`

`calibrated_cost_min` is a post-hoc decision rule. It does two things after training:

1. Fits a scalar temperature `T` on the calibration split by minimizing calibration negative log likelihood.
2. Converts test logits to calibrated probabilities and predicts the class with minimum expected cost:

`prediction = argmin_k sum_j C[j][k] * P(Y=j | x)`

The convention is always `C[true_label][predicted_label]`. This means calibrated-cost-min can change predictions even when the trained model is unchanged. In the binary 10:1 setting, this often behaves like a recall-favoring threshold shift toward the cared class. That is useful for ATC and target recall, but Stop 2 shows it can also collapse accuracy if the threshold becomes too aggressive.

Important consequence: temperature scaling alone does not change `argmax`. It changes the probability scale. The deployment decision changes only when those calibrated probabilities are fed into the minimum-expected-cost rule.

## Cost Matrices Used In Stop 2

- Spider class order: `[black_widow, false_widow]`
- Spider target class: `black_widow`
- Spider matrix: `[[0, 10], [1, 0]]`, so `black_widow -> false_widow` costs 10 and `false_widow -> black_widow` costs 1.
- BreaKHis class order: `[benign, malignant]`
- BreaKHis target class: `malignant`
- BreaKHis matrix: `[[0, 1], [10, 0]]`, so `malignant -> benign` costs 10 and `benign -> malignant` costs 1.

## Stop 2 High-Level Summary

- Clean run status: 32/32 configured training rows passed.
- Evaluation rows: 64 total because every trained row was evaluated under both `argmax` and `calibrated_cost_min`.
- Prototype runtime: 982.04 seconds, or 16.37 minutes.
- Both prototype datasets were balanced in every split: 256 train, 64 validation, 64 calibration, 64 test.
- BreaKHis prototype remained patient-disjoint with zero missing images.
- Official Meta `facebook/dinov3-*` checkpoints remained shelved; these rows use accessible ConvNeXt, ViT, timm DINOv3 ViT LoRA, and timm DINOv3 ConvNeXt MLP-LoRA.

- `spider_balanced_prototype`: best overall by selection score was `vit` / `ce_calibrated_cost_min` / `calibrated_cost_min` with nATC 0.0422, target recall 1.0000, accuracy 0.5781, selection 0.0914. Best argmax was `timm_dinov3_convnext_lora` / `nicme_hybrid` (nATC 0.0594, recall 0.9688, acc 0.5469). Best calibrated-cost-min was `vit` / `ce_calibrated_cost_min` (nATC 0.0422, recall 1.0000, acc 0.5781).
- `breakhis_balanced_prototype`: best overall by selection score was `convnext` / `ce` / `argmax` with nATC 0.0203, target recall 0.9688, accuracy 0.9375, selection 0.0203. Best argmax was `convnext` / `ce` (nATC 0.0203, recall 0.9688, acc 0.9375). Best calibrated-cost-min was `convnext` / `ce` (nATC 0.0437, recall 1.0000, acc 0.5625).

## Interpretation For Calibration And Model Adjustment

- `calibrated_cost_min` is currently very effective at pushing target recall to 0.9688-1.0000, but it often pays for that by overpredicting the target class and lowering accuracy.
- The best BreaKHis result did **not** require post-hoc cost-min: ConvNeXt CE argmax reached nATC 0.0203, target recall 0.9688, and accuracy 0.9375.
- The strongest training-time NICME signal was BreaKHis ViT `nicme_hybrid` argmax, with nATC 0.0234, target recall 0.9688, and accuracy 0.9062.
- Spider remains harder: the best nATC/recall rows are mostly calibrated-cost-min rows, but accuracy is usually 0.5000-0.5781. This is the main calibration-adjustment target before Stop 3.
- For the next tuning phase, the key question is not whether we can force recall high. We can. The question is how to reduce ATC while keeping the configured accuracy floor from being violated.

## Best Rows By Dataset

| dataset | model | method | mode | target | cost matrix C[true][pred] | nATC | target recall | target FNR | target precision | target FPR | acc | bal acc | macro-F1 | selection | AUROC | AUPRC | ECE | T | confusion rows=true cols=pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breakhis_balanced_prototype | convnext | ce_calibrated_cost_min | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0203 | 0.9688 | 0.0312 | 0.9118 | 0.0938 | 0.9375 | 0.9375 | 0.9374 | 0.0203 | 0.9541 | 0.9527 | 0.2577 |  | [[29, 3], [1, 31]] |
| breakhis_balanced_prototype | convnext | menon_logit_adjusted | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0203 | 0.9688 | 0.0312 | 0.9118 | 0.0938 | 0.9375 | 0.9375 | 0.9374 | 0.0203 | 0.9541 | 0.9527 | 0.2577 |  | [[29, 3], [1, 31]] |
| breakhis_balanced_prototype | convnext | ce | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0203 | 0.9688 | 0.0312 | 0.9118 | 0.0938 | 0.9375 | 0.9375 | 0.9374 | 0.0203 | 0.9541 | 0.9527 | 0.2577 |  | [[29, 3], [1, 31]] |
| breakhis_balanced_prototype | vit | nicme_hybrid | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0234 | 0.9688 | 0.0312 | 0.8611 | 0.1562 | 0.9062 | 0.9062 | 0.9059 | 0.0234 | 0.9756 | 0.9845 | 0.1706 |  | [[27, 5], [1, 31]] |
| breakhis_balanced_prototype | vit | menon_logit_adjusted | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0469 | 0.9062 | 0.0938 | 1.0000 | 0.0000 | 0.9531 | 0.9531 | 0.9530 | 0.0631 | 0.9619 | 0.9770 | 0.2855 |  | [[32, 0], [3, 29]] |
| breakhis_balanced_prototype | vit | ce | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0469 | 0.9062 | 0.0938 | 1.0000 | 0.0000 | 0.9531 | 0.9531 | 0.9530 | 0.0631 | 0.9619 | 0.9770 | 0.2855 |  | [[32, 0], [3, 29]] |
| breakhis_balanced_prototype | vit | ce_calibrated_cost_min | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0469 | 0.9062 | 0.0938 | 1.0000 | 0.0000 | 0.9531 | 0.9531 | 0.9530 | 0.0631 | 0.9619 | 0.9770 | 0.2852 |  | [[32, 0], [3, 29]] |
| spider_balanced_prototype | vit | ce_calibrated_cost_min | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0422 | 1.0000 | 0.0000 | 0.5424 | 0.8438 | 0.5781 | 0.5781 | 0.4868 | 0.0914 | 0.9443 | 0.9571 | 0.1213 | 0.5000 | [[32, 0], [27, 5]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | menon_logit_adjusted | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0563 | 0.9688 | 0.0312 | 0.5439 | 0.8125 | 0.5781 | 0.5781 | 0.5022 | 0.1055 | 0.7686 | 0.7734 | 0.1672 | 0.7000 | [[31, 1], [26, 6]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | ce | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0563 | 0.9688 | 0.0312 | 0.5439 | 0.8125 | 0.5781 | 0.5781 | 0.5022 | 0.1055 | 0.7686 | 0.7734 | 0.1672 | 0.7000 | [[31, 1], [26, 6]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0563 | 0.9688 | 0.0312 | 0.5439 | 0.8125 | 0.5781 | 0.5781 | 0.5022 | 0.1055 | 0.7686 | 0.7734 | 0.1672 | 0.7000 | [[31, 1], [26, 6]] |
| spider_balanced_prototype | vit | menon_logit_adjusted | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0453 | 1.0000 | 0.0000 | 0.5246 | 0.9062 | 0.5469 | 0.5469 | 0.4298 | 0.1094 | 0.9424 | 0.9548 | 0.1296 | 0.5000 | [[32, 0], [29, 3]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | nicme_hybrid | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0594 | 0.9688 | 0.0312 | 0.5254 | 0.8750 | 0.5469 | 0.5469 | 0.4488 | 0.1234 | 0.6357 | 0.6551 | 0.2522 |  | [[31, 1], [28, 4]] |
| breakhis_balanced_prototype | convnext | ce | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0437 | 1.0000 | 0.0000 | 0.5333 | 0.8750 | 0.5625 | 0.5625 | 0.4589 | 0.1264 | 0.9541 | 0.9527 | 0.1336 | 0.5000 | [[4, 28], [0, 32]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.6357 | 0.6551 | 0.0774 | 5.5000 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | timm_dinov3_vit_lora | nicme_hybrid | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.6621 | 0.6108 | 0.0142 | 0.5000 | [[32, 0], [32, 0]] |

## Complete Stop 2 Result Table

| dataset | model | method | mode | target | cost matrix C[true][pred] | nATC | target recall | target FNR | target precision | target FPR | acc | bal acc | macro-F1 | selection | AUROC | AUPRC | ECE | T | confusion rows=true cols=pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spider_balanced_prototype | convnext | ce | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.1781 | 0.6562 | 0.3438 | 0.8400 | 0.1250 | 0.7656 | 0.7656 | 0.7628 | 0.5245 | 0.8711 | 0.8851 | 0.1713 |  | [[21, 11], [4, 28]] |
| spider_balanced_prototype | convnext | ce | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.8711 | 0.8851 | 0.1181 | 0.5000 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | convnext | ce_calibrated_cost_min | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.1781 | 0.6562 | 0.3438 | 0.8400 | 0.1250 | 0.7656 | 0.7656 | 0.7628 | 0.5245 | 0.8711 | 0.8851 | 0.1713 |  | [[21, 11], [4, 28]] |
| spider_balanced_prototype | convnext | ce_calibrated_cost_min | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.8711 | 0.8851 | 0.1181 | 0.5000 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | convnext | menon_logit_adjusted | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.1781 | 0.6562 | 0.3438 | 0.8400 | 0.1250 | 0.7656 | 0.7656 | 0.7628 | 0.5245 | 0.8711 | 0.8851 | 0.1713 |  | [[21, 11], [4, 28]] |
| spider_balanced_prototype | convnext | menon_logit_adjusted | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.8711 | 0.8851 | 0.1181 | 0.5000 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | convnext | nicme_hybrid | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.5947 | 0.5446 | 0.2372 |  | [[32, 0], [32, 0]] |
| spider_balanced_prototype | convnext | nicme_hybrid | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.5947 | 0.5446 | 0.0991 | 7.2500 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | vit | ce | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.1453 | 0.7188 | 0.2812 | 0.8846 | 0.0938 | 0.8125 | 0.8125 | 0.8108 | 0.3592 | 0.9336 | 0.9455 | 0.2341 |  | [[23, 9], [3, 29]] |
| spider_balanced_prototype | vit | ce | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.9336 | 0.9455 | 0.1880 | 0.5000 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | vit | ce_calibrated_cost_min | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0969 | 0.8125 | 0.1875 | 0.9286 | 0.0625 | 0.8750 | 0.8750 | 0.8745 | 0.1725 | 0.9443 | 0.9571 | 0.2267 |  | [[26, 6], [2, 30]] |
| spider_balanced_prototype | vit | ce_calibrated_cost_min | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0422 | 1.0000 | 0.0000 | 0.5424 | 0.8438 | 0.5781 | 0.5781 | 0.4868 | 0.0914 | 0.9443 | 0.9571 | 0.1213 | 0.5000 | [[32, 0], [27, 5]] |
| spider_balanced_prototype | vit | menon_logit_adjusted | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0969 | 0.8125 | 0.1875 | 0.9286 | 0.0625 | 0.8750 | 0.8750 | 0.8745 | 0.1725 | 0.9424 | 0.9548 | 0.2307 |  | [[26, 6], [2, 30]] |
| spider_balanced_prototype | vit | menon_logit_adjusted | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0453 | 1.0000 | 0.0000 | 0.5246 | 0.9062 | 0.5469 | 0.5469 | 0.4298 | 0.1094 | 0.9424 | 0.9548 | 0.1296 | 0.5000 | [[32, 0], [29, 3]] |
| spider_balanced_prototype | vit | nicme_hybrid | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.6514 | 0.6124 | 0.1778 |  | [[32, 0], [32, 0]] |
| spider_balanced_prototype | vit | nicme_hybrid | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.6514 | 0.6124 | 0.1068 | 5.7500 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | timm_dinov3_vit_lora | ce | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.1375 | 0.7500 | 0.2500 | 0.7500 | 0.2500 | 0.7500 | 0.7500 | 0.7500 | 0.3000 | 0.8096 | 0.8307 | 0.2465 |  | [[24, 8], [8, 24]] |
| spider_balanced_prototype | timm_dinov3_vit_lora | ce | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.8096 | 0.8307 | 0.2431 | 0.5000 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | timm_dinov3_vit_lora | ce_calibrated_cost_min | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.1375 | 0.7500 | 0.2500 | 0.7500 | 0.2500 | 0.7500 | 0.7500 | 0.7500 | 0.3000 | 0.8096 | 0.8307 | 0.2465 |  | [[24, 8], [8, 24]] |
| spider_balanced_prototype | timm_dinov3_vit_lora | ce_calibrated_cost_min | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.8096 | 0.8307 | 0.2431 | 0.5000 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | timm_dinov3_vit_lora | menon_logit_adjusted | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.1375 | 0.7500 | 0.2500 | 0.7500 | 0.2500 | 0.7500 | 0.7500 | 0.7500 | 0.3000 | 0.8096 | 0.8307 | 0.2465 |  | [[24, 8], [8, 24]] |
| spider_balanced_prototype | timm_dinov3_vit_lora | menon_logit_adjusted | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.8096 | 0.8307 | 0.2431 | 0.5000 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | timm_dinov3_vit_lora | nicme_hybrid | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.6621 | 0.6113 | 0.0071 |  | [[32, 0], [32, 0]] |
| spider_balanced_prototype | timm_dinov3_vit_lora | nicme_hybrid | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.6621 | 0.6108 | 0.0142 | 0.5000 | [[32, 0], [32, 0]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | ce | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.2266 | 0.5625 | 0.4375 | 0.7826 | 0.1562 | 0.7031 | 0.7031 | 0.6971 | 0.8366 | 0.7686 | 0.7734 | 0.1596 |  | [[18, 14], [5, 27]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | ce | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0563 | 0.9688 | 0.0312 | 0.5439 | 0.8125 | 0.5781 | 0.5781 | 0.5022 | 0.1055 | 0.7686 | 0.7734 | 0.1672 | 0.7000 | [[31, 1], [26, 6]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | ce_calibrated_cost_min | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.2266 | 0.5625 | 0.4375 | 0.7826 | 0.1562 | 0.7031 | 0.7031 | 0.6971 | 0.8366 | 0.7686 | 0.7734 | 0.1596 |  | [[18, 14], [5, 27]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0563 | 0.9688 | 0.0312 | 0.5439 | 0.8125 | 0.5781 | 0.5781 | 0.5022 | 0.1055 | 0.7686 | 0.7734 | 0.1672 | 0.7000 | [[31, 1], [26, 6]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | menon_logit_adjusted | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.2266 | 0.5625 | 0.4375 | 0.7826 | 0.1562 | 0.7031 | 0.7031 | 0.6971 | 0.8366 | 0.7686 | 0.7734 | 0.1596 |  | [[18, 14], [5, 27]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | menon_logit_adjusted | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0563 | 0.9688 | 0.0312 | 0.5439 | 0.8125 | 0.5781 | 0.5781 | 0.5022 | 0.1055 | 0.7686 | 0.7734 | 0.1672 | 0.7000 | [[31, 1], [26, 6]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | nicme_hybrid | argmax | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0594 | 0.9688 | 0.0312 | 0.5254 | 0.8750 | 0.5469 | 0.5469 | 0.4488 | 0.1234 | 0.6357 | 0.6551 | 0.2522 |  | [[31, 1], [28, 4]] |
| spider_balanced_prototype | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_cost_min | black_widow | [[0.0,10.0],[1.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1400 | 0.6357 | 0.6551 | 0.0774 | 5.5000 | [[32, 0], [32, 0]] |
| breakhis_balanced_prototype | convnext | ce | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0203 | 0.9688 | 0.0312 | 0.9118 | 0.0938 | 0.9375 | 0.9375 | 0.9374 | 0.0203 | 0.9541 | 0.9527 | 0.2577 |  | [[29, 3], [1, 31]] |
| breakhis_balanced_prototype | convnext | ce | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0437 | 1.0000 | 0.0000 | 0.5333 | 0.8750 | 0.5625 | 0.5625 | 0.4589 | 0.1264 | 0.9541 | 0.9527 | 0.1336 | 0.5000 | [[4, 28], [0, 32]] |
| breakhis_balanced_prototype | convnext | ce_calibrated_cost_min | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0203 | 0.9688 | 0.0312 | 0.9118 | 0.0938 | 0.9375 | 0.9375 | 0.9374 | 0.0203 | 0.9541 | 0.9527 | 0.2577 |  | [[29, 3], [1, 31]] |
| breakhis_balanced_prototype | convnext | ce_calibrated_cost_min | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0437 | 1.0000 | 0.0000 | 0.5333 | 0.8750 | 0.5625 | 0.5625 | 0.4589 | 0.1264 | 0.9541 | 0.9527 | 0.1336 | 0.5000 | [[4, 28], [0, 32]] |
| breakhis_balanced_prototype | convnext | menon_logit_adjusted | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0203 | 0.9688 | 0.0312 | 0.9118 | 0.0938 | 0.9375 | 0.9375 | 0.9374 | 0.0203 | 0.9541 | 0.9527 | 0.2577 |  | [[29, 3], [1, 31]] |
| breakhis_balanced_prototype | convnext | menon_logit_adjusted | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0437 | 1.0000 | 0.0000 | 0.5333 | 0.8750 | 0.5625 | 0.5625 | 0.4589 | 0.1264 | 0.9541 | 0.9527 | 0.1336 | 0.5000 | [[4, 28], [0, 32]] |
| breakhis_balanced_prototype | convnext | nicme_hybrid | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 | 0.6924 | 0.6554 | 0.1396 |  | [[0, 32], [0, 32]] |
| breakhis_balanced_prototype | convnext | nicme_hybrid | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 | 0.6924 | 0.6554 | 0.0617 | 2.6500 | [[0, 32], [0, 32]] |
| breakhis_balanced_prototype | vit | ce | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0469 | 0.9062 | 0.0938 | 1.0000 | 0.0000 | 0.9531 | 0.9531 | 0.9530 | 0.0631 | 0.9619 | 0.9770 | 0.2855 |  | [[32, 0], [3, 29]] |
| breakhis_balanced_prototype | vit | ce | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0484 | 1.0000 | 0.0000 | 0.5079 | 0.9688 | 0.5156 | 0.5156 | 0.3671 | 0.1602 | 0.9619 | 0.9770 | 0.1660 | 0.5000 | [[1, 31], [0, 32]] |
| breakhis_balanced_prototype | vit | ce_calibrated_cost_min | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0469 | 0.9062 | 0.0938 | 1.0000 | 0.0000 | 0.9531 | 0.9531 | 0.9530 | 0.0631 | 0.9619 | 0.9770 | 0.2852 |  | [[32, 0], [3, 29]] |
| breakhis_balanced_prototype | vit | ce_calibrated_cost_min | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0484 | 1.0000 | 0.0000 | 0.5079 | 0.9688 | 0.5156 | 0.5156 | 0.3671 | 0.1602 | 0.9619 | 0.9770 | 0.1653 | 0.5000 | [[1, 31], [0, 32]] |
| breakhis_balanced_prototype | vit | menon_logit_adjusted | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0469 | 0.9062 | 0.0938 | 1.0000 | 0.0000 | 0.9531 | 0.9531 | 0.9530 | 0.0631 | 0.9619 | 0.9770 | 0.2855 |  | [[32, 0], [3, 29]] |
| breakhis_balanced_prototype | vit | menon_logit_adjusted | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0484 | 1.0000 | 0.0000 | 0.5079 | 0.9688 | 0.5156 | 0.5156 | 0.3671 | 0.1602 | 0.9619 | 0.9770 | 0.1661 | 0.5000 | [[1, 31], [0, 32]] |
| breakhis_balanced_prototype | vit | nicme_hybrid | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0234 | 0.9688 | 0.0312 | 0.8611 | 0.1562 | 0.9062 | 0.9062 | 0.9059 | 0.0234 | 0.9756 | 0.9845 | 0.1706 |  | [[27, 5], [1, 31]] |
| breakhis_balanced_prototype | vit | nicme_hybrid | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 | 0.9756 | 0.9845 | 0.1810 | 0.9000 | [[0, 32], [0, 32]] |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | ce | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.1672 | 0.6875 | 0.3125 | 0.7586 | 0.2188 | 0.7344 | 0.7344 | 0.7338 | 0.4998 | 0.8135 | 0.8252 | 0.2315 |  | [[25, 7], [10, 22]] |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | ce | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 | 0.8135 | 0.8252 | 0.2287 | 0.5000 | [[0, 32], [0, 32]] |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | ce_calibrated_cost_min | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.1672 | 0.6875 | 0.3125 | 0.7586 | 0.2188 | 0.7344 | 0.7344 | 0.7338 | 0.4998 | 0.8135 | 0.8252 | 0.2315 |  | [[25, 7], [10, 22]] |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | ce_calibrated_cost_min | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 | 0.8135 | 0.8252 | 0.2287 | 0.5000 | [[0, 32], [0, 32]] |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | menon_logit_adjusted | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.1672 | 0.6875 | 0.3125 | 0.7586 | 0.2188 | 0.7344 | 0.7344 | 0.7338 | 0.4998 | 0.8135 | 0.8252 | 0.2315 |  | [[25, 7], [10, 22]] |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | menon_logit_adjusted | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 | 0.8135 | 0.8252 | 0.2287 | 0.5000 | [[0, 32], [0, 32]] |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | nicme_hybrid | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 | 0.7520 | 0.7790 | 0.0096 |  | [[0, 32], [0, 32]] |
| breakhis_balanced_prototype | timm_dinov3_vit_lora | nicme_hybrid | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 | 0.7520 | 0.7790 | 0.0083 | 1.1500 | [[0, 32], [0, 32]] |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | ce | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.1359 | 0.7500 | 0.2500 | 0.7742 | 0.2188 | 0.7656 | 0.7656 | 0.7656 | 0.3367 | 0.8271 | 0.8369 | 0.1319 |  | [[25, 7], [8, 24]] |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | ce | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0578 | 0.9688 | 0.0312 | 0.5345 | 0.8438 | 0.5625 | 0.5625 | 0.4760 | 0.1405 | 0.8271 | 0.8369 | 0.0860 | 0.5000 | [[5, 27], [1, 31]] |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | ce_calibrated_cost_min | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.1359 | 0.7500 | 0.2500 | 0.7742 | 0.2188 | 0.7656 | 0.7656 | 0.7656 | 0.3367 | 0.8271 | 0.8369 | 0.1319 |  | [[25, 7], [8, 24]] |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0578 | 0.9688 | 0.0312 | 0.5345 | 0.8438 | 0.5625 | 0.5625 | 0.4760 | 0.1405 | 0.8271 | 0.8369 | 0.0860 | 0.5000 | [[5, 27], [1, 31]] |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | menon_logit_adjusted | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.1359 | 0.7500 | 0.2500 | 0.7742 | 0.2188 | 0.7656 | 0.7656 | 0.7656 | 0.3367 | 0.8271 | 0.8369 | 0.1319 |  | [[25, 7], [8, 24]] |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | menon_logit_adjusted | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0578 | 0.9688 | 0.0312 | 0.5345 | 0.8438 | 0.5625 | 0.5625 | 0.4760 | 0.1405 | 0.8271 | 0.8369 | 0.0860 | 0.5000 | [[5, 27], [1, 31]] |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | nicme_hybrid | argmax | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0609 | 0.9688 | 0.0312 | 0.5167 | 0.9062 | 0.5312 | 0.5312 | 0.4203 | 0.1625 | 0.5244 | 0.5613 | 0.1327 |  | [[3, 29], [1, 31]] |
| breakhis_balanced_prototype | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_cost_min | malignant | [[0.0,1.0],[10.0,0.0]] | 0.0500 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | 0.1725 | 0.5244 | 0.5613 | 0.0154 | 10.0000 | [[0, 32], [0, 32]] |
