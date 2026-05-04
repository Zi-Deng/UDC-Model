# stop4b_breakhis_cost_ratio Results And Scientific Read

Generated from local run logs and metrics files.

## Executive Summary

- Decision rows: `135`.
- Aggregate rows: `45`.
- Lower `selection_score` is better; floor satisfaction uses the configured `selection_accuracy_metric` for each dataset.
- Interpret natural/imbalanced results as deployment realism and decoupling stress tests, not replacements for balanced-dataset evidence.

## breakhis_balanced

Target: `malignant`. Floors: recall `0.97`, accuracy `0.85`.

### Top Overall Rows

| rank | model | method | mode | cost ratio | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_threshold | 20.0000 | 67% | 0.0114 +/- 0.0024 | 0.0112 +/- 0.0025 | 0.9929 +/- 0.0061 | 0.8434 +/- 0.0181 | 0.8434 +/- 0.0181 | 0.8434 +/- 0.0181 | 0.8397 |
| 2 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_cost_min | 20.0000 | 33% | 0.0133 +/- 0.0055 | 0.0105 +/- 0.0014 | 0.9976 +/- 0.0020 | 0.8121 +/- 0.0453 | 0.8121 +/- 0.0453 | 0.8121 +/- 0.0453 | 0.8044 |
| 3 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | 10.0000 | all | 0.0145 +/- 0.0017 | 0.0145 +/- 0.0017 | 0.9965 +/- 0.0035 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8691 |
| 4 | timm_dinov3_convnext_lora | nicme_hybrid | argmax | 20.0000 | 67% | 0.0166 +/- 0.0078 | 0.0165 +/- 0.0076 | 0.9787 +/- 0.0155 | 0.8729 +/- 0.0091 | 0.8729 +/- 0.0091 | 0.8729 +/- 0.0091 | 0.8715 |
| 5 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_cost_min | 10.0000 | 67% | 0.0167 +/- 0.0028 | 0.0164 +/- 0.0023 | 0.9976 +/- 0.0041 | 0.8469 +/- 0.0256 | 0.8469 +/- 0.0256 | 0.8469 +/- 0.0256 | 0.8431 |
| 6 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_threshold | 20.0000 | 33% | 0.0170 +/- 0.0096 | 0.0167 +/- 0.0090 | 0.9787 +/- 0.0221 | 0.8688 +/- 0.0325 | 0.8688 +/- 0.0325 | 0.8688 +/- 0.0325 | 0.8667 |
| 7 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | 10.0000 | all | 0.0171 +/- 0.0038 | 0.0171 +/- 0.0038 | 0.9799 +/- 0.0054 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9193 |
| 8 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_cost_min | 10.0000 | 33% | 0.0180 +/- 0.0052 | 0.0170 +/- 0.0035 | 0.9976 +/- 0.0020 | 0.8410 +/- 0.0438 | 0.8410 +/- 0.0438 | 0.8410 +/- 0.0438 | 0.8363 |
| 9 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_threshold | 10.0000 | all | 0.0190 +/- 0.0028 | 0.0190 +/- 0.0028 | 0.9846 +/- 0.0125 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8772 |
| 10 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | 20.0000 | 67% | 0.0233 +/- 0.0233 | 0.0206 +/- 0.0188 | 0.9693 +/- 0.0389 | 0.8794 +/- 0.0081 | 0.8794 +/- 0.0081 | 0.8794 +/- 0.0081 | 0.8784 |

### Top NICME Rows

| rank | model | method | mode | cost ratio | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | 10.0000 | all | 0.0145 +/- 0.0017 | 0.0145 +/- 0.0017 | 0.9965 +/- 0.0035 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8691 |
| 2 | timm_dinov3_convnext_lora | nicme_hybrid | argmax | 20.0000 | 67% | 0.0166 +/- 0.0078 | 0.0165 +/- 0.0076 | 0.9787 +/- 0.0155 | 0.8729 +/- 0.0091 | 0.8729 +/- 0.0091 | 0.8729 +/- 0.0091 | 0.8715 |
| 3 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_cost_min | 10.0000 | 67% | 0.0167 +/- 0.0028 | 0.0164 +/- 0.0023 | 0.9976 +/- 0.0041 | 0.8469 +/- 0.0256 | 0.8469 +/- 0.0256 | 0.8469 +/- 0.0256 | 0.8431 |
| 4 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_threshold | 20.0000 | 33% | 0.0170 +/- 0.0096 | 0.0167 +/- 0.0090 | 0.9787 +/- 0.0221 | 0.8688 +/- 0.0325 | 0.8688 +/- 0.0325 | 0.8688 +/- 0.0325 | 0.8667 |
| 5 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | 10.0000 | all | 0.0171 +/- 0.0038 | 0.0171 +/- 0.0038 | 0.9799 +/- 0.0054 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9193 |
| 6 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | 20.0000 | 67% | 0.0233 +/- 0.0233 | 0.0206 +/- 0.0188 | 0.9693 +/- 0.0389 | 0.8794 +/- 0.0081 | 0.8794 +/- 0.0081 | 0.8794 +/- 0.0081 | 0.8784 |
| 7 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_threshold | 10.0000 | 67% | 0.0262 +/- 0.0133 | 0.0247 +/- 0.0107 | 0.9669 +/- 0.0275 | 0.9019 +/- 0.0190 | 0.9019 +/- 0.0190 | 0.9019 +/- 0.0190 | 0.9013 |
| 8 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | 5.0000 | 33% | 0.0274 +/- 0.0088 | 0.0270 +/- 0.0083 | 0.9669 +/- 0.0168 | 0.9314 +/- 0.0091 | 0.9314 +/- 0.0091 | 0.9314 +/- 0.0091 | 0.9314 |
| 9 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_threshold | 5.0000 | 67% | 0.0282 +/- 0.0098 | 0.0281 +/- 0.0098 | 0.9752 +/- 0.0106 | 0.9090 +/- 0.0297 | 0.9090 +/- 0.0297 | 0.9090 +/- 0.0297 | 0.9085 |
| 10 | timm_dinov3_convnext_lora | nicme_hybrid | argmax | 5.0000 | 67% | 0.0286 +/- 0.0095 | 0.0286 +/- 0.0095 | 0.9728 +/- 0.0074 | 0.9113 +/- 0.0354 | 0.9113 +/- 0.0354 | 0.9113 +/- 0.0354 | 0.9109 |

### Strict All-Seed Floor Rows

| rank | model | method | mode | cost ratio | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | 10.0000 | all | 0.0145 +/- 0.0017 | 0.0145 +/- 0.0017 | 0.9965 +/- 0.0035 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8691 |
| 2 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | 10.0000 | all | 0.0171 +/- 0.0038 | 0.0171 +/- 0.0038 | 0.9799 +/- 0.0054 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9193 |
| 3 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_threshold | 10.0000 | all | 0.0190 +/- 0.0028 | 0.0190 +/- 0.0028 | 0.9846 +/- 0.0125 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8772 |
| 4 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_cost_min | 5.0000 | all | 0.0288 +/- 0.0067 | 0.0288 +/- 0.0067 | 0.9846 +/- 0.0108 | 0.8865 +/- 0.0123 | 0.8865 +/- 0.0123 | 0.8865 +/- 0.0123 | 0.8854 |

### Mean-Floor-Compliant Rows

| rank | model | method | mode | cost ratio | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | 10.0000 | all | 0.0145 +/- 0.0017 | 0.0145 +/- 0.0017 | 0.9965 +/- 0.0035 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8691 |
| 2 | timm_dinov3_convnext_lora | nicme_hybrid | argmax | 20.0000 | 67% | 0.0166 +/- 0.0078 | 0.0165 +/- 0.0076 | 0.9787 +/- 0.0155 | 0.8729 +/- 0.0091 | 0.8729 +/- 0.0091 | 0.8729 +/- 0.0091 | 0.8715 |
| 3 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_threshold | 20.0000 | 33% | 0.0170 +/- 0.0096 | 0.0167 +/- 0.0090 | 0.9787 +/- 0.0221 | 0.8688 +/- 0.0325 | 0.8688 +/- 0.0325 | 0.8688 +/- 0.0325 | 0.8667 |
| 4 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | 10.0000 | all | 0.0171 +/- 0.0038 | 0.0171 +/- 0.0038 | 0.9799 +/- 0.0054 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9193 |
| 5 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_threshold | 10.0000 | all | 0.0190 +/- 0.0028 | 0.0190 +/- 0.0028 | 0.9846 +/- 0.0125 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8772 |
| 6 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_threshold | 5.0000 | 67% | 0.0282 +/- 0.0098 | 0.0281 +/- 0.0098 | 0.9752 +/- 0.0106 | 0.9090 +/- 0.0297 | 0.9090 +/- 0.0297 | 0.9090 +/- 0.0297 | 0.9085 |
| 7 | timm_dinov3_convnext_lora | nicme_hybrid | argmax | 5.0000 | 67% | 0.0286 +/- 0.0095 | 0.0286 +/- 0.0095 | 0.9728 +/- 0.0074 | 0.9113 +/- 0.0354 | 0.9113 +/- 0.0354 | 0.9113 +/- 0.0354 | 0.9109 |
| 8 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_cost_min | 5.0000 | 67% | 0.0288 +/- 0.0060 | 0.0287 +/- 0.0059 | 0.9799 +/- 0.0168 | 0.8966 +/- 0.0121 | 0.8966 +/- 0.0121 | 0.8966 +/- 0.0121 | 0.8958 |
| 9 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_cost_min | 5.0000 | all | 0.0288 +/- 0.0067 | 0.0288 +/- 0.0067 | 0.9846 +/- 0.0108 | 0.8865 +/- 0.0123 | 0.8865 +/- 0.0123 | 0.8865 +/- 0.0123 | 0.8854 |
| 10 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | 5.0000 | 67% | 0.0298 +/- 0.0083 | 0.0296 +/- 0.0081 | 0.9752 +/- 0.0177 | 0.9019 +/- 0.0194 | 0.9019 +/- 0.0194 | 0.9019 +/- 0.0194 | 0.9013 |

Best mean selection row: `timm_dinov3_convnext_lora + ce_calibrated_cost_min + calibrated_threshold` with selection `0.0114`, nATC `0.0112`, target recall `0.9929`, and selected accuracy `0.8434`.

Best NICME row: `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` with selection `0.0145`, nATC `0.0145`, target recall `0.9965`, and selected accuracy `0.8712`.

## Artifacts

- Full decision rows: `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_full_decision_rows.csv`
- Aggregate summary: `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_aggregate_summary.csv`
- Ranked summary: `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_ranked_summary.csv`
