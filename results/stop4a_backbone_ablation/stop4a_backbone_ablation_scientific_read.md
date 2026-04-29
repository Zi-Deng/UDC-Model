# stop4a_backbone_ablation Results And Scientific Read

Generated from local run logs and metrics files.

## Executive Summary

- Decision rows: `108`.
- Aggregate rows: `36`.
- Lower `selection_score` is better; floor satisfaction uses the configured `selection_accuracy_metric` for each dataset.
- Interpret natural/imbalanced results as deployment realism and decoupling stress tests, not replacements for balanced-dataset evidence.

## breakhis_balanced

Target: `malignant`. Floors: recall `0.97`, accuracy `0.85`.

### Top Overall Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | all | 0.0145 +/- 0.0017 | 0.0145 +/- 0.0017 | 0.9965 +/- 0.0035 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8691 |
| 2 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_cost_min | 67% | 0.0167 +/- 0.0028 | 0.0164 +/- 0.0023 | 0.9976 +/- 0.0041 | 0.8469 +/- 0.0256 | 0.8469 +/- 0.0256 | 0.8469 +/- 0.0256 | 0.8431 |
| 3 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | all | 0.0171 +/- 0.0038 | 0.0171 +/- 0.0038 | 0.9799 +/- 0.0054 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9193 |
| 4 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_cost_min | 33% | 0.0180 +/- 0.0052 | 0.0170 +/- 0.0035 | 0.9976 +/- 0.0020 | 0.8410 +/- 0.0438 | 0.8410 +/- 0.0438 | 0.8410 +/- 0.0438 | 0.8363 |
| 5 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_threshold | all | 0.0190 +/- 0.0028 | 0.0190 +/- 0.0028 | 0.9846 +/- 0.0125 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8772 |
| 6 | convnext | ce_calibrated_cost_min | calibrated_threshold | 33% | 0.0261 +/- 0.0160 | 0.0241 +/- 0.0129 | 0.9764 +/- 0.0379 | 0.8652 +/- 0.0426 | 0.8652 +/- 0.0426 | 0.8652 +/- 0.0426 | 0.8625 |
| 7 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_threshold | 67% | 0.0262 +/- 0.0133 | 0.0247 +/- 0.0107 | 0.9669 +/- 0.0275 | 0.9019 +/- 0.0190 | 0.9019 +/- 0.0190 | 0.9019 +/- 0.0190 | 0.9013 |
| 8 | convnext | nicme_logit_adjustment | calibrated_threshold | 0% | 0.0299 +/- 0.0097 | 0.0274 +/- 0.0100 | 0.9823 +/- 0.0248 | 0.8056 +/- 0.0161 | 0.8056 +/- 0.0161 | 0.8056 +/- 0.0161 | 0.7989 |
| 9 | convnext | ce_calibrated_cost_min | calibrated_cost_min | 0% | 0.0309 +/- 0.0126 | 0.0267 +/- 0.0108 | 0.9775 +/- 0.0389 | 0.8345 +/- 0.0733 | 0.8345 +/- 0.0733 | 0.8345 +/- 0.0733 | 0.8282 |
| 10 | convnext | nicme_hybrid | calibrated_threshold | 33% | 0.0330 +/- 0.0144 | 0.0312 +/- 0.0121 | 0.9622 +/- 0.0228 | 0.8582 +/- 0.0419 | 0.8582 +/- 0.0419 | 0.8582 +/- 0.0419 | 0.8562 |

### Top NICME Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | all | 0.0145 +/- 0.0017 | 0.0145 +/- 0.0017 | 0.9965 +/- 0.0035 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8691 |
| 2 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_cost_min | 67% | 0.0167 +/- 0.0028 | 0.0164 +/- 0.0023 | 0.9976 +/- 0.0041 | 0.8469 +/- 0.0256 | 0.8469 +/- 0.0256 | 0.8469 +/- 0.0256 | 0.8431 |
| 3 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | all | 0.0171 +/- 0.0038 | 0.0171 +/- 0.0038 | 0.9799 +/- 0.0054 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9193 |
| 4 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_threshold | 67% | 0.0262 +/- 0.0133 | 0.0247 +/- 0.0107 | 0.9669 +/- 0.0275 | 0.9019 +/- 0.0190 | 0.9019 +/- 0.0190 | 0.9019 +/- 0.0190 | 0.9013 |
| 5 | convnext | nicme_logit_adjustment | calibrated_threshold | 0% | 0.0299 +/- 0.0097 | 0.0274 +/- 0.0100 | 0.9823 +/- 0.0248 | 0.8056 +/- 0.0161 | 0.8056 +/- 0.0161 | 0.8056 +/- 0.0161 | 0.7989 |
| 6 | convnext | nicme_hybrid | calibrated_threshold | 33% | 0.0330 +/- 0.0144 | 0.0312 +/- 0.0121 | 0.9622 +/- 0.0228 | 0.8582 +/- 0.0419 | 0.8582 +/- 0.0419 | 0.8582 +/- 0.0419 | 0.8562 |
| 7 | convnext | nicme_logit_adjustment | calibrated_cost_min | 0% | 0.0432 +/- 0.0363 | 0.0272 +/- 0.0110 | 0.9941 +/- 0.0054 | 0.7547 +/- 0.1024 | 0.7547 +/- 0.1024 | 0.7547 +/- 0.1024 | 0.7329 |
| 8 | timm_dinov3_convnext_lora | nicme_hybrid | argmax | 33% | 0.0544 +/- 0.0489 | 0.0395 +/- 0.0269 | 0.9338 +/- 0.0623 | 0.9031 +/- 0.0161 | 0.9031 +/- 0.0161 | 0.9031 +/- 0.0161 | 0.9026 |
| 9 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_cost_min | 33% | 0.0564 +/- 0.0551 | 0.0290 +/- 0.0123 | 0.9941 +/- 0.0054 | 0.7364 +/- 0.1478 | 0.7364 +/- 0.1478 | 0.7364 +/- 0.1478 | 0.7001 |
| 10 | convnext | nicme_logit_adjustment | argmax | 67% | 0.0803 +/- 0.1047 | 0.0455 +/- 0.0446 | 0.9243 +/- 0.1004 | 0.8853 +/- 0.0208 | 0.8853 +/- 0.0208 | 0.8853 +/- 0.0208 | 0.8842 |

### Strict All-Seed Floor Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | all | 0.0145 +/- 0.0017 | 0.0145 +/- 0.0017 | 0.9965 +/- 0.0035 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8691 |
| 2 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | all | 0.0171 +/- 0.0038 | 0.0171 +/- 0.0038 | 0.9799 +/- 0.0054 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9193 |
| 3 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_threshold | all | 0.0190 +/- 0.0028 | 0.0190 +/- 0.0028 | 0.9846 +/- 0.0125 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8772 |

### Mean-Floor-Compliant Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | all | 0.0145 +/- 0.0017 | 0.0145 +/- 0.0017 | 0.9965 +/- 0.0035 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8712 +/- 0.0104 | 0.8691 |
| 2 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | all | 0.0171 +/- 0.0038 | 0.0171 +/- 0.0038 | 0.9799 +/- 0.0054 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9196 +/- 0.0202 | 0.9193 |
| 3 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_threshold | all | 0.0190 +/- 0.0028 | 0.0190 +/- 0.0028 | 0.9846 +/- 0.0125 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8788 +/- 0.0284 | 0.8772 |
| 4 | convnext | ce_calibrated_cost_min | calibrated_threshold | 33% | 0.0261 +/- 0.0160 | 0.0241 +/- 0.0129 | 0.9764 +/- 0.0379 | 0.8652 +/- 0.0426 | 0.8652 +/- 0.0426 | 0.8652 +/- 0.0426 | 0.8625 |

Best mean selection row: `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` with selection `0.0145`, nATC `0.0145`, target recall `0.9965`, and selected accuracy `0.8712`.

Best NICME row: `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` with selection `0.0145`, nATC `0.0145`, target recall `0.9965`, and selected accuracy `0.8712`.

## spider_balanced

Target: `black_widow`. Floors: recall `0.95`, accuracy `0.8`.

### Top Overall Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | nicme_hybrid | calibrated_threshold | all | 0.0183 +/- 0.0026 | 0.0183 +/- 0.0026 | 0.9911 +/- 0.0038 | 0.8567 +/- 0.0400 | 0.8567 +/- 0.0400 | 0.8567 +/- 0.0400 | 0.8535 |
| 2 | convnext | ce_calibrated_cost_min | calibrated_cost_min | all | 0.0184 +/- 0.0019 | 0.0184 +/- 0.0019 | 0.9956 +/- 0.0077 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8307 |
| 3 | convnext | ce_calibrated_cost_min | calibrated_threshold | all | 0.0193 +/- 0.0023 | 0.0193 +/- 0.0023 | 0.9889 +/- 0.0102 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8539 |
| 4 | convnext | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0198 +/- 0.0013 | 0.0198 +/- 0.0013 | 0.9933 +/- 0.0067 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8270 |
| 5 | convnext | nicme_logit_adjustment | calibrated_cost_min | 33% | 0.0220 +/- 0.0024 | 0.0216 +/- 0.0018 | 0.9978 +/- 0.0038 | 0.7944 +/- 0.0347 | 0.7944 +/- 0.0347 | 0.7944 +/- 0.0347 | 0.7849 |
| 6 | convnext | nicme_hybrid | calibrated_cost_min | 33% | 0.0246 +/- 0.0048 | 0.0233 +/- 0.0031 | 0.9978 +/- 0.0038 | 0.7767 +/- 0.0321 | 0.7767 +/- 0.0321 | 0.7767 +/- 0.0321 | 0.7646 |
| 7 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_threshold | 67% | 0.0248 +/- 0.0033 | 0.0243 +/- 0.0034 | 0.9889 +/- 0.0102 | 0.8067 +/- 0.0418 | 0.8067 +/- 0.0418 | 0.8067 +/- 0.0418 | 0.7991 |
| 8 | convnext | ce_calibrated_cost_min | argmax | all | 0.0252 +/- 0.0019 | 0.0252 +/- 0.0019 | 0.9622 +/- 0.0077 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9176 |
| 9 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | all | 0.0258 +/- 0.0030 | 0.0258 +/- 0.0030 | 0.9778 +/- 0.0077 | 0.8422 +/- 0.0069 | 0.8422 +/- 0.0069 | 0.8422 +/- 0.0069 | 0.8392 |
| 10 | convnext | nicme_hybrid | argmax | all | 0.0261 +/- 0.0049 | 0.0261 +/- 0.0049 | 0.9622 +/- 0.0102 | 0.9089 +/- 0.0051 | 0.9089 +/- 0.0051 | 0.9089 +/- 0.0051 | 0.9086 |

### Top NICME Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | nicme_hybrid | calibrated_threshold | all | 0.0183 +/- 0.0026 | 0.0183 +/- 0.0026 | 0.9911 +/- 0.0038 | 0.8567 +/- 0.0400 | 0.8567 +/- 0.0400 | 0.8567 +/- 0.0400 | 0.8535 |
| 2 | convnext | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0198 +/- 0.0013 | 0.0198 +/- 0.0013 | 0.9933 +/- 0.0067 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8270 |
| 3 | convnext | nicme_logit_adjustment | calibrated_cost_min | 33% | 0.0220 +/- 0.0024 | 0.0216 +/- 0.0018 | 0.9978 +/- 0.0038 | 0.7944 +/- 0.0347 | 0.7944 +/- 0.0347 | 0.7944 +/- 0.0347 | 0.7849 |
| 4 | convnext | nicme_hybrid | calibrated_cost_min | 33% | 0.0246 +/- 0.0048 | 0.0233 +/- 0.0031 | 0.9978 +/- 0.0038 | 0.7767 +/- 0.0321 | 0.7767 +/- 0.0321 | 0.7767 +/- 0.0321 | 0.7646 |
| 5 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | all | 0.0258 +/- 0.0030 | 0.0258 +/- 0.0030 | 0.9778 +/- 0.0077 | 0.8422 +/- 0.0069 | 0.8422 +/- 0.0069 | 0.8422 +/- 0.0069 | 0.8392 |
| 6 | convnext | nicme_hybrid | argmax | all | 0.0261 +/- 0.0049 | 0.0261 +/- 0.0049 | 0.9622 +/- 0.0102 | 0.9089 +/- 0.0051 | 0.9089 +/- 0.0051 | 0.9089 +/- 0.0051 | 0.9086 |
| 7 | convnext | nicme_logit_adjustment | argmax | 67% | 0.0265 +/- 0.0060 | 0.0264 +/- 0.0060 | 0.9622 +/- 0.0139 | 0.9056 +/- 0.0084 | 0.9056 +/- 0.0084 | 0.9056 +/- 0.0084 | 0.9052 |
| 8 | timm_dinov3_convnext_lora | nicme_logit_adjustment | calibrated_threshold | 0% | 0.0283 +/- 0.0026 | 0.0267 +/- 0.0021 | 0.9933 +/- 0.0067 | 0.7633 +/- 0.0208 | 0.7633 +/- 0.0208 | 0.7633 +/- 0.0208 | 0.7498 |
| 9 | timm_dinov3_convnext_lora | nicme_hybrid | argmax | 33% | 0.0314 +/- 0.0113 | 0.0310 +/- 0.0110 | 0.9778 +/- 0.0214 | 0.7900 +/- 0.0233 | 0.7900 +/- 0.0233 | 0.7900 +/- 0.0233 | 0.7822 |
| 10 | timm_dinov3_convnext_lora | nicme_hybrid | calibrated_threshold | 0% | 0.0330 +/- 0.0056 | 0.0291 +/- 0.0042 | 0.9933 +/- 0.0067 | 0.7389 +/- 0.0126 | 0.7389 +/- 0.0126 | 0.7389 +/- 0.0126 | 0.7208 |

### Strict All-Seed Floor Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | nicme_hybrid | calibrated_threshold | all | 0.0183 +/- 0.0026 | 0.0183 +/- 0.0026 | 0.9911 +/- 0.0038 | 0.8567 +/- 0.0400 | 0.8567 +/- 0.0400 | 0.8567 +/- 0.0400 | 0.8535 |
| 2 | convnext | ce_calibrated_cost_min | calibrated_cost_min | all | 0.0184 +/- 0.0019 | 0.0184 +/- 0.0019 | 0.9956 +/- 0.0077 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8307 |
| 3 | convnext | ce_calibrated_cost_min | calibrated_threshold | all | 0.0193 +/- 0.0023 | 0.0193 +/- 0.0023 | 0.9889 +/- 0.0102 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8539 |
| 4 | convnext | ce_calibrated_cost_min | argmax | all | 0.0252 +/- 0.0019 | 0.0252 +/- 0.0019 | 0.9622 +/- 0.0077 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9176 |
| 5 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | all | 0.0258 +/- 0.0030 | 0.0258 +/- 0.0030 | 0.9778 +/- 0.0077 | 0.8422 +/- 0.0069 | 0.8422 +/- 0.0069 | 0.8422 +/- 0.0069 | 0.8392 |
| 6 | convnext | nicme_hybrid | argmax | all | 0.0261 +/- 0.0049 | 0.0261 +/- 0.0049 | 0.9622 +/- 0.0102 | 0.9089 +/- 0.0051 | 0.9089 +/- 0.0051 | 0.9089 +/- 0.0051 | 0.9086 |

### Mean-Floor-Compliant Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | nicme_hybrid | calibrated_threshold | all | 0.0183 +/- 0.0026 | 0.0183 +/- 0.0026 | 0.9911 +/- 0.0038 | 0.8567 +/- 0.0400 | 0.8567 +/- 0.0400 | 0.8567 +/- 0.0400 | 0.8535 |
| 2 | convnext | ce_calibrated_cost_min | calibrated_cost_min | all | 0.0184 +/- 0.0019 | 0.0184 +/- 0.0019 | 0.9956 +/- 0.0077 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8307 |
| 3 | convnext | ce_calibrated_cost_min | calibrated_threshold | all | 0.0193 +/- 0.0023 | 0.0193 +/- 0.0023 | 0.9889 +/- 0.0102 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8539 |
| 4 | convnext | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0198 +/- 0.0013 | 0.0198 +/- 0.0013 | 0.9933 +/- 0.0067 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8270 |
| 5 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | calibrated_threshold | 67% | 0.0248 +/- 0.0033 | 0.0243 +/- 0.0034 | 0.9889 +/- 0.0102 | 0.8067 +/- 0.0418 | 0.8067 +/- 0.0418 | 0.8067 +/- 0.0418 | 0.7991 |
| 6 | convnext | ce_calibrated_cost_min | argmax | all | 0.0252 +/- 0.0019 | 0.0252 +/- 0.0019 | 0.9622 +/- 0.0077 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9176 |
| 7 | timm_dinov3_convnext_lora | nicme_logit_adjustment | argmax | all | 0.0258 +/- 0.0030 | 0.0258 +/- 0.0030 | 0.9778 +/- 0.0077 | 0.8422 +/- 0.0069 | 0.8422 +/- 0.0069 | 0.8422 +/- 0.0069 | 0.8392 |
| 8 | convnext | nicme_hybrid | argmax | all | 0.0261 +/- 0.0049 | 0.0261 +/- 0.0049 | 0.9622 +/- 0.0102 | 0.9089 +/- 0.0051 | 0.9089 +/- 0.0051 | 0.9089 +/- 0.0051 | 0.9086 |
| 9 | convnext | nicme_logit_adjustment | argmax | 67% | 0.0265 +/- 0.0060 | 0.0264 +/- 0.0060 | 0.9622 +/- 0.0139 | 0.9056 +/- 0.0084 | 0.9056 +/- 0.0084 | 0.9056 +/- 0.0084 | 0.9052 |
| 10 | timm_dinov3_convnext_lora | ce_calibrated_cost_min | argmax | 67% | 0.0336 +/- 0.0051 | 0.0334 +/- 0.0049 | 0.9511 +/- 0.0102 | 0.8856 +/- 0.0255 | 0.8856 +/- 0.0255 | 0.8856 +/- 0.0255 | 0.8849 |

Best mean selection row: `convnext + nicme_hybrid + calibrated_threshold` with selection `0.0183`, nATC `0.0183`, target recall `0.9911`, and selected accuracy `0.8567`.

Best NICME row: `convnext + nicme_hybrid + calibrated_threshold` with selection `0.0183`, nATC `0.0183`, target recall `0.9911`, and selected accuracy `0.8567`.

## Artifacts

- Full decision rows: `results/stop4a_backbone_ablation/stop4a_backbone_ablation_full_decision_rows.csv`
- Aggregate summary: `results/stop4a_backbone_ablation/stop4a_backbone_ablation_aggregate_summary.csv`
- Ranked summary: `results/stop4a_backbone_ablation/stop4a_backbone_ablation_ranked_summary.csv`
