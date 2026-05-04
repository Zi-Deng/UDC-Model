# stop3b_partial_spider_done Results And Scientific Read

Generated from local run logs and metrics files.

## Executive Summary

- Decision rows: `216`.
- Aggregate rows: `72`.
- Lower `selection_score` is better; floor satisfaction uses the configured `selection_accuracy_metric` for each dataset.
- Interpret natural/imbalanced results as deployment realism and decoupling stress tests, not replacements for balanced-dataset evidence.

## spider_target_majority

Target: `black_widow`. Floors: recall `0.95`, balanced_accuracy `0.75`.

### Top Overall Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | menon_logit_adjusted | calibrated_cost_min | 67% | 0.0156 +/- 0.0016 | 0.0155 +/- 0.0018 | 0.9956 +/- 0.0038 | 0.7544 +/- 0.0234 | 0.8750 +/- 0.0100 | 0.7544 +/- 0.0234 | 0.7971 |
| 2 | vit | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0170 +/- 0.0053 | 0.0162 +/- 0.0053 | 0.9933 +/- 0.0115 | 0.7733 +/- 0.0751 | 0.8833 +/- 0.0325 | 0.7733 +/- 0.0751 | 0.8108 |
| 3 | vit | cs_regularized_ce | calibrated_cost_min | 67% | 0.0180 +/- 0.0084 | 0.0147 +/- 0.0030 | 0.9978 +/- 0.0038 | 0.7389 +/- 0.0781 | 0.8683 +/- 0.0379 | 0.7389 +/- 0.0781 | 0.7770 |
| 4 | vit | nicme_hybrid | calibrated_threshold | 0% | 0.0205 +/- 0.0042 | 0.0170 +/- 0.0031 | 0.9978 +/- 0.0038 | 0.6922 +/- 0.0158 | 0.8450 +/- 0.0087 | 0.6922 +/- 0.0158 | 0.7304 |
| 5 | vit | nicme_hybrid | argmax | 67% | 0.0209 +/- 0.0063 | 0.0203 +/- 0.0069 | 0.9867 +/- 0.0133 | 0.7867 +/- 0.0664 | 0.8867 +/- 0.0275 | 0.7867 +/- 0.0664 | 0.8214 |
| 6 | vit | menon_logit_adjusted | calibrated_threshold | 33% | 0.0219 +/- 0.0049 | 0.0173 +/- 0.0018 | 0.9956 +/- 0.0077 | 0.7178 +/- 0.0851 | 0.8567 +/- 0.0388 | 0.7178 +/- 0.0851 | 0.7521 |
| 7 | vit | ce_calibrated_cost_min | calibrated_threshold | 67% | 0.0219 +/- 0.0096 | 0.0213 +/- 0.0102 | 0.9867 +/- 0.0176 | 0.7667 +/- 0.0567 | 0.8767 +/- 0.0202 | 0.7667 +/- 0.0567 | 0.8029 |
| 8 | vit | ce_calibrated_cost_min | calibrated_cost_min | 33% | 0.0220 +/- 0.0089 | 0.0178 +/- 0.0053 | 0.9956 +/- 0.0077 | 0.7078 +/- 0.0593 | 0.8517 +/- 0.0284 | 0.7078 +/- 0.0593 | 0.7442 |
| 9 | vit | cs_regularized_ce | calibrated_threshold | 0% | 0.0230 +/- 0.0042 | 0.0185 +/- 0.0036 | 0.9956 +/- 0.0077 | 0.6944 +/- 0.0468 | 0.8450 +/- 0.0200 | 0.6944 +/- 0.0468 | 0.7300 |
| 10 | vit | nicme_logit_adjustment | argmax | all | 0.0253 +/- 0.0133 | 0.0253 +/- 0.0133 | 0.9778 +/- 0.0234 | 0.8156 +/- 0.0746 | 0.8967 +/- 0.0257 | 0.8156 +/- 0.0746 | 0.8428 |

### Top NICME Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0170 +/- 0.0053 | 0.0162 +/- 0.0053 | 0.9933 +/- 0.0115 | 0.7733 +/- 0.0751 | 0.8833 +/- 0.0325 | 0.7733 +/- 0.0751 | 0.8108 |
| 2 | vit | nicme_hybrid | calibrated_threshold | 0% | 0.0205 +/- 0.0042 | 0.0170 +/- 0.0031 | 0.9978 +/- 0.0038 | 0.6922 +/- 0.0158 | 0.8450 +/- 0.0087 | 0.6922 +/- 0.0158 | 0.7304 |
| 3 | vit | nicme_hybrid | argmax | 67% | 0.0209 +/- 0.0063 | 0.0203 +/- 0.0069 | 0.9867 +/- 0.0133 | 0.7867 +/- 0.0664 | 0.8867 +/- 0.0275 | 0.7867 +/- 0.0664 | 0.8214 |
| 4 | vit | nicme_logit_adjustment | argmax | all | 0.0253 +/- 0.0133 | 0.0253 +/- 0.0133 | 0.9778 +/- 0.0234 | 0.8156 +/- 0.0746 | 0.8967 +/- 0.0257 | 0.8156 +/- 0.0746 | 0.8428 |
| 5 | vit | nicme_hybrid | calibrated_cost_min | 0% | 0.0567 +/- 0.0315 | 0.0225 +/- 0.0023 | 0.9978 +/- 0.0038 | 0.5822 +/- 0.0953 | 0.7900 +/- 0.0458 | 0.5822 +/- 0.0953 | 0.5649 |
| 6 | vit | nicme_logit_adjustment | calibrated_cost_min | 33% | 0.0617 +/- 0.0447 | 0.0200 +/- 0.0087 | 1.0000 +/- 0.0000 | 0.6000 +/- 0.1732 | 0.8000 +/- 0.0866 | 0.6000 +/- 0.1732 | 0.5670 |
| 7 | timm_dinov3_vit_lora | nicme_logit_adjustment | argmax | 0% | 0.0875 +/- 0.0000 | 0.0250 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.7500 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.4286 |
| 8 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_cost_min | 0% | 0.0875 +/- 0.0000 | 0.0250 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.7500 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.4286 |
| 9 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_threshold | 0% | 0.0875 +/- 0.0000 | 0.0250 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.7500 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.4286 |
| 10 | timm_dinov3_vit_lora | nicme_hybrid | argmax | 0% | 0.0875 +/- 0.0000 | 0.0250 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.7500 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.4286 |

### Strict All-Seed Floor Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | nicme_logit_adjustment | argmax | all | 0.0253 +/- 0.0133 | 0.0253 +/- 0.0133 | 0.9778 +/- 0.0234 | 0.8156 +/- 0.0746 | 0.8967 +/- 0.0257 | 0.8156 +/- 0.0746 | 0.8428 |
| 2 | vit | menon_logit_adjusted | argmax | all | 0.0255 +/- 0.0030 | 0.0255 +/- 0.0030 | 0.9756 +/- 0.0038 | 0.8444 +/- 0.0069 | 0.9100 +/- 0.0050 | 0.8444 +/- 0.0069 | 0.8703 |

### Mean-Floor-Compliant Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | menon_logit_adjusted | calibrated_cost_min | 67% | 0.0156 +/- 0.0016 | 0.0155 +/- 0.0018 | 0.9956 +/- 0.0038 | 0.7544 +/- 0.0234 | 0.8750 +/- 0.0100 | 0.7544 +/- 0.0234 | 0.7971 |
| 2 | vit | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0170 +/- 0.0053 | 0.0162 +/- 0.0053 | 0.9933 +/- 0.0115 | 0.7733 +/- 0.0751 | 0.8833 +/- 0.0325 | 0.7733 +/- 0.0751 | 0.8108 |
| 3 | vit | nicme_hybrid | argmax | 67% | 0.0209 +/- 0.0063 | 0.0203 +/- 0.0069 | 0.9867 +/- 0.0133 | 0.7867 +/- 0.0664 | 0.8867 +/- 0.0275 | 0.7867 +/- 0.0664 | 0.8214 |
| 4 | vit | ce_calibrated_cost_min | calibrated_threshold | 67% | 0.0219 +/- 0.0096 | 0.0213 +/- 0.0102 | 0.9867 +/- 0.0176 | 0.7667 +/- 0.0567 | 0.8767 +/- 0.0202 | 0.7667 +/- 0.0567 | 0.8029 |
| 5 | vit | nicme_logit_adjustment | argmax | all | 0.0253 +/- 0.0133 | 0.0253 +/- 0.0133 | 0.9778 +/- 0.0234 | 0.8156 +/- 0.0746 | 0.8967 +/- 0.0257 | 0.8156 +/- 0.0746 | 0.8428 |
| 6 | vit | menon_logit_adjusted | argmax | all | 0.0255 +/- 0.0030 | 0.0255 +/- 0.0030 | 0.9756 +/- 0.0038 | 0.8444 +/- 0.0069 | 0.9100 +/- 0.0050 | 0.8444 +/- 0.0069 | 0.8703 |
| 7 | vit | ce_calibrated_cost_min | argmax | 67% | 0.0355 +/- 0.0202 | 0.0352 +/- 0.0196 | 0.9622 +/- 0.0269 | 0.8444 +/- 0.0102 | 0.9033 +/- 0.0153 | 0.8444 +/- 0.0102 | 0.8638 |
| 8 | vit | ce | argmax | 33% | 0.0373 +/- 0.0280 | 0.0355 +/- 0.0264 | 0.9644 +/- 0.0391 | 0.8056 +/- 0.0869 | 0.8850 +/- 0.0377 | 0.8056 +/- 0.0869 | 0.8264 |

Best mean selection row: `vit + menon_logit_adjusted + calibrated_cost_min` with selection `0.0156`, nATC `0.0155`, target recall `0.9956`, and selected accuracy `0.7544`.

Best NICME row: `vit + nicme_logit_adjustment + calibrated_threshold` with selection `0.0170`, nATC `0.0162`, target recall `0.9933`, and selected accuracy `0.7733`.

## spider_target_minority

Target: `black_widow`. Floors: recall `0.95`, balanced_accuracy `0.75`.

### Top Overall Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | menon_logit_adjusted | calibrated_threshold | all | 0.0152 +/- 0.0077 | 0.0152 +/- 0.0077 | 0.9867 +/- 0.0231 | 0.9144 +/- 0.0241 | 0.8783 +/- 0.0247 | 0.9144 +/- 0.0241 | 0.8574 |
| 2 | vit | menon_logit_adjusted | calibrated_cost_min | all | 0.0197 +/- 0.0037 | 0.0197 +/- 0.0037 | 0.9867 +/- 0.0231 | 0.8844 +/- 0.0212 | 0.8333 +/- 0.0388 | 0.8844 +/- 0.0212 | 0.8121 |
| 3 | vit | nicme_logit_adjustment | calibrated_threshold | all | 0.0220 +/- 0.0059 | 0.0220 +/- 0.0059 | 0.9867 +/- 0.0115 | 0.8689 +/- 0.0386 | 0.8100 +/- 0.0589 | 0.8689 +/- 0.0386 | 0.7899 |
| 4 | vit | nicme_hybrid | calibrated_threshold | all | 0.0230 +/- 0.0066 | 0.0230 +/- 0.0066 | 0.9933 +/- 0.0115 | 0.8544 +/- 0.0417 | 0.7850 +/- 0.0626 | 0.8544 +/- 0.0417 | 0.7663 |
| 5 | vit | nicme_logit_adjustment | argmax | 33% | 0.0238 +/- 0.0117 | 0.0225 +/- 0.0098 | 0.9467 +/- 0.0306 | 0.9122 +/- 0.0337 | 0.8950 +/- 0.0397 | 0.9122 +/- 0.0337 | 0.8731 |
| 6 | vit | nicme_logit_adjustment | calibrated_cost_min | all | 0.0263 +/- 0.0057 | 0.0263 +/- 0.0057 | 0.9933 +/- 0.0115 | 0.8322 +/- 0.0299 | 0.7517 +/- 0.0425 | 0.8322 +/- 0.0299 | 0.7347 |
| 7 | vit | ce | calibrated_cost_min | 33% | 0.0267 +/- 0.0077 | 0.0253 +/- 0.0064 | 0.9467 +/- 0.0306 | 0.8933 +/- 0.0145 | 0.8667 +/- 0.0225 | 0.8933 +/- 0.0145 | 0.8425 |
| 8 | vit | ce_calibrated_cost_min | calibrated_threshold | 67% | 0.0272 +/- 0.0103 | 0.0260 +/- 0.0083 | 0.9733 +/- 0.0462 | 0.8578 +/- 0.0051 | 0.8000 +/- 0.0229 | 0.8578 +/- 0.0051 | 0.7783 |
| 9 | vit | cs_regularized_ce | calibrated_cost_min | 67% | 0.0283 +/- 0.0172 | 0.0250 +/- 0.0115 | 0.9600 +/- 0.0529 | 0.8800 +/- 0.0318 | 0.8400 +/- 0.0477 | 0.8800 +/- 0.0318 | 0.8169 |
| 10 | vit | ce_calibrated_cost_min | calibrated_cost_min | 33% | 0.0288 +/- 0.0105 | 0.0285 +/- 0.0103 | 0.9533 +/- 0.0231 | 0.8644 +/- 0.0462 | 0.8200 +/- 0.0614 | 0.8644 +/- 0.0462 | 0.7971 |

### Top NICME Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | nicme_logit_adjustment | calibrated_threshold | all | 0.0220 +/- 0.0059 | 0.0220 +/- 0.0059 | 0.9867 +/- 0.0115 | 0.8689 +/- 0.0386 | 0.8100 +/- 0.0589 | 0.8689 +/- 0.0386 | 0.7899 |
| 2 | vit | nicme_hybrid | calibrated_threshold | all | 0.0230 +/- 0.0066 | 0.0230 +/- 0.0066 | 0.9933 +/- 0.0115 | 0.8544 +/- 0.0417 | 0.7850 +/- 0.0626 | 0.8544 +/- 0.0417 | 0.7663 |
| 3 | vit | nicme_logit_adjustment | argmax | 33% | 0.0238 +/- 0.0117 | 0.0225 +/- 0.0098 | 0.9467 +/- 0.0306 | 0.9122 +/- 0.0337 | 0.8950 +/- 0.0397 | 0.9122 +/- 0.0337 | 0.8731 |
| 4 | vit | nicme_logit_adjustment | calibrated_cost_min | all | 0.0263 +/- 0.0057 | 0.0263 +/- 0.0057 | 0.9933 +/- 0.0115 | 0.8322 +/- 0.0299 | 0.7517 +/- 0.0425 | 0.8322 +/- 0.0299 | 0.7347 |
| 5 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0295 +/- 0.0079 | 0.0295 +/- 0.0079 | 1.0000 +/- 0.0000 | 0.8033 +/- 0.0524 | 0.7050 +/- 0.0786 | 0.8033 +/- 0.0524 | 0.6922 |
| 6 | vit | nicme_hybrid | argmax | 0% | 0.0343 +/- 0.0081 | 0.0297 +/- 0.0036 | 0.9200 +/- 0.0200 | 0.8956 +/- 0.0107 | 0.8833 +/- 0.0202 | 0.8956 +/- 0.0107 | 0.8582 |
| 7 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_cost_min | 0% | 0.0432 +/- 0.0042 | 0.0420 +/- 0.0031 | 1.0000 +/- 0.0000 | 0.7200 +/- 0.0208 | 0.5800 +/- 0.0312 | 0.7200 +/- 0.0208 | 0.5771 |
| 8 | timm_dinov3_vit_lora | nicme_hybrid | calibrated_threshold | 0% | 0.0531 +/- 0.0057 | 0.0510 +/- 0.0048 | 0.9600 +/- 0.0200 | 0.7067 +/- 0.0088 | 0.5800 +/- 0.0050 | 0.7067 +/- 0.0088 | 0.5757 |
| 9 | vit | nicme_hybrid | calibrated_cost_min | 67% | 0.0587 +/- 0.0648 | 0.0390 +/- 0.0309 | 1.0000 +/- 0.0000 | 0.7400 +/- 0.2060 | 0.6100 +/- 0.3090 | 0.7400 +/- 0.2060 | 0.5831 |
| 10 | timm_dinov3_vit_lora | nicme_logit_adjustment | argmax | 0% | 0.1249 +/- 0.0161 | 0.0498 +/- 0.0033 | 0.8133 +/- 0.0115 | 0.8856 +/- 0.0102 | 0.9217 +/- 0.0115 | 0.8856 +/- 0.0102 | 0.8935 |

### Strict All-Seed Floor Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | menon_logit_adjusted | calibrated_threshold | all | 0.0152 +/- 0.0077 | 0.0152 +/- 0.0077 | 0.9867 +/- 0.0231 | 0.9144 +/- 0.0241 | 0.8783 +/- 0.0247 | 0.9144 +/- 0.0241 | 0.8574 |
| 2 | vit | menon_logit_adjusted | calibrated_cost_min | all | 0.0197 +/- 0.0037 | 0.0197 +/- 0.0037 | 0.9867 +/- 0.0231 | 0.8844 +/- 0.0212 | 0.8333 +/- 0.0388 | 0.8844 +/- 0.0212 | 0.8121 |
| 3 | vit | nicme_logit_adjustment | calibrated_threshold | all | 0.0220 +/- 0.0059 | 0.0220 +/- 0.0059 | 0.9867 +/- 0.0115 | 0.8689 +/- 0.0386 | 0.8100 +/- 0.0589 | 0.8689 +/- 0.0386 | 0.7899 |
| 4 | vit | nicme_hybrid | calibrated_threshold | all | 0.0230 +/- 0.0066 | 0.0230 +/- 0.0066 | 0.9933 +/- 0.0115 | 0.8544 +/- 0.0417 | 0.7850 +/- 0.0626 | 0.8544 +/- 0.0417 | 0.7663 |
| 5 | vit | nicme_logit_adjustment | calibrated_cost_min | all | 0.0263 +/- 0.0057 | 0.0263 +/- 0.0057 | 0.9933 +/- 0.0115 | 0.8322 +/- 0.0299 | 0.7517 +/- 0.0425 | 0.8322 +/- 0.0299 | 0.7347 |

### Mean-Floor-Compliant Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | menon_logit_adjusted | calibrated_threshold | all | 0.0152 +/- 0.0077 | 0.0152 +/- 0.0077 | 0.9867 +/- 0.0231 | 0.9144 +/- 0.0241 | 0.8783 +/- 0.0247 | 0.9144 +/- 0.0241 | 0.8574 |
| 2 | vit | menon_logit_adjusted | calibrated_cost_min | all | 0.0197 +/- 0.0037 | 0.0197 +/- 0.0037 | 0.9867 +/- 0.0231 | 0.8844 +/- 0.0212 | 0.8333 +/- 0.0388 | 0.8844 +/- 0.0212 | 0.8121 |
| 3 | vit | nicme_logit_adjustment | calibrated_threshold | all | 0.0220 +/- 0.0059 | 0.0220 +/- 0.0059 | 0.9867 +/- 0.0115 | 0.8689 +/- 0.0386 | 0.8100 +/- 0.0589 | 0.8689 +/- 0.0386 | 0.7899 |
| 4 | vit | nicme_hybrid | calibrated_threshold | all | 0.0230 +/- 0.0066 | 0.0230 +/- 0.0066 | 0.9933 +/- 0.0115 | 0.8544 +/- 0.0417 | 0.7850 +/- 0.0626 | 0.8544 +/- 0.0417 | 0.7663 |
| 5 | vit | nicme_logit_adjustment | calibrated_cost_min | all | 0.0263 +/- 0.0057 | 0.0263 +/- 0.0057 | 0.9933 +/- 0.0115 | 0.8322 +/- 0.0299 | 0.7517 +/- 0.0425 | 0.8322 +/- 0.0299 | 0.7347 |
| 6 | vit | ce_calibrated_cost_min | calibrated_threshold | 67% | 0.0272 +/- 0.0103 | 0.0260 +/- 0.0083 | 0.9733 +/- 0.0462 | 0.8578 +/- 0.0051 | 0.8000 +/- 0.0229 | 0.8578 +/- 0.0051 | 0.7783 |
| 7 | vit | cs_regularized_ce | calibrated_cost_min | 67% | 0.0283 +/- 0.0172 | 0.0250 +/- 0.0115 | 0.9600 +/- 0.0529 | 0.8800 +/- 0.0318 | 0.8400 +/- 0.0477 | 0.8800 +/- 0.0318 | 0.8169 |
| 8 | vit | ce_calibrated_cost_min | calibrated_cost_min | 33% | 0.0288 +/- 0.0105 | 0.0285 +/- 0.0103 | 0.9533 +/- 0.0231 | 0.8644 +/- 0.0462 | 0.8200 +/- 0.0614 | 0.8644 +/- 0.0462 | 0.7971 |
| 9 | timm_dinov3_vit_lora | nicme_logit_adjustment | calibrated_threshold | 67% | 0.0295 +/- 0.0079 | 0.0295 +/- 0.0079 | 1.0000 +/- 0.0000 | 0.8033 +/- 0.0524 | 0.7050 +/- 0.0786 | 0.8033 +/- 0.0524 | 0.6922 |
| 10 | vit | cs_regularized_ce | calibrated_threshold | 67% | 0.0300 +/- 0.0160 | 0.0267 +/- 0.0104 | 0.9667 +/- 0.0577 | 0.8611 +/- 0.0234 | 0.8083 +/- 0.0473 | 0.8611 +/- 0.0234 | 0.7862 |

Best mean selection row: `vit + menon_logit_adjusted + calibrated_threshold` with selection `0.0152`, nATC `0.0152`, target recall `0.9867`, and selected accuracy `0.9144`.

Best NICME row: `vit + nicme_logit_adjustment + calibrated_threshold` with selection `0.0220`, nATC `0.0220`, target recall `0.9867`, and selected accuracy `0.8689`.

## Artifacts

- Full decision rows: `results/stop3b_imbalance_decoupling/partial_spider_done/stop3b_partial_spider_done_full_decision_rows.csv`
- Aggregate summary: `results/stop3b_imbalance_decoupling/partial_spider_done/stop3b_partial_spider_done_aggregate_summary.csv`
- Ranked summary: `results/stop3b_imbalance_decoupling/partial_spider_done/stop3b_partial_spider_done_ranked_summary.csv`
