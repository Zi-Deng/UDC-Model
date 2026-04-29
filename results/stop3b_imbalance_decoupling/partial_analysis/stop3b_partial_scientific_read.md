# stop3b_partial Results And Scientific Read

Generated from local run logs and metrics files.

## Executive Summary

- Decision rows: `111`.
- Aggregate rows: `39`.
- Lower `selection_score` is better; floor satisfaction uses the configured `selection_accuracy_metric` for each dataset.
- Interpret natural/imbalanced results as deployment realism and decoupling stress tests, not replacements for balanced-dataset evidence.

## spider_target_majority

Target: `black_widow`. Floors: recall `0.95`, balanced_accuracy `0.75`.

### Top Overall Rows

| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vit | ce | calibrated_threshold | 0% | 0.0209 +/- 0.0000 | 0.0160 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.6800 +/- 0.0000 | 0.8400 +/- 0.0000 | 0.6800 +/- 0.0000 | 0.7165 |
| 2 | vit | ce | argmax | 0% | 0.0214 +/- 0.0000 | 0.0195 +/- 0.0000 | 0.9933 +/- 0.0000 | 0.7067 +/- 0.0000 | 0.8500 +/- 0.0000 | 0.7067 +/- 0.0000 | 0.7459 |
| 3 | vit | ce | calibrated_cost_min | 0% | 0.0461 +/- 0.0000 | 0.0205 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.5900 +/- 0.0000 | 0.7950 +/- 0.0000 | 0.5900 +/- 0.0000 | 0.5924 |

### Top NICME Rows

No NICME rows found.

### Strict All-Seed Floor Rows

No aggregate row met both floors in all seeds.

### Mean-Floor-Compliant Rows

No aggregate row met both floors on mean metrics.

Best mean selection row: `vit + ce + calibrated_threshold` with selection `0.0209`, nATC `0.0160`, target recall `1.0000`, and selected accuracy `0.6800`.

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

- Full decision rows: `results/stop3b_imbalance_decoupling/partial_analysis/stop3b_partial_full_decision_rows.csv`
- Aggregate summary: `results/stop3b_imbalance_decoupling/partial_analysis/stop3b_partial_aggregate_summary.csv`
- Ranked summary: `results/stop3b_imbalance_decoupling/partial_analysis/stop3b_partial_ranked_summary.csv`
