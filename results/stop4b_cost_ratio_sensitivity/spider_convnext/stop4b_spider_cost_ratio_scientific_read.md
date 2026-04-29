# stop4b_spider_cost_ratio Results And Scientific Read

Generated from local run logs and metrics files.

## Executive Summary

- Decision rows: `135`.
- Aggregate rows: `45`.
- Lower `selection_score` is better; floor satisfaction uses the configured `selection_accuracy_metric` for each dataset.
- Interpret natural/imbalanced results as deployment realism and decoupling stress tests, not replacements for balanced-dataset evidence.

## spider_balanced

Target: `black_widow`. Floors: recall `0.95`, accuracy `0.8`.

### Top Overall Rows

| rank | model | method | mode | cost ratio | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | nicme_hybrid | calibrated_threshold | 20.0000 | 67% | 0.0119 +/- 0.0020 | 0.0115 +/- 0.0017 | 0.9978 +/- 0.0038 | 0.7911 +/- 0.0241 | 0.7911 +/- 0.0241 | 0.7911 +/- 0.0241 | 0.7815 |
| 2 | convnext | ce_calibrated_cost_min | calibrated_threshold | 20.0000 | all | 0.0124 +/- 0.0037 | 0.0124 +/- 0.0037 | 0.9889 +/- 0.0102 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8539 |
| 3 | convnext | nicme_logit_adjustment | calibrated_threshold | 20.0000 | 67% | 0.0124 +/- 0.0013 | 0.0124 +/- 0.0013 | 0.9933 +/- 0.0000 | 0.8144 +/- 0.0252 | 0.8144 +/- 0.0252 | 0.8144 +/- 0.0252 | 0.8081 |
| 4 | convnext | ce_calibrated_cost_min | calibrated_cost_min | 20.0000 | 33% | 0.0147 +/- 0.0057 | 0.0124 +/- 0.0019 | 0.9978 +/- 0.0038 | 0.7733 +/- 0.0498 | 0.7733 +/- 0.0498 | 0.7733 +/- 0.0498 | 0.7598 |
| 5 | convnext | ce_calibrated_cost_min | calibrated_cost_min | 10.0000 | all | 0.0184 +/- 0.0019 | 0.0184 +/- 0.0019 | 0.9956 +/- 0.0077 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8307 |
| 6 | convnext | ce_calibrated_cost_min | calibrated_threshold | 10.0000 | all | 0.0193 +/- 0.0023 | 0.0193 +/- 0.0023 | 0.9889 +/- 0.0102 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8539 |
| 7 | convnext | nicme_hybrid | calibrated_threshold | 10.0000 | all | 0.0193 +/- 0.0020 | 0.0193 +/- 0.0020 | 0.9933 +/- 0.0000 | 0.8367 +/- 0.0200 | 0.8367 +/- 0.0200 | 0.8367 +/- 0.0200 | 0.8324 |
| 8 | convnext | nicme_logit_adjustment | calibrated_cost_min | 20.0000 | 0% | 0.0197 +/- 0.0146 | 0.0129 +/- 0.0035 | 1.0000 +/- 0.0000 | 0.7411 +/- 0.0707 | 0.7411 +/- 0.0707 | 0.7411 +/- 0.0707 | 0.7190 |
| 9 | convnext | nicme_logit_adjustment | calibrated_threshold | 10.0000 | 67% | 0.0198 +/- 0.0013 | 0.0198 +/- 0.0013 | 0.9933 +/- 0.0067 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8270 |
| 10 | convnext | ce_calibrated_cost_min | argmax | 20.0000 | all | 0.0221 +/- 0.0028 | 0.0221 +/- 0.0028 | 0.9622 +/- 0.0077 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9176 |

### Top NICME Rows

| rank | model | method | mode | cost ratio | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | nicme_hybrid | calibrated_threshold | 20.0000 | 67% | 0.0119 +/- 0.0020 | 0.0115 +/- 0.0017 | 0.9978 +/- 0.0038 | 0.7911 +/- 0.0241 | 0.7911 +/- 0.0241 | 0.7911 +/- 0.0241 | 0.7815 |
| 2 | convnext | nicme_logit_adjustment | calibrated_threshold | 20.0000 | 67% | 0.0124 +/- 0.0013 | 0.0124 +/- 0.0013 | 0.9933 +/- 0.0000 | 0.8144 +/- 0.0252 | 0.8144 +/- 0.0252 | 0.8144 +/- 0.0252 | 0.8081 |
| 3 | convnext | nicme_hybrid | calibrated_threshold | 10.0000 | all | 0.0193 +/- 0.0020 | 0.0193 +/- 0.0020 | 0.9933 +/- 0.0000 | 0.8367 +/- 0.0200 | 0.8367 +/- 0.0200 | 0.8367 +/- 0.0200 | 0.8324 |
| 4 | convnext | nicme_logit_adjustment | calibrated_cost_min | 20.0000 | 0% | 0.0197 +/- 0.0146 | 0.0129 +/- 0.0035 | 1.0000 +/- 0.0000 | 0.7411 +/- 0.0707 | 0.7411 +/- 0.0707 | 0.7411 +/- 0.0707 | 0.7190 |
| 5 | convnext | nicme_logit_adjustment | calibrated_threshold | 10.0000 | 67% | 0.0198 +/- 0.0013 | 0.0198 +/- 0.0013 | 0.9933 +/- 0.0067 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8270 |
| 6 | convnext | nicme_logit_adjustment | calibrated_cost_min | 10.0000 | 33% | 0.0226 +/- 0.0026 | 0.0222 +/- 0.0022 | 0.9956 +/- 0.0038 | 0.7978 +/- 0.0336 | 0.7978 +/- 0.0336 | 0.7978 +/- 0.0336 | 0.7890 |
| 7 | convnext | nicme_hybrid | argmax | 20.0000 | 67% | 0.0227 +/- 0.0109 | 0.0227 +/- 0.0109 | 0.9622 +/- 0.0214 | 0.9044 +/- 0.0150 | 0.9044 +/- 0.0150 | 0.9044 +/- 0.0150 | 0.9041 |
| 8 | convnext | nicme_hybrid | calibrated_cost_min | 10.0000 | 0% | 0.0247 +/- 0.0047 | 0.0234 +/- 0.0029 | 0.9978 +/- 0.0038 | 0.7756 +/- 0.0310 | 0.7756 +/- 0.0310 | 0.7756 +/- 0.0310 | 0.7633 |
| 9 | convnext | nicme_logit_adjustment | argmax | 20.0000 | all | 0.0248 +/- 0.0035 | 0.0248 +/- 0.0035 | 0.9578 +/- 0.0077 | 0.9056 +/- 0.0126 | 0.9056 +/- 0.0126 | 0.9056 +/- 0.0126 | 0.9053 |
| 10 | convnext | nicme_logit_adjustment | calibrated_cost_min | 5.0000 | all | 0.0260 +/- 0.0007 | 0.0260 +/- 0.0007 | 0.9911 +/- 0.0038 | 0.8878 +/- 0.0084 | 0.8878 +/- 0.0084 | 0.8878 +/- 0.0084 | 0.8865 |

### Strict All-Seed Floor Rows

| rank | model | method | mode | cost ratio | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | ce_calibrated_cost_min | calibrated_threshold | 20.0000 | all | 0.0124 +/- 0.0037 | 0.0124 +/- 0.0037 | 0.9889 +/- 0.0102 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8539 |
| 2 | convnext | ce_calibrated_cost_min | calibrated_cost_min | 10.0000 | all | 0.0184 +/- 0.0019 | 0.0184 +/- 0.0019 | 0.9956 +/- 0.0077 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8307 |
| 3 | convnext | ce_calibrated_cost_min | calibrated_threshold | 10.0000 | all | 0.0193 +/- 0.0023 | 0.0193 +/- 0.0023 | 0.9889 +/- 0.0102 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8539 |
| 4 | convnext | nicme_hybrid | calibrated_threshold | 10.0000 | all | 0.0193 +/- 0.0020 | 0.0193 +/- 0.0020 | 0.9933 +/- 0.0000 | 0.8367 +/- 0.0200 | 0.8367 +/- 0.0200 | 0.8367 +/- 0.0200 | 0.8324 |
| 5 | convnext | ce_calibrated_cost_min | argmax | 20.0000 | all | 0.0221 +/- 0.0028 | 0.0221 +/- 0.0028 | 0.9622 +/- 0.0077 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9176 |
| 6 | convnext | nicme_logit_adjustment | argmax | 20.0000 | all | 0.0248 +/- 0.0035 | 0.0248 +/- 0.0035 | 0.9578 +/- 0.0077 | 0.9056 +/- 0.0126 | 0.9056 +/- 0.0126 | 0.9056 +/- 0.0126 | 0.9053 |
| 7 | convnext | ce_calibrated_cost_min | argmax | 10.0000 | all | 0.0252 +/- 0.0019 | 0.0252 +/- 0.0019 | 0.9622 +/- 0.0077 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9176 |
| 8 | convnext | nicme_logit_adjustment | calibrated_cost_min | 5.0000 | all | 0.0260 +/- 0.0007 | 0.0260 +/- 0.0007 | 0.9911 +/- 0.0038 | 0.8878 +/- 0.0084 | 0.8878 +/- 0.0084 | 0.8878 +/- 0.0084 | 0.8865 |
| 9 | convnext | nicme_hybrid | argmax | 10.0000 | all | 0.0262 +/- 0.0050 | 0.0262 +/- 0.0050 | 0.9622 +/- 0.0102 | 0.9078 +/- 0.0051 | 0.9078 +/- 0.0051 | 0.9078 +/- 0.0051 | 0.9075 |
| 10 | convnext | ce_calibrated_cost_min | calibrated_threshold | 5.0000 | all | 0.0267 +/- 0.0050 | 0.0267 +/- 0.0050 | 0.9889 +/- 0.0102 | 0.8889 +/- 0.0084 | 0.8889 +/- 0.0084 | 0.8889 +/- 0.0084 | 0.8878 |

### Mean-Floor-Compliant Rows

| rank | model | method | mode | cost ratio | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | convnext | ce_calibrated_cost_min | calibrated_threshold | 20.0000 | all | 0.0124 +/- 0.0037 | 0.0124 +/- 0.0037 | 0.9889 +/- 0.0102 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8539 |
| 2 | convnext | nicme_logit_adjustment | calibrated_threshold | 20.0000 | 67% | 0.0124 +/- 0.0013 | 0.0124 +/- 0.0013 | 0.9933 +/- 0.0000 | 0.8144 +/- 0.0252 | 0.8144 +/- 0.0252 | 0.8144 +/- 0.0252 | 0.8081 |
| 3 | convnext | ce_calibrated_cost_min | calibrated_cost_min | 10.0000 | all | 0.0184 +/- 0.0019 | 0.0184 +/- 0.0019 | 0.9956 +/- 0.0077 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8356 +/- 0.0350 | 0.8307 |
| 4 | convnext | ce_calibrated_cost_min | calibrated_threshold | 10.0000 | all | 0.0193 +/- 0.0023 | 0.0193 +/- 0.0023 | 0.9889 +/- 0.0102 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8567 +/- 0.0233 | 0.8539 |
| 5 | convnext | nicme_hybrid | calibrated_threshold | 10.0000 | all | 0.0193 +/- 0.0020 | 0.0193 +/- 0.0020 | 0.9933 +/- 0.0000 | 0.8367 +/- 0.0200 | 0.8367 +/- 0.0200 | 0.8367 +/- 0.0200 | 0.8324 |
| 6 | convnext | nicme_logit_adjustment | calibrated_threshold | 10.0000 | 67% | 0.0198 +/- 0.0013 | 0.0198 +/- 0.0013 | 0.9933 +/- 0.0067 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8322 +/- 0.0407 | 0.8270 |
| 7 | convnext | ce_calibrated_cost_min | argmax | 20.0000 | all | 0.0221 +/- 0.0028 | 0.0221 +/- 0.0028 | 0.9622 +/- 0.0077 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9176 |
| 8 | convnext | nicme_hybrid | argmax | 20.0000 | 67% | 0.0227 +/- 0.0109 | 0.0227 +/- 0.0109 | 0.9622 +/- 0.0214 | 0.9044 +/- 0.0150 | 0.9044 +/- 0.0150 | 0.9044 +/- 0.0150 | 0.9041 |
| 9 | convnext | nicme_logit_adjustment | argmax | 20.0000 | all | 0.0248 +/- 0.0035 | 0.0248 +/- 0.0035 | 0.9578 +/- 0.0077 | 0.9056 +/- 0.0126 | 0.9056 +/- 0.0126 | 0.9056 +/- 0.0126 | 0.9053 |
| 10 | convnext | ce_calibrated_cost_min | argmax | 10.0000 | all | 0.0252 +/- 0.0019 | 0.0252 +/- 0.0019 | 0.9622 +/- 0.0077 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9178 +/- 0.0171 | 0.9176 |

Best mean selection row: `convnext + nicme_hybrid + calibrated_threshold` with selection `0.0119`, nATC `0.0115`, target recall `0.9978`, and selected accuracy `0.7911`.

Best NICME row: `convnext + nicme_hybrid + calibrated_threshold` with selection `0.0119`, nATC `0.0115`, target recall `0.9978`, and selected accuracy `0.7911`.

## Artifacts

- Full decision rows: `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_full_decision_rows.csv`
- Aggregate summary: `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_aggregate_summary.csv`
- Ranked summary: `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_ranked_summary.csv`
