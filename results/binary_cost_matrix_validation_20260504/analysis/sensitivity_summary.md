# Binary Cost-Ratio Sensitivity Summary

Generated from completed Stop 4B aggregate CSVs. Ranking uses all-seed recall/accuracy floor satisfaction first, then normalized ATC ascending, target recall descending, and accuracy descending.

| Dataset | R | Best row | Best norm. ATC | Best target recall | Best floor pass | Best NICME row | NICME rank | NICME norm. ATC | NICME target recall |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| spider | 1 | `nicme_logit_adjustment / calibrated_threshold` | 0.0589 | 0.9578 | 1.00 | `nicme_logit_adjustment / calibrated_threshold` | 1 | 0.0589 | 0.9578 |
| spider | 2 | `nicme_hybrid / calibrated_threshold` | 0.0439 | 0.9822 | 1.00 | `nicme_hybrid / calibrated_threshold` | 1 | 0.0439 | 0.9822 |
| spider | 5 | `nicme_logit_adjustment / calibrated_cost_min` | 0.0260 | 0.9911 | 1.00 | `nicme_logit_adjustment / calibrated_cost_min` | 1 | 0.0260 | 0.9911 |
| spider | 10 | `ce_calibrated_cost_min / calibrated_cost_min` | 0.0184 | 0.9956 | 1.00 | `nicme_hybrid / calibrated_threshold` | 3 | 0.0193 | 0.9933 |
| spider | 20 | `ce_calibrated_cost_min / calibrated_threshold` | 0.0124 | 0.9889 | 1.00 | `nicme_logit_adjustment / argmax` | 3 | 0.0248 | 0.9578 |
| breakhis | 1 | `nicme_hybrid / calibrated_threshold` | 0.1022 | 0.8617 | 0.33 | `nicme_hybrid / calibrated_threshold` | 1 | 0.1022 | 0.8617 |
| breakhis | 2 | `nicme_hybrid / argmax` | 0.0393 | 0.9669 | 0.33 | `nicme_hybrid / argmax` | 1 | 0.0393 | 0.9669 |
| breakhis | 5 | `nicme_logit_adjustment / calibrated_cost_min` | 0.0288 | 0.9846 | 1.00 | `nicme_logit_adjustment / calibrated_cost_min` | 1 | 0.0288 | 0.9846 |
| breakhis | 10 | `nicme_logit_adjustment / calibrated_threshold` | 0.0145 | 0.9965 | 1.00 | `nicme_logit_adjustment / calibrated_threshold` | 1 | 0.0145 | 0.9965 |
| breakhis | 20 | `ce_calibrated_cost_min / calibrated_cost_min` | 0.0105 | 0.9976 | 0.33 | `nicme_logit_adjustment / calibrated_cost_min` | 3 | 0.0142 | 0.9976 |

This is the broad Stop 4B sensitivity grid. The current primary integer ratios are Spider `R=8` and BreaKHis `R=7`, evaluated in `results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505/`. `R=1` remains a symmetric-cost negative control.
