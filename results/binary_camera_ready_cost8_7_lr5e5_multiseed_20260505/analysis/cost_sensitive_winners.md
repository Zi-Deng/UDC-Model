# Binary Cost-Sensitive Endpoint Winners

| Dataset | Endpoint | Direction | Winner | Value | NICME? |
|---|---|---|---|---:|---|
| BreaKHis | target_recall_min | higher | CSADA | 1.000000 | no |
| BreaKHis | target_recall_macro | higher | CSADA | 1.000000 | no |
| BreaKHis | normalized_atc | lower | NICME (alpha=0.5, lambda=0.1) | 0.032590 | yes |
| BreaKHis | atc | lower | NICME (alpha=0.5, lambda=0.1) | 0.228132 | yes |
| BreaKHis | critical_pair_error_count | lower | CSADA | 0.000000 | no |
| BreaKHis | composite_recall_first_cost_sensitive_rank | rank | CSADA | 1 | no |
| Spider | target_recall_min | higher | CSADA | 1.000000 | no |
| Spider | target_recall_min | higher | Cost-weighted CE | 1.000000 | no |
| Spider | target_recall_macro | higher | CSADA | 1.000000 | no |
| Spider | target_recall_macro | higher | Cost-weighted CE | 1.000000 | no |
| Spider | normalized_atc | lower | CE + cost-min inference | 0.013333 | no |
| Spider | atc | lower | CE + cost-min inference | 0.106667 | no |
| Spider | critical_pair_error_count | lower | CSADA | 0.000000 | no |
| Spider | critical_pair_error_count | lower | Cost-weighted CE | 0.000000 | no |
| Spider | composite_recall_first_cost_sensitive_rank | rank | CSADA | 1 | no |
