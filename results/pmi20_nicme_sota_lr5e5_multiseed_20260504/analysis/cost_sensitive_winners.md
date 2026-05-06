# Cost-Sensitive Endpoint Winners

All endpoint winners are reported; non-NICME wins are not hidden.

| Endpoint | Direction | Winner | Value | NICME? |
|---|---|---|---:|---|
| composite_recall_first_cost_sensitive_rank | lower | NICME (alpha=0.5, lambda=0.1) | 1.000000 | yes |
| target_recall_min | higher | NICME (alpha=0.5, lambda=0.1) | 0.916667 | yes |
| target_recall_min | higher | cost-sensitive regularized CE | 0.916667 | no |
| target_recall_min | higher | Menon logit adjustment | 0.916667 | no |
| target_recall_min | higher | CE | 0.916667 | no |
| target_recall_min | higher | Cost-weighted CE | 0.916667 | no |
| target_recall_macro | higher | NICME (alpha=0.5, lambda=0.1) | 0.973958 | yes |
| normalized_atc | lower | NICME (alpha=0.5, lambda=0.1) | 0.003698 | yes |
| atc | lower | NICME (alpha=0.5, lambda=0.1) | 0.036979 | yes |
| critical_pair_error_count | lower | CE + cost-min inference | 0.666667 | no |

## Interpretation

- NICME ranks first under the predeclared recall-first composite.
- NICME ties the best target-min recall and wins target-macro recall, normalized ATC, and ATC.
- CE + cost-min inference has the lowest critical-pair error count, but at substantially lower target-min recall. Among argmax trained-model rows, NICME has the lowest critical-pair error count.
