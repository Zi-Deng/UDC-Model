# NICME Top-5 Claim Audit

This analysis compares only five NICME alpha/lambda candidates selected from pilot HPO.
It should not be presented as evidence that NICME beats external baselines; use the separate camera-ready baseline table for that.

## Aggregate Leader

- Composite leader: `NICME run 53 (alpha=0.09, lambda=0.2)`

## Endpoint Winners

| Endpoint | Winner | Value |
|---|---|---:|
| target_recall_min | NICME run 53 (alpha=0.09, lambda=0.2) | 0.937500 |
| target_recall_macro | NICME run 20 (alpha=0.06, lambda=0.03) | 0.973958 |
| normalized_atc | NICME run 50 (alpha=0.09, lambda=0.07) | 0.007083 |
| atc | NICME run 50 (alpha=0.09, lambda=0.07) | 0.070833 |
| critical_pair_error_count | NICME run 20 (alpha=0.06, lambda=0.03) | 1.333333 |
| composite_recall_first_cost_sensitive_rank | NICME run 53 (alpha=0.09, lambda=0.2) | 1 |

## Guardrail

- Describe this as a robustness confirmation of pilot-selected candidates, not as fresh HPO.
- Report all five settings even if run 95 is not the aggregate winner.
