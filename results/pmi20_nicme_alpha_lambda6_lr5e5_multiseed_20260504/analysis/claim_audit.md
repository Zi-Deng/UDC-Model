# PMI-20 NICME Six-Candidate Claim Audit

This analysis compares only six fixed NICME alpha/lambda candidates on PMI-20.
Use the separate PMI-20 SOTA table for comparisons against CE, Menon, CSADA, cost-sensitive regularized CE, and cost-weighted CE.

## Aggregate Leader

- Composite leader: `NICME PMI-10 run 95 (alpha=0.5, lambda=0.07)`

## Endpoint Winners

| Endpoint | Winner | Value |
|---|---|---:|
| target_recall_min | NICME PMI-10 run 95 (alpha=0.5, lambda=0.07) | 0.927083 |
| target_recall_macro | NICME PMI-10 run 95 (alpha=0.5, lambda=0.07) | 0.973958 |
| target_recall_macro | NICME added PMI-20 candidate (alpha=0.5, lambda=0.1) | 0.973958 |
| normalized_atc | NICME added PMI-20 candidate (alpha=0.5, lambda=0.1) | 0.003698 |
| atc | NICME added PMI-20 candidate (alpha=0.5, lambda=0.1) | 0.036979 |
| critical_pair_error_count | NICME added PMI-20 candidate (alpha=0.5, lambda=0.1) | 1.000000 |
| composite_recall_first_cost_sensitive_rank | NICME PMI-10 run 95 (alpha=0.5, lambda=0.07) | 1 |

## Guardrail

- Describe this as a fixed-candidate robustness/sensitivity rerun, not as unbiased HPO.
- Report all six settings even if the previously used run-50 setting is not the aggregate winner.
- Disclose that the candidates were selected after prior PMI-10/PMI-20 evidence.
