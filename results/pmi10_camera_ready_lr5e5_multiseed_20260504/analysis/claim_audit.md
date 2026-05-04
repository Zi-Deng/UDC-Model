# Camera-Ready Claim Audit

Primary NICME config: alpha `0.5`, lambda `0.07`, LR `5e-5`, fixed from pilot HPO before this multi-seed rerun.

## Supported Claims

- No predeclared cost-sensitive endpoint was won by NICME v3 in this aggregate.

## Claims Not Supported By This Aggregate

- NICME v3 did not win best recall-first cost-sensitive tradeoff; winner: Menon logit adjustment.
- NICME v3 did not win best normalized expected-cost performance; winner: Menon logit adjustment.
- NICME v3 did not win best expected-cost performance; winner: Menon logit adjustment.
- NICME v3 did not win best critical-confusion control; winner: Menon logit adjustment, CE + cost-min inference, SOSR-CNN.
- NICME v3 did not win best cared-class minimum recall; winner: Menon logit adjustment, LDAM-DRW.
- NICME v3 did not win best cared-class macro recall; winner: Menon logit adjustment.

## Guardrail

Do not claim universal superiority unless NICME v3 wins every endpoint in the aggregate table.
