# PMI-20 Camera-Ready Claim Audit

Primary NICME config: alpha `0.09`, lambda `0.07`, LR `5e-5`, fixed from PMI-10 HPO run 50 before this run.

## Supported Claims

- NICME v3 tied for best cared-class minimum recall.

## Claims Not Supported By This Aggregate

- NICME v3 did not win best recall-first cost-sensitive tradeoff; winner: cost-sensitive regularized CE.
- NICME v3 did not win best normalized expected-cost performance; winner: cost-sensitive regularized CE.
- NICME v3 did not win best expected-cost performance; winner: cost-sensitive regularized CE.
- NICME v3 did not win best critical-confusion control; winner: CE + cost-min inference.
- NICME v3 did not win best cared-class macro recall; winner: Cost-weighted CE, CSADA.

## Guardrail

This is a repository SOTA/baseline comparison for selected PMI-20 methods, not an external global SOTA claim.
