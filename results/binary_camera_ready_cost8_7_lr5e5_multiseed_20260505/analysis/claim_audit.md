# Binary Camera-Ready Claim Audit

Primary NICME config: alpha `0.5`, lambda `0.1`, LR `5e-5`.

## Supported Claims By Dataset

### Spider
- NICME did not win a predeclared cost-sensitive endpoint in this aggregate.
- Non-NICME endpoint wins are preserved in `cost_sensitive_winners.md`.

### BreaKHis
- NICME won or tied `normalized_atc`.
- NICME won or tied `atc`.
- Non-NICME endpoint wins are preserved in `cost_sensitive_winners.md`.

## Guardrails

- Do not average Spider and BreaKHis into one headline rank.
- Do not claim external global SOTA.
- Error bars are sample standard deviations over training seeds on one fixed split per dataset.
