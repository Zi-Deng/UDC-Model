# PMI-20 Paper Claim Audit

Primary NICME config: alpha `0.5`, lambda `0.1`, LR `5e-5`, selected as the paper-facing PMI-20 NICME row after the completed fixed six-candidate rerun.

## Supported Claims

- NICME supports the claim: best recall-first cost-sensitive tradeoff.
- NICME supports the claim: tied-best cared-class minimum recall.
- NICME supports the claim: best cared-class macro recall.
- NICME supports the claim: best normalized expected-cost performance.
- NICME supports the claim: best expected-cost performance.

## Claims Requiring Caveats

- Critical-confusion control: NICME has the lowest critical-pair error count among argmax trained-model rows, but CE + cost-min inference has the lowest count overall while sacrificing target-min recall.
- Balanced accuracy and macro-F1: cost-sensitive regularized CE remains slightly higher than NICME on both metrics in this aggregate.

## Unsupported Claims

- Do not claim universal superiority across all metrics.
- Do not describe this table as an external global SOTA claim; it is the repository SOTA/baseline comparison under this fixed PMI-20 protocol.
- Error bars reflect variation over three training seeds on one fixed split, not dataset resampling uncertainty.
