# Binary Cost Matrix Claim Audit

Updated: 2026-05-04

Sources:

- `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_aggregate_summary.csv`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_aggregate_summary.csv`
- `data/cost_matrix_evidence/harm_scoring.csv`

## Supported Claims

1. The Spider and BreaKHis binary matrices are no longer arbitrary assumptions; they are evidence-derived through a documented public-source review and harm-index mapping.

2. The exact ratio should be described as a primary decision-context ratio, not a true universal clinical utility.

3. The plausible ratio interval `{2,5,10,20}` is empirically stress-tested, with `R=1` as a symmetric-cost negative control.

4. In this historical broad-ratio grid, BreaKHis NICME is the best composite row at `R=5` and `R=10`; it is also ranked first at `R=1` and `R=2`, although strict all-seed floors are not met at those low-ratio settings.

5. In this historical broad-ratio grid, Spider NICME is the best composite row at `R=1`, `R=2`, and `R=5`. At `R=10`, NICME remains an all-seed-floor row but is not the lowest-normalized-ATC row.

6. Ratio sensitivity changes winners, especially at `R=20`. This supports the paper’s methodological claim that cost matrices are a substantive part of cost-sensitive learning, not a decorative evaluation choice.

## Unsupported Claims

1. Do not claim that any single primary integer ratio is the empirically true cost ratio for either dataset.

2. Do not claim that NICME uniformly dominates every baseline across all plausible ratios.

3. Do not claim Spider false-widow errors are harmless. The review records case-level evidence of Steatoda envenomation, which is why the lower-cost error is normalized to `1`, not `0`.

4. Do not claim BreaKHis false positives are harmless. The review records anxiety and workup burden, but treats malignant-as-benign as substantially higher cost in the decision-support context.

5. Do not claim screening mammography evidence is identical to BreaKHis histopathology. It is used only as secondary proxy evidence where direct public cost matrices are unavailable.

## Paper-Safe Position

Use:

> We constructed primary binary matrices from public toxicology, clinical, dataset, and public-health evidence, then audited conclusions over a plausible cost-ratio interval.

Avoid:

> We discovered the true cost matrix for Spider and BreaKHis.

Best concise result framing:

> Under evidence-derived matrices, NICME remains a strong cost-sensitive model, especially on BreaKHis in the historical broad-ratio grid. The current primary integer matrices are evaluated in the 2026-05-05 camera-ready binary suite. Across the wider plausible interval, winner changes show why transparent cost-matrix specification is necessary.
