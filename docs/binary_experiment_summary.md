# Binary Experiment Summary

Updated: 2026-05-04

The binary-first NICME extension tested cost-sensitive learning on Spider and BreaKHis classification. The final completed sequence is Stop 3A, Stop 3B, Stop 4A, and Stop 4B.

## Operational Result

| Stop | Purpose | Successful runs | Notes |
|---|---|---:|---|
| Stop 3A | Balanced primary evidence | 72 | No failures |
| Stop 3B | Imbalance and deployment decoupling | 108 | No failures |
| Stop 4A | Balanced backbone ablation | 36 | No failures |
| Stop 4B Spider | Cost-ratio sensitivity | 45 | One old segfault retried successfully |
| Stop 4B BreaKHis | Cost-ratio sensitivity | 45 | No failures |

Total: 306 successful final planned runs and 918 exported decision rows.

## Main Scientific Read

The strongest historical Stop 3/4 binary claim is that, on balanced Spider and balanced BreaKHis, where class-frequency imbalance is removed, NICME-family methods produced strong strict all-seed recall and average-test-cost tradeoffs across the broad cost-ratio grid.

The current primary integer matrices are Spider `[[0,8],[1,0]]` and BreaKHis `[[0,1],[7,0]]`. They are documented as public-evidence-derived decision-context matrices, not arbitrary assumptions and not uniquely true utilities. See [binary_cost_matrix_review_protocol.md](binary_cost_matrix_review_protocol.md), [binary_cost_matrix_justification.md](binary_cost_matrix_justification.md), and [results/binary_cost_matrix_validation_20260504](../results/binary_cost_matrix_validation_20260504/).

Historical balanced 10:1 Stop 3/4 rows:

| Dataset | Row | Selection | Normalized ATC | Target recall | Accuracy |
|---|---|---:|---:|---:|---:|
| `spider_balanced` | `convnext + nicme_hybrid + calibrated_threshold` | 0.0183 | 0.0183 | 0.9911 | 0.8567 |
| `breakhis_balanced` | `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` | 0.0145 | 0.0145 | 0.9965 | 0.8712 |

Important caveats:

- Do not claim `nicme_hybrid` is universally best.
- Do not claim NICME dominates every baseline.
- Do not claim the primary integer ratios are exact empirical truth; report the broad interval `{2,5,10,20}` and symmetric-control `R=1`.
- Menon-style logit adjustment remains a strong class-prior baseline on controlled imbalanced Spider.
- CE calibrated cost-min is a strong baseline and should stay in comparisons.
- Stop 4B sensitivity shows winner flips at some ratios; use that honestly as evidence that cost-matrix specification matters.

## Archived Source Material

The full historical reports are preserved under:

- [archive/markdown_consolidation_20260501/docs/experiment_plans/STOP_3_4_complete_results_summary.md](../archive/markdown_consolidation_20260501/docs/experiment_plans/STOP_3_4_complete_results_summary.md)
- [archive/markdown_consolidation_20260501/memory/stop_3_4_complete_results_summary.md](../archive/markdown_consolidation_20260501/memory/stop_3_4_complete_results_summary.md)
- [archive/markdown_consolidation_20260501/results/stop4b_cost_ratio_sensitivity/](../archive/markdown_consolidation_20260501/results/stop4b_cost_ratio_sensitivity/)

The underlying CSV and JSON artifacts remain in `results/`.
