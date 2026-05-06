# Binary Cost Matrix Validation

Updated: 2026-05-04

This package records the broad Stop 4B ratio-sensitivity grid for the evidence-derived Spider and BreaKHis binary cost matrices defined in:

- [docs/binary_cost_matrix_review_protocol.md](../../docs/binary_cost_matrix_review_protocol.md)
- [docs/binary_cost_matrix_justification.md](../../docs/binary_cost_matrix_justification.md)
- [data/cost_matrix_evidence](../../data/cost_matrix_evidence/)

No new GPU training was required. The required sensitivity ratios were already completed by Stop 4B:

- Spider source: `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_aggregate_summary.csv`
- BreaKHis source: `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_aggregate_summary.csv`

## Current Primary Integer Matrices

Spider, `C[true][pred]`, class 0 `black_widow`, class 1 `false_widow`:

```text
[[0, 8],
 [1, 0]]
```

BreaKHis, `C[true][pred]`, class 0 `benign`, class 1 `malignant`:

```text
[[0, 1],
 [7, 0]]
```

The current primary ratios are Spider `R=8` and BreaKHis `R=7`; this package covers broad sensitivity ratios `R={2,5,10,20}` plus `R=1` as a symmetric-cost negative control. The exact primary integer matrices are evaluated in `results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505/`.

## Files

| File | Purpose |
|---|---|
| `analysis/sensitivity_summary.csv` | One row per dataset and ratio with best composite row and best NICME row. |
| `analysis/sensitivity_summary.md` | Reader-friendly summary table. |
| `analysis/per_ratio_ranked_metrics.csv` | Full ranked Stop 4B aggregate rows by dataset and ratio. |
| `analysis/claim_audit.md` | Supported and unsupported binary matrix claims. |
| `plots/nicme_normalized_atc_by_ratio.svg` | Quick visual check of best NICME normalized ATC over ratios. |
| `plots/nicme_target_recall_by_ratio.svg` | Quick visual check of best NICME target recall over ratios. |

## Interpretation

The evidence supports asymmetric primary matrices, but not an exact universally true ratio. The validation therefore reports whether conclusions hold across the plausible interval.

The full interval shows ratio-sensitive baseline flips. This should be reported honestly: it strengthens the paper’s argument that well-specified cost matrices matter.
