# Binary Experiment Summary

Updated: 2026-05-01

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

The strongest supported binary claim is that, on balanced Spider and balanced BreaKHis, where class-frequency imbalance is removed, NICME-family methods produced the cleanest strict all-seed recall and average-test-cost tradeoffs at the original 10:1 cost setting after backbone selection.

Primary balanced 10:1 rows:

| Dataset | Row | Selection | Normalized ATC | Target recall | Accuracy |
|---|---|---:|---:|---:|---:|
| `spider_balanced` | `convnext + nicme_hybrid + calibrated_threshold` | 0.0183 | 0.0183 | 0.9911 | 0.8567 |
| `breakhis_balanced` | `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` | 0.0145 | 0.0145 | 0.9965 | 0.8712 |

Important caveats:

- Do not claim `nicme_hybrid` is universally best.
- Do not claim NICME dominates every baseline.
- Menon-style logit adjustment remains a strong class-prior baseline on controlled imbalanced Spider.
- CE calibrated cost-min is a strong baseline and should stay in comparisons.

## Archived Source Material

The full historical reports are preserved under:

- [archive/markdown_consolidation_20260501/docs/experiment_plans/STOP_3_4_complete_results_summary.md](../archive/markdown_consolidation_20260501/docs/experiment_plans/STOP_3_4_complete_results_summary.md)
- [archive/markdown_consolidation_20260501/memory/stop_3_4_complete_results_summary.md](../archive/markdown_consolidation_20260501/memory/stop_3_4_complete_results_summary.md)
- [archive/markdown_consolidation_20260501/results/stop4b_cost_ratio_sensitivity/](../archive/markdown_consolidation_20260501/results/stop4b_cost_ratio_sensitivity/)

The underlying CSV and JSON artifacts remain in `results/`.
