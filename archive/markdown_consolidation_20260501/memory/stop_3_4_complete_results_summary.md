# Stop 3 And Stop 4 Complete Results Memory

Generated: 2026-04-29

This is the Codex-facing handoff memory for completed Stop 3 and Stop 4 work. It complements the human report at `docs/experiment_plans/STOP_3_4_complete_results_summary.md`.

## Current Status

Stop 3 and Stop 4 are complete.

Final successful planned runs:

| stop | successful runs | failures | decision rows |
| --- | ---: | ---: | ---: |
| Stop 3A balanced primary | 72 | 0 | 216 |
| Stop 3B imbalance decoupling | 108 | 0 | 324 |
| Stop 4A backbone ablation | 36 | 0 | 108 |
| Stop 4B Spider ConvNeXt | 45 | 1 old failed attempt, retried successfully | 135 |
| Stop 4B BreaKHis DINOv3 ConvNeXt LoRA | 45 | 0 | 135 |

Total: 306 successful final planned runs and 918 decision rows across Stop 3/4.

The only failure was Stop 4B Spider run index 13, `costr2 convnext ce_calibrated_cost_min seed43`, which segfaulted with `returncode=-11` after about 1.9 seconds. It was retried successfully with `returncode=0`; final Spider Stop 4B status is 45/45 planned rows successful.

## High-Level Read

The strongest supported claim is:

> On balanced Spider and balanced BreaKHis, where class-frequency imbalance is removed, NICME-family methods achieve the cleanest strict all-seed recall/ATC tradeoffs at the original 10:1 cost setting after backbone selection.

Use this nuance:

- Spider balanced primary Stop 4A paper row: `convnext + nicme_hybrid + calibrated_threshold` at 10:1, floors all, selection `0.0183`, nATC `0.0183`, recall `0.9911`, accuracy `0.8567`.
- Stop 4B independently reran the same Spider 10:1 operating point in the sensitivity sweep and observed selection `0.0193`, nATC `0.0193`, recall `0.9933`, accuracy `0.8367`.
- BreaKHis balanced best paper row: `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold` at 10:1, floors all, selection `0.0145`, nATC `0.0145`, recall `0.9965`, accuracy `0.8712`.
- Do not claim `nicme_hybrid` is universally best.
- Do not claim NICME dominates every baseline.
- Do claim the NICME family gives strong balanced-data evidence that explicit costs matter beyond class imbalance.

## Stop 3A

Purpose: balanced primary evidence after Stop 2A-2C tuning.

Scope:

- Datasets: `spider_balanced`, `breakhis_balanced`.
- Models: `vit`, `timm_dinov3_vit_lora`.
- Methods: `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `cs_regularized_ce`, `nicme_logit_adjustment`, `nicme_hybrid`.
- Seeds: `42`, `43`, `44`.
- Modes: `argmax`, `calibrated_cost_min`, `calibrated_threshold`.

Results:

- Spider best overall: `timm_dinov3_vit_lora + nicme_logit_adjustment + calibrated_threshold`, floors 33%, selection `0.0214`, nATC `0.0211`, recall `1.0000`, accuracy `0.7889`.
- Spider best strict row: `timm_dinov3_vit_lora + nicme_hybrid + argmax`, floors all, selection `0.0272`, nATC `0.0272`, recall `0.9600`, accuracy `0.9078`.
- Spider best mean-floor NICME row: `vit + nicme_logit_adjustment + calibrated_threshold`, floors 67%, selection `0.0230`, nATC `0.0229`, recall `0.9933`, accuracy `0.8011`.
- BreaKHis best overall: `vit + ce_calibrated_cost_min + calibrated_cost_min`, floors 67%, selection `0.0173`, nATC `0.0164`, recall `0.9976`, accuracy `0.8463`.
- BreaKHis best NICME / mean-floor row: `vit + nicme_logit_adjustment + calibrated_threshold`, floors 33%, selection `0.0186`, nATC `0.0178`, recall `0.9929`, accuracy `0.8534`.
- No BreaKHis Stop 3A row met both floors in all three seeds.

Takeaway: NICME became competitive and sometimes best on balanced data, but floor stability was not yet enough for the final paper claim.

## Stop 3B

Purpose: imbalance/deployment decoupling.

Scope:

- Datasets: `spider_target_minority`, `spider_target_majority`, `breakhis_natural`.
- Models: `vit`, `timm_dinov3_vit_lora`.
- Same six-method family as Stop 3A.
- 108/108 runs passed.

Best overall rows:

- `breakhis_natural`: `timm_dinov3_vit_lora + nicme_hybrid + argmax`, floors all, selection `0.0153`, nATC `0.0153`, recall `0.9907`, selected balanced accuracy `0.8742`, plain accuracy `0.8981`.
- `spider_target_majority`: `vit + menon_logit_adjusted + calibrated_cost_min`, floors 67%, selection `0.0156`, nATC `0.0155`, recall `0.9956`, selected balanced accuracy `0.7544`.
- `spider_target_minority`: `vit + menon_logit_adjusted + calibrated_threshold`, floors all, selection `0.0152`, nATC `0.0152`, recall `0.9867`, selected balanced accuracy `0.9144`.

Best NICME rows:

- `breakhis_natural`: same as best overall, `timm_dinov3_vit_lora + nicme_hybrid + argmax`.
- `spider_target_majority`: `vit + nicme_logit_adjustment + calibrated_threshold`, floors 67%, selection `0.0170`, nATC `0.0162`, recall `0.9933`, selected balanced accuracy `0.7733`.
- `spider_target_minority`: `vit + nicme_logit_adjustment + calibrated_threshold`, floors all, selection `0.0220`, nATC `0.0220`, recall `0.9867`, selected balanced accuracy `0.8689`.

Takeaway: natural BreaKHis favors NICME hybrid; controlled imbalanced Spider often favors Menon. This is not a contradiction because Menon is an imbalance/class-prior baseline.

## Stop 4A

Purpose: balanced-data backbone ablation to resolve Stop 3A floor instability.

Status: 36/36 runs passed.

Main results:

- `spider_balanced`: `convnext + nicme_hybrid + calibrated_threshold`, floors all, selection `0.0183`, nATC `0.0183`, recall `0.9911`, accuracy `0.8567`.
- `breakhis_balanced`: `timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold`, floors all, selection `0.0145`, nATC `0.0145`, recall `0.9965`, accuracy `0.8712`.

Best CE-family comparisons:

- `spider_balanced`: `convnext + ce_calibrated_cost_min + calibrated_cost_min`, floors all, selection `0.0184`, nATC `0.0184`, recall `0.9956`, accuracy `0.8356`.
- `breakhis_balanced`: `timm_dinov3_convnext_lora + ce_calibrated_cost_min + calibrated_cost_min`, floors 33%, selection `0.0180`, nATC `0.0170`, recall `0.9976`, accuracy `0.8410`.

Takeaway: Stop 4A is the hinge result. It turns the balanced evidence into strict all-seed NICME wins for both applications.

## Stop 4B

Purpose: cost-ratio sensitivity on Stop 4A application-specific backbones.

Scope:

- Spider: `spider_balanced`, `convnext`.
- BreaKHis: `breakhis_balanced`, `timm_dinov3_convnext_lora`.
- Methods: `ce_calibrated_cost_min`, `nicme_logit_adjustment`, `nicme_hybrid`.
- Ratios: `1`, `2`, `5`, `10`, `20`.
- Seeds: `42`, `43`, `44`.
- Total planned runs: 90. All final planned runs succeeded.

Spider ratio read:

- Ratio 1 best: CE argmax, selection `0.0587`, but floors 0%.
- Ratio 2 best: NICME logit adjustment calibrated threshold, selection `0.0429`, floors 67%.
- Ratio 5 best: NICME logit adjustment calibrated cost-min, selection `0.0260`, floors all.
- Ratio 10 best: CE calibrated cost-min calibrated cost-min, selection `0.0184`, floors all. Best NICME is hybrid calibrated threshold, selection `0.0193`, floors all.
- Ratio 20 best mean row: NICME hybrid calibrated threshold, selection `0.0119`, floors 67%, accuracy `0.7911`. Best strict row is CE calibrated threshold, selection `0.0124`, floors all.

BreaKHis ratio read:

- Ratio 1 best: CE calibrated threshold, selection `0.1990`, floors 33%.
- Ratio 2 best: NICME hybrid argmax, selection `0.0400`, floors 33%.
- Ratio 5 best: NICME logit adjustment argmax, selection `0.0274`, floors 33%; strict row is NICME logit adjustment calibrated cost-min, selection `0.0288`.
- Ratio 10 best: NICME logit adjustment calibrated threshold, selection `0.0145`, floors all.
- Ratio 20 best mean row: CE calibrated threshold, selection `0.0114`, floors 67%, accuracy `0.8434`; no strict all-seed row.

Takeaway: cost sensitivity behaves sensibly, but high ratios can weaken accuracy-floor stability. For BreaKHis, 10:1 is clearly the robust NICME point. For Spider, 10:1 is a close CE-vs-NICME comparison, while 20:1 gives lower mean selection for NICME hybrid but loses strict floor stability.

## Interpretation Rules For Future Codex

- Treat Stop 4B as complete. Do not resume it unless the user explicitly asks for reruns or extra ratios.
- If asked about the old Stop 4B segfault, say it was run index 13 in Spider, returncode -11, and retry succeeded.
- Use balanced Stop 4A/4B 10:1 rows as the cleanest paper evidence.
- Mention Menon wins on imbalanced Spider honestly.
- Mention CE calibrated cost-min as a strong baseline.
- Prefer `calibrated_threshold` as the main binary deployment mode in discussion.

## Source Artifacts

Human synthesis:

- `docs/experiment_plans/STOP_3_4_complete_results_summary.md`

Prior reports:

- `docs/experiment_plans/STOP_3A_balanced_primary_scientific_read.md`
- `docs/experiment_plans/STOP_3B_4A_results_and_stop4B_plan.md`
- `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_scientific_read.md`
- `results/stop4a_backbone_ablation/stop4a_backbone_ablation_scientific_read.md`
- `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_scientific_read.md`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_scientific_read.md`

Core CSVs:

- `results/stop3a_balanced_primary/stop3a_aggregate_summary.csv`
- `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_aggregate_summary.csv`
- `results/stop4a_backbone_ablation/stop4a_backbone_ablation_aggregate_summary.csv`
- `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_aggregate_summary.csv`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_aggregate_summary.csv`
