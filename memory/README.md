# NICME Memory Index

This folder is a Codex/GPT memory pack for this repository, similar in purpose to a `CLAUDE.md` onboarding file but split into focused documents.

Start here:

- `repository_overview.md` - what this repo is, its layout, stack, and source-of-truth files.
- `commands_and_environment.md` - environment setup, common commands, run directories, and verification notes.
- `training_pipeline.md` - parent repository training flow, datasets, models, losses, metrics, and outputs.
- `sweeps_hpo_and_playground.md` - cost-matrix sweeps, HPO, comparison scripts, and the nested playground project.
- `data_configs_and_artifacts.md` - local data, config files, weights, results, checkpoints, and generated artifacts.
- `maintenance_notes.md` - coding conventions, known pitfalls, stale docs, and safety notes for future agents.
- `nicme_binary_extension_2026.md` - Research Proposal extension memory: literature gate, binary data plan, cost convention, BreaKHis prep, metrics, baselines, model backends, experiment tiers, and verification status.
- `huggingface_dinov3_access.md` - Hugging Face DINOv3 authentication notes, gated-model access steps, intended DINOv3 model sizes, and DINOv3-specific LoRA target modules.
- `experiment_plan_stop_gated_master.md` - mandatory stop-gated experiment protocol; each completed stop must create `memory/experiment_plan_stop_N.md` and `docs/experiment_plans/STOP_N_results_and_next_plan.md` before any later stop begins.
- `experiment_plan_stop_0.md` - completed Stop 0 data audit: spider and BreaKHis prepared splits, label/prevalence checks, patient-overlap checks, BreaKHis magnification/tumor-type distributions, data-prep bug fix, and Stop 1 checkpoint.
- `experiment_plan_stop_1.md` - completed final revised Stop 1 smoke-run checkpoint: 24/24 accessible-DINOv3 rows passed using ConvNeXt, ViT, timm DINOv3 ViT/ConvNeXt full fine-tuning, and timm DINOv3 ViT/ConvNeXt LoRA on balanced spider and BreaKHis subsets; official Meta `facebook/dinov3-*` rows remain queued for later rerun after Hugging Face gated access is approved.
- `experiment_plan_stop_2.md` - completed Stop 2 prototype checkpoint: 32/32 rows passed in 16.37 minutes using ConvNeXt, ViT, timm DINOv3 ViT LoRA, and timm DINOv3 ConvNeXt MLP-LoRA on balanced spider/BreaKHis prototype splits; results are operationally successful but scientifically mixed, so Stop 3 should scale carefully and keep CE argmax plus CE calibrated-cost-min as strong baselines.
- `stop_2_calibration_adjustment_analysis.md` - active pause artifact after Stop 2: explains `argmax` vs `calibrated_cost_min`, records every Stop 2 result row with the associated cost matrix, and frames the next calibration/model-adjustment goal as improving NICME target recall and normalized ATC without accuracy-floor collapse.
- `stop_2a_2c_nicme_tuning_results.md` - Stop 2A-2C tuning record: added `nicme_logit_cost_scale` and `calibrated_threshold`, ran 119 successful tuning/rerun rows with checkpoint cleanup, found tuned NICME best on BreaKHis but not Spider, where CE + calibrated threshold remains the best prototype row.
- `experiment_plan_stop_3_updated.md` - active Stop 3 execution memory: records storage cleanup, adjusted tuned NICME settings, the Stop 3A balanced primary queue, and the rule to pause before imbalance runs if NICME still loses balanced Spider to CE + calibrated threshold.
- `stop_3a_balanced_primary_scientific_read.md` - Stop 3A balanced primary scientific read: 72/72 runs passed, aggregate tables were exported, tuned NICME is promising on balanced Spider and BreaKHis, and the key caveat is that BreaKHis has mean-floor-compliant NICME results but not strict all-seed floor stability yet.
- `experiment_plan_stop_3b_4_updated.md` - revised Stop 3B and Stop 4 execution plan after Stop 3A: run imbalance decoupling with `vit` and `timm_dinov3_vit_lora`, then missing backbone ablations with ConvNeXt families, while including `nicme_logit_adjustment` because it is currently stronger than `nicme_hybrid` in several Stop 3A comparisons.
- `experiment_plan_stop_3b_4a_results_stop4b.md` - completed Stop 3B and Stop 4A read: 108/108 Stop 3B and 36/36 Stop 4A runs passed with zero failures; Stop 4A produced strict all-seed balanced NICME winners for Spider (`convnext + nicme_hybrid + calibrated_threshold`) and BreaKHis (`timm_dinov3_convnext_lora + nicme_logit_adjustment + calibrated_threshold`), so Stop 4B was revised to a focused 90-run cost-ratio sensitivity pass using those application-specific backbones.
- `stop_3_4_complete_results_summary.md` - complete Stop 3 and Stop 4 handoff after Stop 4B finished: 306 successful planned runs, one Spider Stop 4B segfault retried successfully, strict balanced-data NICME winners from Stop 4A, focused Stop 4B cost-ratio sensitivity results, and paper-claim wording/caveats.

## Standing Accuracy Instruction

The user explicitly asked that the following instruction be remembered and followed:

> "Ensure that your response is absolutely correct, thoroughly investigated, up to date for 2026, and does not contain any errors or hallucinations.
> Be as thorough and detailed as possible in your research. Cite your sources if needed.
> Be unbiased in your response and show no inherent lean towards agreeing or disagreeing with me. "

## Validation Scope

These notes were created from local repository inspection on 2026-04-27 and extended with the binary-first NICME Research Proposal implementation record on 2026-04-28. Source files, configs, docs, generated result manifests, dataset counts, git status, and the nested playground project were inspected. Large binary artifacts such as image files, checkpoint tensors, PNG/PDF/HTML plots, and model weight binaries were summarized by path, size, file count, or adjacent metadata rather than read byte-for-byte.
