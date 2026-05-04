# NICME Stop-Gated Experimental Master Plan

Date: 2026-04-28

This is the archived master plan for NICME's staged binary experiments. Each completed stop must generate a new stop report before any later stop begins.

## Summary

Run NICME experiments as a staged, NeurIPS-quality evaluation of cared-class recall and ATC/normalized ATC under explicit user-defined cost matrices, while separating cost sensitivity from class imbalance.

At every stop, Codex must:

- Summarize results from the just-completed stage.
- Archive the plan that governed that stage into both `docs/` and `memory/`.
- Construct a revised plan for the next stage using observed results.
- Ask the user to inspect results and explicitly continue before any next-stage work.

Stop rule:

- Do not begin Stop `N+1` execution until Stop `N` has a saved summary/next-plan artifact and the user has approved continuing.

## Required Stop Artifacts

For each completed stop, create:

- `docs/experiment_plans/STOP_N_results_and_next_plan.md`
- `memory/experiment_plan_stop_N.md`

Each stop artifact must include:

- Observed results summary.
- What changed and why.
- Decision-complete next plan.
- User checkpoint with the exact question: `Are the results satisfactory enough to continue, revise, or pause?`
- Archived previous plan.

Use the command:

```bash
nicme-experiment-stop --stop N --archived-plan PATH_TO_PRIOR_PLAN --split-dir name=PATH --results-dir name=PATH --notes "What changed and why"
```

## Stop 0: Data Audit

- Prepare spider and BreaKHis splits.
- Report split sizes, class prevalence, patient overlap, BreaKHis magnification/tumor-type distributions, and image-load failures.
- Verify 70/10/10/10 train/validation/calibration/test and patient-disjoint BreaKHis splits.
- Archive this master plan, then write the Stop 1 plan based on observed data quality.
- User checkpoint: approve data validity before model training.

## Stop 1: Smoke Runs

- Datasets: balanced spider and balanced BreaKHis subset.
- Models: ConvNeXt, ViT, DINOv3-ViT.
- Methods: `ce`, `nicme_hybrid`.
- Seeds: 1; epochs: 1.
- Success: no crashes, no NaNs, metrics export correctly, calibration split is not used for training.
- Archive Stop 1 plan/results, then write Stop 2 plan with measured runtime and failure fixes.
- User checkpoint: approve moving to prototype runs.

## Stop 2: Prototype Runs

- Datasets: balanced spider and balanced BreaKHis.
- Models: ConvNeXt and DINOv3-ViT.
- Methods: `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `nicme_hybrid`.
- Seeds: 1; epochs: 5 spider, 8 BreaKHis, early stopping enabled.
- Target: under 1 hour total on RTX 5090.
- Success: at least one cost-sensitive method improves ATC or cared-class recall over CE without unacceptable accuracy collapse.
- Archive Stop 2 plan/results, then write Stop 3 plan with selected runtime, seed count, and any method/config revisions.
- User checkpoint: approve main paper-scale runs.

## Stop 3: Main Binary Paper Runs

- Datasets: balanced spider, spider target-minority, spider target-majority, balanced BreaKHis, natural BreaKHis.
- Model: DINOv3-ViT-S + LoRA.
- Methods: all six implemented methods.
- Seeds: 3 for all methods; extend to 5 seeds if Stop 2 runtime predicts completion under 20 hours per dataset, otherwise 5 seeds only for `ce`, `ce_calibrated_cost_min`, and `nicme_hybrid`.
- Epoch cap: 20; early stopping patience 3.
- Success: NICME hybrid reduces normalized ATC or improves cared-class recall on balanced datasets while maintaining configured accuracy floors.
- Archive Stop 3 plan/results, then write Stop 4 ablation plan based on strongest and weakest findings.
- User checkpoint: approve ablations or request reframing.

## Stop 4: Ablations And Robustness

- Backbone ablation: balanced spider and balanced BreaKHis; ConvNeXt, ViT, DINOv3-ConvNeXt, DINOv3-ViT; methods `ce_calibrated_cost_min` and `nicme_hybrid`; 3 seeds.
- Cost-ratio sensitivity: 1:1, 2:1, 5:1, 10:1, 20:1 on balanced spider and BreaKHis.
- Calibration ablation: argmax vs calibrated-cost-min for all trained models.
- Component ablation: CE, CS regularizer only, NICME logit only, NICME hybrid.
- Archive Stop 4 plan/results, then write final paper experiment summary and unresolved-risk memo.
- User checkpoint: decide whether to run extra experiments or freeze results.

## Reporting And Acceptance Criteria

Every stop report must include, when available:

- ATC, normalized ATC, CRR.
- Cared-class recall, FNR, precision, FPR.
- Accuracy, balanced accuracy, macro-F1.
- AUROC/AUPRC for binary tasks.
- NLL, Brier, ECE, fitted temperature.
- Confusion matrix.
- Class prevalence for every split and result row.
- Runtime per run and projected runtime for the next stop.

Paper-level acceptance:

- Balanced-dataset gains are required before claiming cost sensitivity is decoupled from class imbalance.
- Natural/imbalanced gains alone are not enough for the central claim.
- `menon_logit_adjusted` must be interpreted as an imbalance baseline.
- `ce_calibrated_cost_min` must be interpreted as the post-hoc minimum expected cost baseline.
- If NICME hybrid fails to beat post-hoc cost-min on ATC, report that honestly and reframe the contribution around when training-time costs help.

Statistical reporting:

- Mean, standard deviation, and 95% CI across seeds.
- Paired comparisons for `nicme_hybrid` vs `ce` and `nicme_hybrid` vs `ce_calibrated_cost_min`.
- With only 3 seeds, mark significance claims as exploratory.

## Assumptions And Sources

Defaults:

- Hardware: one Linux RTX 5090.
- Primary backbone: `facebook/dinov3-vits16-pretrain-lvd1689m` with LoRA.
- Cost convention: `C[true_label][predicted_label]`.
- Spider target: `black_widow`; BreaKHis target: `malignant`.
- EyePACS/QWK and multiclass experiments stay out of scope.

Source anchors:

- Temperature scaling: Guo et al., ICML 2017, https://proceedings.mlr.press/v70/guo17a.html
- MetaCost / cost-sensitive wrapper baseline: Domingos 1999, https://aiweb.cs.washington.edu/ai/metacost.html
- Logit adjustment baseline: Menon et al., ICLR 2021, https://openreview.net/forum?id=37nvvqkCo5
- CSADA and ViT nuance: https://pubsonline.informs.org/doi/10.1287/ijds.2022.0033
- DINOv3: https://huggingface.co/docs/transformers/model_doc/dinov3
- PEFT LoRA image classification: https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
