# STOP 0 Results And Next Plan

Generated: 2026-04-28 03:59:36

## Stop Status

This artifact archives the plan used for Stop 0, summarizes observed artifacts supplied to the stop reporter, and defines the next plan. Do not begin the next stop until the user checkpoint at the end is approved.

## Observed Results Summary

### Data Audit: spider_natural

- Path: `data/prepared/spider/splits/natural`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2099 | 69.99% | 70.00% | {'0': 1050, '1': 1049} | {'0': 0.5002382086707956, '1': 0.4997617913292044} | {'black_widow': 1050, 'false_widow': 1049} | 2099 | 0 | {} | {} |
| validation | 300 | 10.00% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |
| calibration | 300 | 10.00% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |
| test | 300 | 10.00% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: spider_balanced

- Path: `data/prepared/spider/splits/balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2098 | 69.98% | 70.00% | {'0': 1049, '1': 1049} | {'0': 0.5, '1': 0.5} | {'black_widow': 1049, 'false_widow': 1049} | 2098 | 0 | {} | {} |
| validation | 300 | 10.01% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |
| calibration | 300 | 10.01% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |
| test | 300 | 10.01% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: spider_target_minority

- Path: `data/prepared/spider/splits/target_minority`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1398 | 69.97% | 70.00% | {'0': 349, '1': 1049} | {'0': 0.24964234620886983, '1': 0.7503576537911302} | {'black_widow': 349, 'false_widow': 1049} | 1398 | 0 | {} | {} |
| validation | 200 | 10.01% | 10.00% | {'0': 50, '1': 150} | {'0': 0.25, '1': 0.75} | {'black_widow': 50, 'false_widow': 150} | 200 | 0 | {} | {} |
| calibration | 200 | 10.01% | 10.00% | {'0': 50, '1': 150} | {'0': 0.25, '1': 0.75} | {'black_widow': 50, 'false_widow': 150} | 200 | 0 | {} | {} |
| test | 200 | 10.01% | 10.00% | {'0': 50, '1': 150} | {'0': 0.25, '1': 0.75} | {'black_widow': 50, 'false_widow': 150} | 200 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: spider_target_majority

- Path: `data/prepared/spider/splits/target_majority`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1400 | 70.00% | 70.00% | {'0': 1050, '1': 350} | {'0': 0.75, '1': 0.25} | {'black_widow': 1050, 'false_widow': 350} | 1400 | 0 | {} | {} |
| validation | 200 | 10.00% | 10.00% | {'0': 150, '1': 50} | {'0': 0.75, '1': 0.25} | {'black_widow': 150, 'false_widow': 50} | 200 | 0 | {} | {} |
| calibration | 200 | 10.00% | 10.00% | {'0': 150, '1': 50} | {'0': 0.75, '1': 0.25} | {'black_widow': 150, 'false_widow': 50} | 200 | 0 | {} | {} |
| test | 200 | 10.00% | 10.00% | {'0': 150, '1': 50} | {'0': 0.75, '1': 0.25} | {'black_widow': 150, 'false_widow': 50} | 200 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: breakhis_natural

- Path: `data/prepared/breakhis/splits/natural`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 5616 | 71.01% | 70.00% | {'0': 1724, '1': 3892} | {'0': 0.306980056980057, '1': 0.6930199430199431} | {'benign': 1724, 'malignant': 3892} | 56 | 0 | {'40': 1435, '100': 1504, '200': 1416, '400': 1261} | {'A': 383, 'DC': 2587, 'F': 667, 'LC': 351, 'MC': 737, 'PC': 217, 'PT': 235, 'TA': 439} |
| validation | 768 | 9.71% | 10.00% | {'0': 205, '1': 563} | {'0': 0.2669270833333333, '1': 0.7330729166666666} | {'benign': 205, 'malignant': 563} | 8 | 0 | {'40': 201, '100': 191, '200': 194, '400': 182} | {'DC': 489, 'F': 205, 'PC': 74} |
| calibration | 815 | 10.30% | 10.00% | {'0': 269, '1': 546} | {'0': 0.3300613496932515, '1': 0.6699386503067485} | {'benign': 269, 'malignant': 546} | 8 | 0 | {'40': 203, '100': 207, '200': 210, '400': 195} | {'A': 61, 'DC': 203, 'F': 142, 'LC': 201, 'PC': 142, 'TA': 66} |
| test | 710 | 8.98% | 10.00% | {'0': 282, '1': 428} | {'0': 0.3971830985915493, '1': 0.6028169014084507} | {'benign': 282, 'malignant': 428} | 9 | 0 | {'40': 156, '100': 179, '200': 193, '400': 182} | {'DC': 172, 'LC': 74, 'MC': 55, 'PC': 127, 'PT': 218, 'TA': 64} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: breakhis_balanced

- Path: `data/prepared/breakhis/splits/balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3448 | 69.52% | 70.00% | {'0': 1724, '1': 1724} | {'0': 0.5, '1': 0.5} | {'benign': 1724, 'malignant': 1724} | 56 | 0 | {'40': 887, '100': 925, '200': 847, '400': 789} | {'A': 383, 'DC': 1134, 'F': 667, 'LC': 158, 'MC': 331, 'PC': 101, 'PT': 235, 'TA': 439} |
| validation | 410 | 8.27% | 10.00% | {'0': 205, '1': 205} | {'0': 0.5, '1': 0.5} | {'benign': 205, 'malignant': 205} | 8 | 0 | {'40': 109, '100': 104, '200': 108, '400': 89} | {'DC': 179, 'F': 205, 'PC': 26} |
| calibration | 538 | 10.85% | 10.00% | {'0': 269, '1': 269} | {'0': 0.5, '1': 0.5} | {'benign': 269, 'malignant': 269} | 8 | 0 | {'40': 134, '100': 129, '200': 144, '400': 131} | {'A': 61, 'DC': 97, 'F': 142, 'LC': 110, 'PC': 62, 'TA': 66} |
| test | 564 | 11.37% | 10.00% | {'0': 282, '1': 282} | {'0': 0.5, '1': 0.5} | {'benign': 282, 'malignant': 282} | 9 | 0 | {'40': 121, '100': 142, '200': 150, '400': 151} | {'DC': 110, 'LC': 47, 'MC': 33, 'PC': 92, 'PT': 218, 'TA': 64} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`

## What Changed And Why

Stop 0 data prep completed. During audit, the first balanced split generation exposed a pandas groupby behavior that dropped the numeric label column from balanced CSVs; this was fixed in nicme/data_prep.py, covered by a regression test, and all spider/BreaKHis splits were regenerated. Spider produced natural, balanced, target-minority, and target-majority splits with no missing images, numeric label columns present, no split overlap, and ratios within tolerance. BreaKHis downloaded the official 4,273,561,758-byte archive, extracted 7,909 PNG images, and produced natural and balanced patient-level splits with no missing images, numeric label columns present, no patient overlap, and ratios within 5 percentage points of 70/10/10/10. BreaKHis manifest matches the expected image and label counts: 2,480 benign and 5,429 malignant. The parser produced 81 patient groups rather than the commonly cited 82 because one numeric patient code spans two malignant tumor types; the split keeps that code grouped together conservatively to avoid leakage.

## Next Plan: Stop 1 Smoke Runs

- Use only audited, approved data splits.
- Run balanced spider and a balanced BreaKHis subset.
- Models: ConvNeXt, ViT, DINOv3-ViT.
- Methods: `ce`, `nicme_hybrid`.
- Seeds: 1; epochs: 1.
- Confirm no crashes, no NaNs, valid metric export, and no calibration/test leakage.
- Record runtime per run and projected Tier 1 runtime.


## User Checkpoint

Are the results satisfactory enough to continue, revise, or pause?

## Archived Previous Plan

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
