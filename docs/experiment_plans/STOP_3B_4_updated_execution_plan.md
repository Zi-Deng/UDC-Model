# Stop 3B And Stop 4 Updated Execution Plan

Generated: 2026-04-28

## Purpose

This plan updates the remaining Stop 3 and Stop 4 work after observing all Stop 2, Stop 2A-2C, and Stop 3A results. The objective remains to evaluate whether NICME can be the best-performing family for the widow and tumor applications under explicit user-defined misclassification costs, optimizing cared-class recall and ATC/normalized ATC while preserving the configured accuracy or balanced-accuracy floors.

The plan is intentionally revised rather than blindly continuing the original queue. Stop 3A changed the scientific read in three important ways:

- NICME is now competitive and often strongest on the balanced datasets, which is the central decoupling condition.
- `nicme_logit_adjustment` is currently stronger and more stable than `nicme_hybrid` on mean-floor-compliant rows, especially for BreaKHis.
- `calibrated_threshold` is a crucial binary deployment mode because it optimizes the study objective on the calibration split; plain `calibrated_cost_min` often over-prioritizes recall and can collapse accuracy.

## Input Evidence

### Stop 2 And Stop 2A-2C

Stop 2 showed that the initial prototype pipeline worked but that raw NICME cost pressure could collapse predictions into the cared class. Stop 2A-2C therefore introduced:

- `nicme_logit_cost_scale`
- calibrated-threshold inference
- gentler Spider NICME scales
- BreaKHis-specific NICME scale/lambda settings

### Stop 3A

Stop 3A completed `72/72` balanced primary runs over:

- `spider_balanced`
- `breakhis_balanced`
- `vit`
- `timm_dinov3_vit_lora`
- all six methods
- seeds `42,43,44`

Scientific read:

- `spider_balanced`: favorable to NICME. The best mean-selection rows are NICME rows. Strict all-seed floor stability is best for `timm_dinov3_vit_lora + nicme_hybrid + argmax`, while the best mean-floor-compliant calibrated row is `vit + nicme_logit_adjustment + calibrated_threshold`.
- `breakhis_balanced`: promising but not fully stable. `vit + nicme_logit_adjustment + calibrated_threshold` is the best mean-floor-compliant NICME row, but no BreaKHis aggregate row meets both floors in all three seeds.
- The central paper claim remains viable, but should be stated narrowly: NICME-family methods show balanced-dataset cost-sensitive gains, but the current strongest member is not always the hybrid.

## Updated Stop 3B: Imbalance Decoupling Queue

Stop 3B tests whether the balanced-dataset finding survives controlled imbalance and natural imbalance. These results are not allowed to replace balanced evidence for the central claim, but they are necessary for deployment realism.

Datasets:

- `spider_target_minority`
- `spider_target_majority`
- `breakhis_natural`

Models:

- `vit`
- `timm_dinov3_vit_lora`

Methods:

- `ce`
- `ce_calibrated_cost_min`
- `menon_logit_adjusted`
- `cs_regularized_ce`
- `nicme_logit_adjustment`
- `nicme_hybrid`

Seeds:

- `42,43,44`

Run count:

- `3 datasets x 2 models x 6 methods x 3 seeds = 108` runs

Selection floors:

- Spider imbalance variants: target recall floor `0.95`; balanced-accuracy floor `0.75`
- BreaKHis natural: malignant recall floor `0.97`; balanced-accuracy floor `0.80`

Decision modes:

- `argmax`
- `calibrated_cost_min`
- `calibrated_threshold`

Expected runtime:

- Stop 3A took about `3.31` hours for `72` runs.
- Stop 3B is projected around `5` to `6` hours for `108` runs on the RTX 5090, depending on early stopping.

Execution command:

```bash
micromamba run -n ml python scripts/run_stop3_main.py \
  --phase stop3b_imbalance_decoupling \
  --output-root results/stop3b_imbalance_decoupling \
  --datasets spider_target_minority,spider_target_majority,breakhis_natural \
  --models vit,timm_dinov3_vit_lora \
  --methods ce,ce_calibrated_cost_min,menon_logit_adjusted,cs_regularized_ce,nicme_logit_adjustment,nicme_hybrid \
  --seeds 42,43,44 \
  --time-budget-hours 8 \
  --cleanup-checkpoints \
  --execute
```

## Updated Stop 4: Ablations And Robustness

Stop 4 is split into slices so that GPU time remains useful even if one slice reveals a problem.

### Stop 4A: Missing Backbone Ablation

Stop 3A already covers `vit` and `timm_dinov3_vit_lora` on balanced datasets. Stop 4A adds the missing backbone families on the same balanced splits.

Datasets:

- `spider_balanced`
- `breakhis_balanced`

Models:

- `convnext`
- `timm_dinov3_convnext_lora`

Methods:

- `ce_calibrated_cost_min`
- `nicme_logit_adjustment`
- `nicme_hybrid`

Seeds:

- `42,43,44`

Run count:

- `2 datasets x 2 models x 3 methods x 3 seeds = 36` runs

Rationale:

- The original Stop 4 plan only listed `ce_calibrated_cost_min` and `nicme_hybrid`, but Stop 3A shows that `nicme_logit_adjustment` is a stronger NICME candidate than the hybrid in several places. Excluding it would bias the ablation against the current best NICME configuration.
- `timm_dinov3_convnext_lora` is a ConvNeXt MLP-adapter comparison, not attention-query/value LoRA. This distinction must remain explicit in paper tables and memory notes.

Execution command:

```bash
micromamba run -n ml python scripts/run_stop3_main.py \
  --phase stop4a_backbone_ablation \
  --output-root results/stop4a_backbone_ablation \
  --datasets spider_balanced,breakhis_balanced \
  --models convnext,timm_dinov3_convnext_lora \
  --methods ce_calibrated_cost_min,nicme_logit_adjustment,nicme_hybrid \
  --seeds 42,43,44 \
  --time-budget-hours 6 \
  --cleanup-checkpoints \
  --execute
```

### Stop 4B: Cost-Ratio Sensitivity

Run only after Stop 3B and Stop 4A are summarized. This slice should vary cared-class false-negative cost ratios:

- `1:1`
- `2:1`
- `5:1`
- `10:1`
- `20:1`

Recommended scope:

- Datasets: `spider_balanced`, `breakhis_balanced`
- Methods: `ce_calibrated_cost_min`, `nicme_logit_adjustment`, `nicme_hybrid`
- Models: the best Stop 3A/3B model per application, plus one shared cross-application model if runtime allows
- Seeds: `42,43,44`

Purpose:

- Show whether NICME behaves smoothly as the explicit user-defined cost ratio changes.
- Separate cost-sensitivity behavior from class imbalance by doing this first on balanced splits.

### Stop 4C: Calibration And Decision Ablation

This is mostly analysis over already-trained runs:

- Compare `argmax`, `calibrated_cost_min`, and `calibrated_threshold`.
- Report when temperature scaling improves NLL/ECE but does or does not improve ATC.
- Treat calibrated-threshold as a calibrated operating-point selection method, not as evidence that training itself changed.

### Stop 4D: Paper Statistics And Risk Memo

After Stop 3B and Stop 4A are complete:

- Aggregate all rows into one paper table.
- Compute mean, standard deviation, and 95% CI across seeds.
- Run paired exploratory comparisons for NICME family rows versus `ce` and `ce_calibrated_cost_min`.
- Mark significance claims as exploratory when only three seeds are available.
- Write an unresolved-risk memo, especially for BreaKHis all-seed floor instability.

## Execution Decision

The user has approved continuing after Stop 3A. The execution order is:

1. Launch Stop 3B.
2. If Stop 3B completes successfully, launch Stop 4A automatically.
3. Pause before Stop 4B cost-ratio sensitivity until Stop 3B and Stop 4A summaries are inspected, because cost-ratio scope should depend on which model/method pair is empirically strongest.

## Live Execution Note

Execution started on 2026-04-28 in persistent tmux session:

- session: `nicme_stop3b_stop4a`
- launcher: `results/stop3b_stop4a_chain/launch_stop3b_stop4a.sh`
- chain stdout: `results/stop3b_stop4a_chain/chain.stdout.log`
- chain stderr: `results/stop3b_stop4a_chain/chain.stderr.log`
- Stop 3B output root: `results/stop3b_imbalance_decoupling`
- Stop 4A output root: `results/stop4a_backbone_ablation`

Initial health check:

- Stop 3B manifest/config generation succeeded with `108` planned runs.
- Stop 3B run 1, `stop3b_imbalance_decoupling_spider_target_minority_vit_ce_seed42`, completed successfully with return code `0`.
- Runtime for run 1 was `67.73` seconds.
- Checkpoint cleanup worked for run 1.
- Run 2 started automatically, so the queue is progressing.

Support scripts added during monitoring:

- `scripts/analyze_stop_results.py`: joins stop run logs to model-specific metrics files and exports full decision rows, aggregate summaries, ranked summaries, and a scientific read markdown. It groups Stop 4B rows by cost ratio/cost matrix so cost-ratio sweeps are not accidentally averaged together.
- `scripts/run_stop4b_cost_ratio.py`: dedicated Stop 4B runner for balanced cost-ratio sensitivity experiments. It varies the target-class false-negative cost ratio while keeping the reverse error cost at `1`.

## Source Anchors

No new external literature claims are introduced in this plan. The methodological anchors remain:

- Temperature scaling: Guo et al., ICML 2017, https://proceedings.mlr.press/v70/guo17a.html
- MetaCost and post-hoc cost-sensitive decisions: Domingos 1999, https://aiweb.cs.washington.edu/ai/metacost.html
- Logit adjustment baseline: Menon et al., ICLR 2021, https://openreview.net/forum?id=37nvvqkCo5
- CSADA and ViT cost-sensitive nuance: https://pubsonline.informs.org/doi/10.1287/ijds.2022.0033
- DINOv3 model documentation: https://huggingface.co/docs/transformers/model_doc/dinov3
- PEFT LoRA image-classification guide: https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
