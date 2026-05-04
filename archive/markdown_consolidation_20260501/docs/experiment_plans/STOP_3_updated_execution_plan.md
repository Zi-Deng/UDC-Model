# Stop 3 Updated Execution Plan

Generated: 2026-04-28

## Status

Stop 3 is approved to begin, but the plan is updated after Stop 2A-2C tuning and the storage cleanup pass.

The scientific target is still explicit: determine whether NICME can be the best-performing approach for the widow and tumor applications on cared-class recall and ATC/normalized ATC while preserving the configured accuracy or balanced-accuracy floors.

## Storage Cleanup Before Launch

The cleanup preserved best/final reproducibility artifacts and removed redundant intermediate Trainer checkpoints.

| location | before | after | retained |
| --- | ---: | ---: | --- |
| `NICME/checkpoints` | 19G | 3.1G | `resnet_run` best/final, `resnet_reg` best/final, best checkpoint per HPO trial |
| `NICME` | 34G | 15G | data, results, source, compact checkpoint set |
| `OspideR` | 72G | 63G | root exported HF models; numbered checkpoints removed from selected Trainer exports |
| `AnomaInsect` | 56G | 35G | root exported HF models; ambiguous Lightning/PatchCore checkpoints left untouched |
| `SpiderML` | 36G | 12G | root exported ConvNeXtV2 model; numbered checkpoints removed |
| filesystem free space | 58G | 128G | enough for storage-safe Stop 3 queue |

Ambiguous PatchCore/Lightning checkpoints were not deleted because their result folders do not expose a reliable best/final marker.

## What Changed From The Original Stop 3 Plan

Stop 2A-2C found that raw cost values in NICME logit adjustment were too aggressive for Spider, often collapsing into all-target-class predictions. The adjusted Stop 3 queue therefore uses softened `nicme_logit_cost_scale` values.

Stop 2A-2C also showed that post-hoc `calibrated_threshold` is a stronger and safer operating-point search than plain `calibrated_cost_min` for the binary 10:1 setting. Stop 3 therefore reports all three modes:

- `argmax`
- `calibrated_cost_min`
- `calibrated_threshold`

The first Stop 3 launch is limited to the balanced datasets. This directly tests the central paper claim that cost sensitivity is separable from class imbalance. If NICME cannot win, tie, or show a clear tradeoff advantage on balanced data, imbalanced-only gains will not support the main claim.

## Stop 3A: Balanced Primary Queue

This is the queue to run first for the next several hours.

Datasets:

- `spider_balanced`
- `breakhis_balanced`

Models:

- `vit`: `google/vit-base-patch16-224-in21k`, full fine-tuning
- `timm_dinov3_vit_lora`: `timm/vit_small_patch16_dinov3.lvd1689m`, LoRA on `qkv`, classifier saved

Methods:

- `ce`
- `ce_calibrated_cost_min`
- `menon_logit_adjusted`
- `cs_regularized_ce`
- `nicme_logit_adjustment`
- `nicme_hybrid`

Seeds:

- `42, 43, 44`

Runtime/storage controls:

- epochs: `20`
- early stopping patience: `3`
- `save_total_limit=1`
- delete generated checkpoint directory after metrics/log export
- first unattended launch uses a 4-hour queue budget, then stops cleanly before starting another run

Run count: `2 datasets x 2 models x 6 methods x 3 seeds = 72 planned runs`.

## Tuned Method Settings

| dataset family | method | Stop 3 setting | reason |
| --- | --- | --- | --- |
| Spider | `nicme_logit_adjustment` | `nicme_logit_cost_scale=0.50` | best Spider NICME argmax scale in Stop 2B; less collapse than raw cost 10 |
| Spider | `nicme_hybrid` | `scale=0.50`, `cs_lambda=0.10`, warmup 1 | gentler hybrid setting to reduce false-widow destruction |
| Spider | `cs_regularized_ce` | `cs_lambda=0.10`, warmup 1 | conservative regularizer-only baseline |
| BreaKHis | `nicme_logit_adjustment` | `nicme_logit_cost_scale=0.30` | softer scale from Stop 2C frontier |
| BreaKHis | `nicme_hybrid` | `scale=0.30`, `cs_lambda=0.25`, warmup 1 | best observed NICME tumor family in Stop 2A-2C |
| BreaKHis | `cs_regularized_ce` | `cs_lambda=0.25`, warmup 1 | matches tumor hybrid regularization strength without logit adjustment |

## Stop 3B: Imbalance Decoupling Queue

Run only after inspecting Stop 3A results.

Datasets:

- `spider_target_minority`
- `spider_target_majority`
- `breakhis_natural`

Default model:

- `vit` first if Stop 3A confirms it remains the strongest tuned NICME path.
- `timm_dinov3_vit_lora` second for DINOv3 comparison if runtime remains within the one-day target.

Methods:

- all six methods if Stop 3A shows stable runtime
- otherwise prioritize `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `nicme_logit_adjustment`, and `nicme_hybrid`

Selection metric:

- Spider imbalance variants: balanced accuracy floor `0.75`
- BreaKHis natural: balanced accuracy floor `0.80`

## Decision Criteria

NICME can be reported as strongest for an application only if a NICME method wins or is statistically tied on normalized ATC/selection score while satisfying the relevant recall and accuracy floors.

Balanced-dataset results are mandatory for the central decoupling claim. Natural or target-imbalanced gains alone can support deployment realism, but not the core claim that NICME is not merely class imbalance correction.

If Stop 3A shows CE plus calibrated threshold still clearly beats NICME on Spider, pause before Stop 3B and implement a Spider-specific NICME loss refinement, such as margin-capped logit adjustment or an asymmetric target-class regularizer.

## Launch Command

```bash
micromamba run -n ml python scripts/run_stop3_main.py \
  --phase stop3a_balanced_primary \
  --output-root results/stop3a_balanced_primary \
  --datasets spider_balanced,breakhis_balanced \
  --models vit,timm_dinov3_vit_lora \
  --methods ce,ce_calibrated_cost_min,menon_logit_adjusted,cs_regularized_ce,nicme_logit_adjustment,nicme_hybrid \
  --seeds 42,43,44 \
  --time-budget-hours 4 \
  --cleanup-checkpoints \
  --execute
```

## Source Anchors

- BreaKHis official dataset page: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- DINOv3 Transformers documentation: https://huggingface.co/docs/transformers/model_doc/dinov3
- PEFT LoRA image-classification guide: https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
- Temperature scaling: Guo et al., ICML 2017, https://proceedings.mlr.press/v70/guo17a.html
- Post-hoc cost-sensitive decisions: Domingos MetaCost, https://aiweb.cs.washington.edu/ai/metacost.html
- Cost-sensitive DNN caution and CSADA: https://pubsonline.informs.org/doi/10.1287/ijds.2022.0033
- Logit adjustment baseline: Menon et al., ICLR 2021, https://openreview.net/forum?id=37nvvqkCo5
