# NICME Binary-First Extension: Plan And Implementation Record

Date: 2026-04-28

This document records the planned and implemented work for the Research Proposal extension of NICME. It is intended to be the durable handoff note for future paper work, experiments, and code maintenance.

## Research Target

The extension reframes NICME as a binary-first, multiclass-ready research framework for studying explicit user-defined misclassification costs independently from dataset class imbalance.

The central claim should be phrased conservatively:

> Many "cost-sensitive" deep learning papers address class imbalance through class weights, resampling, focal-style losses, or prior/logit corrections. NICME instead evaluates explicit user-defined pairwise misclassification costs under both balanced and imbalanced class distributions, isolating cost sensitivity from class-frequency effects. The extension studies NICME-style cost-matrix logit adjustment and cost-sensitive regularization with DINOv3/LoRA backbones.

Do not claim that no ViT, ConvNeXt, or EfficientNet has ever been paired with explicit cost-sensitive learning. The literature gate found important counterexamples or near-counterexamples, especially CSADA with ViT experiments and an EfficientNet diabetic-retinopathy candidate.

## Literature Gate Outcome

The literature gate is saved in:

- `docs/literature_search_2026_cost_matrix_modern_vision.md`

Current decision:

- Proceed with implementation.
- Avoid broad novelty claims about all modern backbones.
- Treat Menon-style logit adjustment as an imbalance/long-tail baseline, not a user-defined cost-matrix baseline.
- Treat CSADA as true cost-sensitive prior work that blocks "no ViT cost-sensitive work" claims.
- Keep the novelty focus on explicit user-defined pairwise cost matrices, balanced/imbalanced decoupling experiments, NICME-style logit adjustment plus CS regularization, and DINOv3/LoRA.

Pause condition:

- If later literature search finds a method that already combines arbitrary pairwise cost matrices, balanced/imbalanced decoupling, NICME-like logit adjustment, CS regularization, and DINOv3/LoRA, pause implementation and write a gap-analysis memo before continuing.

## Cost Convention

All new code uses:

```text
C[true_label][predicted_label]
```

Rows are ground-truth labels. Columns are predicted labels. Diagonal entries are expected to be zero for normal misclassification-cost experiments.

Primary implementation:

- `nicme/costs.py`
- `utils/loss_functions.py`
- `utils/utils.py`
- `scripts/cost_matrix_sweep.py`

Important bug fix made during implementation:

- Two legacy reference loss paths used `cost_matrix[predicted][target]`; they were corrected to `cost_matrix[target][predicted]`.
- A test now verifies that `nicme_logit_adjustment` penalizes the high-cost `C[true][pred]` direction.

## Dataset Plan

Main binary tasks:

- Spider: Black Widow vs False Widow.
- Tumor: benign vs malignant using BreaKHis.

Spider labels:

```text
["black_widow", "false_widow"]
```

Spider target recall class:

```text
black_widow
```

Spider default cost matrix:

```text
[[0.0, 10.0],
 [1.0,  0.0]]
```

BreaKHis labels:

```text
["benign", "malignant"]
```

BreaKHis target recall class:

```text
malignant
```

BreaKHis default cost matrix:

```text
[[ 0.0, 1.0],
 [10.0, 0.0]]
```

Implemented data preparation:

- `nicme-prepare-data --dataset spider`
- `nicme-prepare-data --dataset breakhis`
- BreaKHis download has resume support and safe tar extraction.
- BreaKHis filenames are parsed for benign/malignant label, patient ID, magnification, and tumor type.
- BreaKHis uses patient-level splits to avoid leakage.
- Prepared calibrated experiments use 70/10/10/10 train/validation/calibration/test.

Implemented dataset variants:

- Spider natural.
- Spider balanced.
- Spider target-minority controlled imbalance, approximately 25% target class.
- Spider target-majority controlled imbalance, approximately 75% target class.
- BreaKHis natural.
- BreaKHis balanced.

Implemented files:

- `nicme/data_prep.py`
- `nicme/dataset_profiles.py`
- `scripts/prepare_data.py`
- `config/profiles/spider_natural.json`
- `config/profiles/spider_balanced.json`
- `config/profiles/spider_target_minority.json`
- `config/profiles/spider_target_majority.json`
- `config/profiles/breakhis_natural.json`
- `config/profiles/breakhis_balanced.json`

## Calibration Split

Prepared splits use:

- `train.csv`: fit model weights.
- `validation.csv`: select checkpoints and HPO settings.
- `calibration.csv`: fit temperature scaling and post-hoc cost-sensitive decisions.
- `test.csv`: untouched final reporting.

The calibration split is necessary because temperature scaling and cost-minimum inference use fitted post-training parameters. Using validation would couple checkpoint choice and calibration; using test would leak final evaluation information.

Implemented files:

- `nicme/calibration.py`
- `utils/utils.py`

## Metrics And Selection

Primary selection score:

```text
selection_score = normalized_ATC + 4.0 * recall_gap^2 + 1.0 * accuracy_gap^2
```

Lower is better.

```text
recall_gap = max(0, target_recall_floor - target_class_recall)
accuracy_gap = max(0, accuracy_floor - selected_accuracy)
```

Implemented metrics:

- ATC.
- Normalized ATC.
- CRR.
- Target-class recall.
- Target precision.
- Target FNR.
- Target FPR.
- Accuracy.
- Balanced accuracy.
- Macro-F1.
- AUROC and AUPRC for binary tasks.
- NLL.
- Brier score.
- ECE.
- Class prevalence.
- Confusion matrix.

Balanced variants use ordinary accuracy for the accuracy gap. Natural or controlled-imbalance variants use balanced accuracy for the accuracy gap.

Implemented files:

- `nicme/costs.py`
- `utils/utils.py`
- `scripts/hpo_search.py`
- `scripts/cost_matrix_sweep.py`

## Methods And Baselines

Implemented same-pipeline loss names:

- `ce`
- `ce_calibrated_cost_min`
- `menon_logit_adjusted`
- `cs_regularized_ce`
- `nicme_logit_adjustment`
- `nicme_hybrid`

Interpretation:

- `ce`: standard cross entropy.
- `ce_calibrated_cost_min`: trains with CE; evaluation distinguishes it through temperature scaling plus minimum expected cost inference.
- `menon_logit_adjusted`: class-prior/long-tail baseline, not a user-defined cost-matrix method.
- `cs_regularized_ce`: CE plus normalized expected-cost regularization.
- `nicme_logit_adjustment`: existing NICME cost-matrix logit adjustment.
- `nicme_hybrid`: NICME logit adjustment plus normalized cost-sensitive regularization and warmup.

Inference modes:

- `argmax`
- `calibrated_cost_min`

Implemented files:

- `utils/loss_functions.py`
- `utils/utils.py`
- `scripts/train.py`
- `scripts/hpo_search.py`
- `scripts/cost_matrix_sweep.py`

## Models And Runtime Plan

Required model families:

- ConvNeXt.
- ViT.
- DINOv3-ConvNeXt.
- DINOv3-ViT.

Implemented model backends:

- `custom`: existing custom ResNet/ConvNeXt path.
- `hf_auto`: `AutoModelForImageClassification`.
- `hf_backbone`: `AutoModel` plus classifier head.
- `dinov3_feature`: DINOv3-style `AutoModel` plus classifier head.

Implemented model-family presets in `scripts/run_binary_experiments.py`:

- `convnext`
- `vit`
- `dinov3_convnext`
- `dinov3_vit`

Default 5090-oriented target:

- `facebook/dinov3-vits16-pretrain-lvd1689m`
- LoRA enabled with `r=8`, `alpha=16`, `dropout=0.1`; official HF DINOv3 ViT targets `q_proj,v_proj`, timm DINOv3 ViT targets fused `qkv`, timm DINOv3 ConvNeXt targets MLP layers `mlp.fc1,mlp.fc2`, and classifier heads are saved.

Dependencies updated:

- `torch>=2.7.1`
- `transformers>=4.56`
- `peft`
- `timm`
- `pytest`

Implemented files:

- `nicme/modeling.py`
- `requirements.txt`
- `environment.yml`

## Experiment Tiers

Implemented runner:

- `scripts/run_binary_experiments.py`
- `nicme-run-binary-experiments`

Tier defaults:

- Tier 0: smoke, 1 seed, 1 epoch, `ce` and `nicme_hybrid`, model presets `convnext`, `vit`, `dinov3_vit`.
- Tier 1: prototype under 1 hour target, 1 seed, 10 epochs, `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `nicme_hybrid`, model presets `convnext`, `dinov3_vit`.
- Tier 2: main binary paper run under 1 day per dataset target, 5 seeds, all six methods, DINOv3-ViT only by default.
- Tier 3: backbone ablation, 3 seeds, `ce_calibrated_cost_min` and `nicme_hybrid`, all four model presets.

Example:

```bash
nicme-run-binary-experiments --base-config config/nicme_spider_balanced_dinov3_vits_lora.json --tier tier0 --model-family dinov3_vit
```

## User-Facing Commands

Prepare spider:

```bash
nicme-prepare-data --dataset spider --input-dir data/2_class_black_widows --output-dir data/prepared/spider
```

Prepare BreaKHis:

```bash
nicme-prepare-data --dataset breakhis --download --extract --raw-dir data/raw/breakhis --output-dir data/prepared/breakhis
```

Train a prepared spider DINOv3 LoRA run:

```bash
python scripts/train.py --config config/nicme_spider_balanced_dinov3_vits_lora.json
```

Train a prepared BreaKHis DINOv3 LoRA run:

```bash
python scripts/train.py --config config/nicme_breakhis_balanced_dinov3_vits_lora.json
```

Create a binary experiment plan:

```bash
nicme-run-binary-experiments --base-config config/nicme_spider_balanced_dinov3_vits_lora.json --tier tier1
```

## Verification Completed

Commands run successfully:

```bash
micromamba run -n ml pytest -q
micromamba run -n ml python -m py_compile nicme/*.py scripts/*.py utils/*.py model/*.py
```

Test result:

```text
13 passed
```

Focused ruff check on new files also passed.

The experiment runner successfully generated a Tier 0 DINOv3-ViT plan without executing training.

## Added Tests

Tests live in `tests/` and cover:

- Cost convention for ATC and normalized ATC.
- Binary and multiclass minimum expected cost decisions.
- Metrics preserving declared class count when a split lacks one class.
- Selection score behavior with ordinary vs balanced accuracy gap.
- Cost matrix validation and target class resolution.
- Temperature scaling NLL non-worsening on synthetic logits.
- Calibrated cost-min prediction.
- BreaKHis filename parsing.
- Controlled prevalence generation.
- 70/10/10/10 split sizing and patient-disjoint group splits.
- NICME logit-adjustment cost convention.

`pytest.ini` restricts pytest discovery to `tests/` because an archived playground script named like a test parses CLI args at import time.

## Work Not Yet Done

The implementation is infrastructure-complete for the requested phase, but these actions were intentionally not run:

- BreaKHis archive download and extraction.
- Any GPU training experiment.
- Any final paper table generation from real experiment outputs.
- Any full systematic literature review beyond the initial gate needed to proceed.

Before paper claims are finalized:

- Rerun the literature search more systematically.
- Fully adjudicate the EfficientNet diabetic-retinopathy candidate.
- Run Tier 0, then Tier 1, then Tier 2 only after runtime is confirmed on the RTX 5090.
- Verify all result tables include class prevalence.

## File Map

New core modules:

- `nicme/costs.py`
- `nicme/calibration.py`
- `nicme/data_prep.py`
- `nicme/dataset_profiles.py`
- `nicme/modeling.py`

New scripts:

- `scripts/prepare_data.py`
- `scripts/run_binary_experiments.py`

Modified training/evaluation:

- `scripts/train.py`
- `scripts/hpo_search.py`
- `scripts/cost_matrix_sweep.py`
- `utils/loss_functions.py`
- `utils/utils.py`

New configs:

- `config/nicme_spider_balanced_dinov3_vits_lora.json`
- `config/nicme_breakhis_balanced_dinov3_vits_lora.json`
- `config/profiles/spider_natural.json`
- `config/profiles/spider_balanced.json`
- `config/profiles/spider_target_minority.json`
- `config/profiles/spider_target_majority.json`
- `config/profiles/breakhis_natural.json`
- `config/profiles/breakhis_balanced.json`

Docs and memory:

- `docs/literature_search_2026_cost_matrix_modern_vision.md`
- `docs/nicme_binary_extension_implementation.md`
- `docs/2026-04-28_nicme_binary_first_extension_plan_and_implementation.md`
- `memory/nicme_binary_extension_2026.md`
