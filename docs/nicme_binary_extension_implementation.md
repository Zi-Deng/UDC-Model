# NICME Binary-First Extension Implementation Notes

This implementation phase adds the infrastructure needed to run the Research Proposal's binary-first extension while keeping the current spider workflow intact.

## Core Convention

All new utilities use:

```text
C[true_label][predicted_label]
```

Rows are true labels. Columns are predictions. Diagonal entries should be zero for ordinary misclassification-cost experiments.

## Data Preparation

Prepare the current spider dataset:

```bash
nicme-prepare-data --dataset spider --input-dir data/2_class_black_widows --output-dir data/prepared/spider
```

This writes natural, balanced, target-minority, and target-majority spider split directories. The controlled imbalance variants are downsampled after stratified splitting so the target class is approximately 25% or 75% of each split.

Prepare BreaKHis:

```bash
nicme-prepare-data --dataset breakhis --download --extract --raw-dir data/raw/breakhis --output-dir data/prepared/breakhis
```

The BreaKHis archive is large, about 4.27 GB, so this command is intentionally explicit. It creates a manifest and patient-level natural/balanced splits. The official BreaKHis page reports 2,480 benign and 5,429 malignant images across 40X, 100X, 200X, and 400X magnifications, so patient-level splitting is mandatory to avoid leakage.

## Calibration Split

Prepared splits use:

- `train.csv`: model weights
- `validation.csv`: checkpoint and HPO selection
- `calibration.csv`: temperature scaling and post-hoc cost-min decisions
- `test.csv`: final reporting only

The calibration split is separate so post-training temperature fitting does not leak validation or test information.

Balanced profile configs use the ordinary accuracy gap in `selection_score`. Natural or controlled-imbalance profile configs use `selection_accuracy_metric="balanced_accuracy"` so a majority-class shortcut is penalized.

## Methods

Supported loss names now include:

- `ce`
- `ce_calibrated_cost_min`
- `menon_logit_adjusted`
- `cs_regularized_ce`
- `nicme_logit_adjustment`
- `nicme_hybrid`

`ce_calibrated_cost_min` trains with CE; its distinction appears during evaluation through calibrated minimum-expected-cost inference.

## Model Backends

Legacy configs continue using `model_backend="custom"` with custom ResNet/ConvNeXt.

New backends:

- `hf_auto`: `AutoModelForImageClassification`
- `hf_backbone`: `AutoModel` plus a classifier head
- `dinov3_feature`: DINOv3-style `AutoModel` plus a classifier head

LoRA is enabled through `peft_enabled=true` and the `peft_*` config fields.

## Runtime Tiers

Use `scripts/run_binary_experiments.py` to create planned runs:

```bash
nicme-run-binary-experiments --base-config config/nicme_spider_balanced_dinov3_vits_lora.json --tier tier1
nicme-run-binary-experiments --base-config config/nicme_spider_balanced_dinov3_vits_lora.json --tier tier0 --model-family dinov3_vit
```

Available model-family presets are `convnext`, `vit`, `dinov3_convnext`, and `dinov3_vit`. Tier defaults are conservative: Tier 2 uses DINOv3-ViT-S + LoRA only, while Tier 3 expands to all four families for backbone ablation.

Do not run Tier 2 until Tier 0 and Tier 1 complete within the RTX 5090 time budgets.

## Verification

Focused unit tests live in `tests/`:

```bash
micromamba run -n ml pytest -q
micromamba run -n ml python -m py_compile nicme/*.py scripts/*.py utils/*.py model/*.py
```
