# NICME Binary Extension Memory

Last updated: 2026-04-28

This memory note summarizes the Research Proposal extension work implemented in the current repo state. For the full human-facing record, see:

- `docs/2026-04-28_nicme_binary_first_extension_plan_and_implementation.md`
- `docs/literature_search_2026_cost_matrix_modern_vision.md`
- `docs/nicme_binary_extension_implementation.md`

## Research Framing

The extension reframes NICME as a binary-first, multiclass-ready framework for studying explicit user-defined misclassification costs independently from class imbalance.

Use this cautious novelty framing:

> Many "cost-sensitive" deep learning papers address class imbalance through class weights, resampling, focal-style losses, or prior/logit corrections. NICME instead evaluates explicit user-defined pairwise misclassification costs under both balanced and imbalanced class distributions, isolating cost sensitivity from class-frequency effects. The extension studies NICME-style cost-matrix logit adjustment and cost-sensitive regularization with DINOv3/LoRA backbones.

Do not claim no ViT cost-sensitive work exists. CSADA includes ViT experiments. Do not claim no EfficientNet cost-sensitive work exists until the EfficientNet diabetic-retinopathy candidate is fully adjudicated.

Pause condition: if later literature search finds a method already combining arbitrary pairwise cost matrices, balanced/imbalanced decoupling, NICME-like logit adjustment, CS regularization, and DINOv3/LoRA, pause and write a gap-analysis memo.

## Cost Convention

The project now standardizes:

```text
C[true_label][predicted_label]
```

Rows are ground truth and columns are predictions. New metrics, decisions, losses, docs, configs, and sweep exports should follow this.

Important implementation note: legacy reference loss paths that used `cost_matrix[predicted][target]` were corrected to `cost_matrix[target][predicted]`.

Core module: `nicme/costs.py`.

## Data Profiles And Splits

Spider:

- Classes: `["black_widow", "false_widow"]`
- Target recall class: `black_widow`
- Cost matrix: `[[0.0, 10.0], [1.0, 0.0]]`
- Variants: natural, balanced, target_minority, target_majority.

BreaKHis:

- Classes: `["benign", "malignant"]`
- Target recall class: `malignant`
- Cost matrix: `[[0.0, 1.0], [10.0, 0.0]]`
- Variants: natural, balanced.
- Preparation downloads/extracts official BreaKHis archive, builds a manifest, parses patient ID/magnification/tumor type, and creates patient-level splits.

Prepared calibrated experiments use:

- `train.csv`
- `validation.csv`
- `calibration.csv`
- `test.csv`

Split target is 70/10/10/10. Validation selects checkpoints/HPO; calibration fits temperature scaling and cost-min decisions; test is final reporting only.

Core files:

- `nicme/data_prep.py`
- `nicme/dataset_profiles.py`
- `scripts/prepare_data.py`
- `config/profiles/*.json`

## Metrics And Selection

Primary selection score:

```text
selection_score = normalized_ATC + 4.0 * recall_gap^2 + 1.0 * accuracy_gap^2
```

Lower is better.

Balanced variants use `selection_accuracy_metric="accuracy"`. Natural and controlled-imbalance variants use `selection_accuracy_metric="balanced_accuracy"`.

Metrics implemented include ATC, normalized ATC, CRR, target recall/FNR/FPR/precision, accuracy, balanced accuracy, macro-F1, AUROC/AUPRC, NLL, Brier, ECE, class prevalence, and confusion matrix.

Key files:

- `nicme/costs.py`
- `utils/utils.py`
- `scripts/hpo_search.py`
- `scripts/cost_matrix_sweep.py`

## Methods

Supported loss/method names:

- `ce`
- `ce_calibrated_cost_min`
- `menon_logit_adjusted`
- `cs_regularized_ce`
- `nicme_logit_adjustment`
- `nicme_hybrid`

Inference modes:

- `argmax`
- `calibrated_cost_min`

`ce_calibrated_cost_min` trains with ordinary CE; its baseline behavior appears at evaluation via temperature-scaled probabilities and the minimum expected cost decision rule.

`menon_logit_adjusted` is an imbalance/long-tail baseline, not a cost-matrix method.

Key files:

- `utils/loss_functions.py`
- `utils/utils.py`
- `scripts/train.py`

## Models And Runtime

Model backends:

- `custom`: legacy custom ResNet/ConvNeXt.
- `hf_auto`: `AutoModelForImageClassification`.
- `hf_backbone`: `AutoModel` plus classifier head.
- `dinov3_feature`: DINOv3-style `AutoModel` plus classifier head.

Experiment runner model presets:

- `convnext`
- `vit`
- `dinov3_convnext`
- `dinov3_vit`

5090-friendly primary target:

- `facebook/dinov3-vits16-pretrain-lvd1689m`
- LoRA defaults: `r=8`, `alpha=16`, `dropout=0.1`, `query,value`, classifier saved.

Dependencies now include `torch>=2.7.1`, `transformers>=4.56`, `peft`, `timm`, and `pytest`.

Key file: `nicme/modeling.py`.

## Experiment Runner

New command:

```bash
nicme-run-binary-experiments --base-config config/nicme_spider_balanced_dinov3_vits_lora.json --tier tier1
```

Tier meanings:

- Tier 0: smoke, minutes.
- Tier 1: prototype, under 1 hour target.
- Tier 2: main binary paper run, under 1 day per dataset target.
- Tier 3: backbone ablation.

Key file: `scripts/run_binary_experiments.py`.

## Commands

Prepare spider:

```bash
nicme-prepare-data --dataset spider --input-dir data/2_class_black_widows --output-dir data/prepared/spider
```

Prepare BreaKHis:

```bash
nicme-prepare-data --dataset breakhis --download --extract --raw-dir data/raw/breakhis --output-dir data/prepared/breakhis
```

Train spider DINOv3 LoRA:

```bash
python scripts/train.py --config config/nicme_spider_balanced_dinov3_vits_lora.json
```

Train BreaKHis DINOv3 LoRA:

```bash
python scripts/train.py --config config/nicme_breakhis_balanced_dinov3_vits_lora.json
```

## Verification

Latest implemented state passed:

```bash
micromamba run -n ml pytest -q
micromamba run -n ml python -m py_compile nicme/*.py scripts/*.py utils/*.py model/*.py
```

Result:

```text
13 passed
```

Focused ruff check on new files also passed.

`pytest.ini` restricts test discovery to `tests/` because an archived playground script named like a test parses CLI args at import time.

## Not Yet Run

Do not assume any of these have happened:

- BreaKHis download/extract.
- Any GPU training run.
- Any Tier 0/Tier 1/Tier 2 experiment.
- Any final paper table generation from real results.

Run Tier 0 before Tier 1, and Tier 1 before any Tier 2 paper-scale run.

