# NICME Model Card

Updated: 2026-05-04

## Model

NICME trains cost-sensitive image classifiers across binary spider, BreaKHis, and multiclass/PMI pill experiments. The current live results include custom ResNet-style models, ConvNeXt backbones, timm models, and official DINOv3 LoRA variants.

The current PMI-10 HPO configuration uses:

- Architecture: `timm/convnext_base.fb_in22k_ft_in1k`.
- Dataset: `data/prepared/pmi_pills_10_no_cal/splits/balanced`.
- Input size: 224x224.
- Decision mode: argmax.
- Loss: `nicme_v3_hybrid`.
- Validation-selected hyperparameters: alpha `0.4`, lambda `0.2`, LR `5e-5`.
- Best observed test-row hyperparameters: alpha `0.5`, lambda `0.07`, LR `5e-5`.

Historical binary configurations include ResNet-50-style custom implementations initialized from `weights/pytorch_model.bin` and HuggingFace Trainer runs with cost-sensitive NICME losses.

## Intended Use

Research on asymmetric, non-identical cost matrices for image classification, especially settings where recall for selected target classes and average test cost matter more than raw accuracy alone.

## Not Intended For

Deployment as a safety-critical biological, medical, or pharmaceutical identification system without external validation, calibrated uncertainty analysis, expert review, data governance, and application-specific risk analysis.

## Key Results

Use the current docs and result indexes:

- PMI-10 HPO and SOTA/baseline summary: `docs/pmi10_hpo_sota_summary.md`
- Current PMI-10 HPO root: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/`
- Binary experiment summary: `docs/binary_experiment_summary.md`
- Multiclass experiment summary: `docs/multiclass_experiment_summary.md`
- Results index: `results/README.md`

## Limitations

- Current results are local-workspace results and should be reproduced from a clean clone before submission.
- PMI-10 run 95 is the best observed test row, not the strict validation-selected row.
- MC3 multiclass results remain incomplete and should not be used for final multiclass claims.
- The binary spider task may not capture real-world image distribution shift.
- Cost choices encode task preferences and should be justified in the paper or application documentation.
