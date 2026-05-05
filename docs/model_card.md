# NICME Model Card

Updated: 2026-05-04

## Model

NICME trains cost-sensitive image classifiers across binary spider, BreaKHis, and multiclass/PMI pill experiments. The current live results include custom ResNet-style models, ConvNeXt backbones, timm models, and official DINOv3 LoRA variants.

The current paper-facing PMI-20 configuration uses:

- Architecture: `timm/convnext_base.fb_in22k_ft_in1k`.
- Dataset: `data/prepared/pmi_pills/splits/balanced`.
- Input size: 224x224.
- Decision mode: argmax.
- Loss: `nicme_hybrid`.
- Main paper hyperparameters: alpha `0.5`, lambda `0.1`, LR `5e-5`.
- Evaluation protocol: three training seeds, fixed balanced PMI-20 split, mean +/- sample standard deviation.

Historical binary configurations include ResNet-50-style custom implementations initialized from `weights/pytorch_model.bin` and HuggingFace Trainer runs with cost-sensitive NICME losses.

## Intended Use

Research on asymmetric, non-identical cost matrices for image classification, especially settings where recall for selected target classes and average test cost matter more than raw accuracy alone.

## Not Intended For

Deployment as a safety-critical biological, medical, or pharmaceutical identification system without external validation, calibrated uncertainty analysis, expert review, data governance, and application-specific risk analysis.

## Key Results

Use the current docs and result indexes:

- Paper results summary: `docs/paper_results_summary.md`
- Canonical PMI-20 paper table: `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/pmi20_sota_table.md`
- PMI-10 supporting HPO and SOTA/baseline summary: `docs/pmi10_hpo_sota_summary.md`
- Binary experiment summary: `docs/binary_experiment_summary.md`
- Multiclass experiment summary: `docs/multiclass_experiment_summary.md`
- Results index: `results/README.md`

## Limitations

- Current results are local-workspace results and should be reproduced from a clean clone before submission.
- The current PMI-20 paper table uses one fixed split and three training seeds; it does not measure split-resampling uncertainty.
- MC3 multiclass results remain incomplete and should not be used for final multiclass claims.
- The binary spider task may not capture real-world image distribution shift.
- Cost choices encode task preferences and should be justified in the paper or application documentation.
