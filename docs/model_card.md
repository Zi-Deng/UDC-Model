# NICME Model Card

## Model

NICME currently trains custom ResNet-50-style and ConvNeXt image classifiers for cost-sensitive image classification.

The main paper-facing configuration uses:

- Architecture: ResNet-50-style custom implementation.
- Initialization: `weights/pytorch_model.bin`.
- Input size: 224x224 transforms through `CustomImageProcessor`.
- Optimizer/trainer: HuggingFace `Trainer` with AdamW defaults.
- Loss: `CELogitAdjustmentRegularized` for the hybrid NICME method.

## Intended Use

Research on asymmetric, non-identical cost matrices for image classification. The spider task is a case study where different misclassification directions may have different practical consequences.

## Not Intended For

Deployment as a safety-critical biological identification system without external validation, calibrated uncertainty analysis, expert review, and dataset governance.

## Key Results

See `docs/results_summary.md`.

## Limitations

- Current results are local-workspace results and should be reproduced from a clean clone before submission.
- The binary spider task may not capture real-world image distribution shift.
- Cost choices encode task preferences and should be justified in the paper.

