# NICME Compute Report

This document captures the compute assumptions visible from the current repository. It should be expanded with exact hardware before submission.

## Current Evidence

- Training scripts detect CUDA, then MPS, then CPU.
- HPO logs in historical docs report 20 Optuna trials completing in about 18 minutes on CUDA.
- Standard tuned configs use 30 epoch ceilings with early stopping patience 5.
- Batch size is 32 for the parent and hybrid NICME configs.
- Gradient accumulation is 4 in the HuggingFace Trainer configuration.

## Compute To Report In The Paper Artifact

Before submission, record:

- GPU model and count.
- CPU model and RAM.
- CUDA/PyTorch versions from the active `ml` environment.
- Runtime for:
  - one standard training run,
  - one hybrid NICME run,
  - one 19-value sweep,
  - HPO search.
- Total number of runs used for the paper table.
- Approximate storage footprint for checkpoints and results.

