# NICME Compute Report

Updated: 2026-05-04

This document captures the compute assumptions visible from the current repository. Exact hardware should still be recorded before paper submission.

## Current Evidence

- Training scripts detect CUDA, then MPS, then CPU.
- Current PMI-10 NICME v3 HPO completed `108/108` grid rows at LR `5e-5`.
- The validation-selected PMI-10 config uses `timm/convnext_base.fb_in22k_ft_in1k`, 224px inputs, batch size `16`, gradient accumulation `4`, cosine LR schedule, `32` epoch ceiling, and early stopping patience `5`.
- Final binary Stop 3/4 evidence contains 306 successful final planned runs and 918 exported decision rows.
- Historical two-class HPO logs report 20 Optuna trials completing in about 18 minutes on CUDA; that output is now archived and is not the current best-model workflow.
- Current local storage footprint from this workspace is approximately: `data/` 145G, `checkpoints/` 327G, `results/` 666M, and `playground/cost_sensitive_loss_classification/` 5.4G.

## Compute To Report In The Paper Artifact

Before submission, record:

- GPU model and count.
- CPU model and RAM.
- CUDA/PyTorch versions from the active `ml` environment.
- Runtime for:
  - one standard binary training run,
  - one NICME hybrid binary run,
  - one PMI-10 baseline run,
  - one PMI-10 NICME v3 HPO grid row,
  - the full PMI-10 108-row HPO wall time,
  - the final binary Stop 3/4 sequences.
- Total number of runs used for each paper table.
- Approximate storage footprint for retained checkpoints and results after archive cleanup.
