# MC-FB2 Official DINOv3 LoRA Stability Summary

Date: 2026-05-01 UTC

## Purpose

MC-FB2 was a post-reboot stability gate before MC2. It repeated the official
Facebook DINOv3 LoRA path for both target multiclass datasets over several
epochs to exercise training, evaluation, checkpointing, CUDA preflight, and
resume-safe run ledger behavior.

## Configuration

- Stop scaffold: `mc1` with overridden `--epochs 5`
- Datasets: `eyepacs_dr`, `pmi_pills`
- Split variant: `balanced`
- Backbones: `facebook_dinov3_vit_lora`, `facebook_dinov3_convnext_lora`
- Methods: `ce`, `nicme_hybrid`
- Seed: `42`
- Per-run timeout: 60 minutes
- GPU canary timeout: 45 seconds
- Cache/storage environment:
  - `HF_HOME=/mnt/storage/huggingface`
  - `HF_HUB_CACHE=/mnt/storage/huggingface/hub`
  - `HF_DATASETS_CACHE=/mnt/storage/huggingface/datasets`
  - `TORCH_HOME=/mnt/storage/torch`
  - `TMPDIR=/mnt/storage/tmp/nicme`
  - `XDG_CACHE_HOME=/mnt/storage/.cache`
  - `MPLCONFIGDIR=/mnt/storage/.cache/matplotlib`

## Result

Status: passed.

- Planned runs: 8
- Completed runs: 8
- Failed runs: 0
- Retry attempts used: 0
- Crash signatures in logs: none found for traceback, CUDA illegal access,
  CUBLAS/CUDNN failures, segmentation fault, kill, or explicit exception text.
- Post-run D-state process check: clean.
- Post-run GPU taint check: clean.
- Post-run CUDA canary: clean.

The run did not require a hard reboot. The preflight/canary guard sequence
remained active before each training job.

## Final Validation Metrics

These are the final validation metrics logged by each run. MC-FB2 is a
stability gate, not a paper-performance result.

| Run | Dataset | Model | Method | Eval epochs logged | Accuracy | Balanced accuracy | Target recall | Normalized ATC | ATC | Runtime |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0001 | eyepacs_dr | facebook_dinov3_vit_lora | ce | 4 | 0.1688 | 0.1624 | 0.6061 | 0.4477 | 7.1620 | 101.6s |
| 0002 | eyepacs_dr | facebook_dinov3_vit_lora | nicme_hybrid | 4 | 0.1375 | 0.1308 | 0.2424 | 0.4820 | 7.7130 | 100.9s |
| 0003 | eyepacs_dr | facebook_dinov3_convnext_lora | ce | 5 | 0.2625 | 0.2561 | 0.7273 | 0.2883 | 4.6120 | 125.0s |
| 0004 | eyepacs_dr | facebook_dinov3_convnext_lora | nicme_hybrid | 5 | 0.2562 | 0.2458 | 0.6667 | 0.2828 | 4.5250 | 124.4s |
| 0005 | pmi_pills | facebook_dinov3_vit_lora | ce | 4 | 0.0813 | 0.0727 | 0.0000 | 0.0919 | 0.9187 | 29.9s |
| 0006 | pmi_pills | facebook_dinov3_vit_lora | nicme_hybrid | 4 | 0.0750 | 0.0682 | 0.0000 | 0.0925 | 0.9250 | 29.4s |
| 0007 | pmi_pills | facebook_dinov3_convnext_lora | ce | 4 | 0.0500 | 0.0517 | 0.0000 | 0.1006 | 1.0060 | 32.2s |
| 0008 | pmi_pills | facebook_dinov3_convnext_lora | nicme_hybrid | 4 | 0.0500 | 0.0606 | 0.0000 | 0.0950 | 0.9500 | 31.4s |

## Interpretation

The official Facebook DINOv3 LoRA training path is now stable enough to restart
MC2. EyePACS produced finite, nontrivial multiclass metrics across both LoRA
backbones. PMI also completed all runs with finite metrics, but the tiny
balanced stability subset still has zero cared-class target recall after only
4 logged evaluation epochs; this should be treated as a signal to tune MC2
carefully, not as a final scientific result.

MC2 should proceed before MC3. MC3 should remain blocked until MC2 selects
stable configurations under the recall-first objective and guardrails.
