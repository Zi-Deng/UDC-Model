# Checkpoint Cleanup Manifest - 2026-05-05

Cleanup time: 2026-05-05T01:25:06

Purpose: free storage for the binary camera-ready SOTA suite while preserving trained model checkpoints.

Policy applied:
- Preserved checkpoint directories, including best and final checkpoint directories.
- Preserved model weights such as `model.safetensors`, `pytorch_model.bin`, adapter weights, configs, trainer state, and preprocessing metadata.
- Deleted only optimizer/scheduler/RNG/scaler training-state files, which are not required for evaluation or using the checkpoint as an initialization anchor.

Summary:
- Removed training-state files: 1869
- Freed bytes: 355662634517
- Freed GiB: 331.24
- Detailed CSV: `removed_training_state_files.csv`

No model-weight files were intentionally deleted by this cleanup.
