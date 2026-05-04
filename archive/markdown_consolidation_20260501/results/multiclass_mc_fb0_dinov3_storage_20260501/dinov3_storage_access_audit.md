# DINOv3 Storage Access Audit

- Timestamp: `2026-05-01T06:07:03.012439+00:00`
- All ready: `True`
- HF user: `zkdeng`
- Download weights: `True`

## Storage Preflight

```text
Storage preflight (model_download): ok=True
Environment:
  HF_HOME=/mnt/storage/huggingface
  HF_HUB_CACHE=/mnt/storage/huggingface/hub
  HF_DATASETS_CACHE=/mnt/storage/huggingface/datasets
  TORCH_HOME=/mnt/storage/torch
  TMPDIR=/mnt/storage/tmp/nicme
  XDG_CACHE_HOME=/mnt/storage/.cache
  MPLCONFIGDIR=/mnt/storage/.cache/matplotlib
Free space:
  root /: 177.62 GB / required 10.00 GB
  home_root /home: 177.62 GB / required 10.00 GB
  user_home /home/zi: 177.62 GB / required 10.00 GB
  storage /mnt/storage: 488.87 GB / required 100.00 GB
```

## Models

| Model | OK | Type | Processor | Parameters | Error |
|---|---:|---|---|---:|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | `True` | `dinov3_vit` | `DINOv3ViTImageProcessorFast` | 21596544 |  |
| `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` | `True` | `dinov3_convnext` | `DINOv3ViTImageProcessorFast` | 27820128 |  |
