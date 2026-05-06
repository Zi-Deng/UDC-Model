# Binary Camera-Ready Smoke Archive - 2026-05-05

Created: 2026-05-05T01:29:33

Reason: smoke rows were generated while the runner temporarily used `/home/zi/nicme_checkpoints/...` as `checkpoint_root`. The user requested all new files stay on repository storage, so these smoke phase directories were archived before relaunching with repository-local checkpoints.

| Original live path | Archived path | Reason |
|---|---|---|
| `results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505/datasets/spider/seeds/seed_42/smoke` | `archive/binary_camera_ready_home_checkpoint_smoke_20260505/datasets/spider/seeds/seed_42/smoke` | archived smoke phase generated before checkpoint_root was moved back into repository storage |
| `results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505/datasets/spider/seeds/seed_43/smoke` | `archive/binary_camera_ready_home_checkpoint_smoke_20260505/datasets/spider/seeds/seed_43/smoke` | archived smoke phase generated before checkpoint_root was moved back into repository storage |
