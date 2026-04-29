# Stop 3 Updated Execution Memory

Generated: 2026-04-28

Stop 3 is now updated after Stop 2A-2C tuning and the storage cleanup pass.

Key state:

- Free disk under `/home/zi/Work`: `128G` after cleanup.
- `UDC-Model/checkpoints`: reduced from `19G` to `3.1G`.
- Preserved in `UDC-Model`: `resnet_run/checkpoint-437`, `resnet_run/checkpoint-532`, `resnet_reg/checkpoint-551`, `resnet_reg/checkpoint-570`, and one best checkpoint per visible HPO trial.
- Removed redundant numbered Hugging Face Trainer checkpoints from selected `OspideR`, `AnomaInsect`, and `SpiderML` model exports while preserving root exported models.
- Ambiguous Lightning/PatchCore checkpoints in sibling repos were left untouched because no reliable best/final marker was found.

Stop 2A-2C lessons now governing Stop 3:

- Raw NICME cost scale was too aggressive, especially on Spider.
- `nicme_logit_cost_scale` must be dataset-specific.
- `calibrated_threshold` should be reported alongside `argmax` and `calibrated_cost_min`.
- BreaKHis prototype: tuned ViT `nicme_hybrid` was the best observed NICME path and slightly beat the best CE/ConvNeXt row by normalized ATC/selection score.
- Spider prototype: CE plus calibrated threshold still beat NICME; Spider must be treated as the high-risk test of the method, not as a presumed win.

Immediate Stop 3A queue:

- datasets: `spider_balanced`, `breakhis_balanced`
- models: `vit`, `timm_dinov3_vit_lora`
- methods: `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `cs_regularized_ce`, `nicme_logit_adjustment`, `nicme_hybrid`
- seeds: `42,43,44`
- decision modes: `argmax,calibrated_cost_min,calibrated_threshold`
- checkpoint policy: delete generated run checkpoint directory after metrics/log export
- launch budget: 4 hours for the first unattended pass

Tuned settings:

- Spider `nicme_logit_adjustment`: `nicme_logit_cost_scale=0.50`
- Spider `nicme_hybrid`: `nicme_logit_cost_scale=0.50`, `cs_lambda=0.10`, warmup 1
- Spider `cs_regularized_ce`: `cs_lambda=0.10`, warmup 1
- BreaKHis `nicme_logit_adjustment`: `nicme_logit_cost_scale=0.30`
- BreaKHis `nicme_hybrid`: `nicme_logit_cost_scale=0.30`, `cs_lambda=0.25`, warmup 1
- BreaKHis `cs_regularized_ce`: `cs_lambda=0.25`, warmup 1

Run script:

- `scripts/run_stop3_main.py`

Primary docs artifact:

- `docs/experiment_plans/STOP_3_updated_execution_plan.md`

Stop 3A should be inspected before launching Stop 3B imbalance variants. If CE plus calibrated threshold still clearly beats NICME on balanced Spider, pause and revise the Spider NICME loss instead of treating imbalanced gains as evidence for the main decoupling claim.
