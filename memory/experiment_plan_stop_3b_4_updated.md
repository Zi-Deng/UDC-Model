# Stop 3B And Stop 4 Updated Execution Memory

Generated: 2026-04-28

This memory note records the revised execution plan after Stop 2, Stop 2A-2C, and Stop 3A.

## Key Observed Facts

- Stop 3A finished `72/72` balanced primary runs.
- Balanced Spider is now favorable to the NICME family.
- Balanced BreaKHis is promising but not strict all-seed floor-stable.
- `nicme_logit_adjustment` is currently stronger than `nicme_hybrid` in several mean-floor-compliant comparisons.
- `calibrated_threshold` is the most useful binary deployment mode for the study objective; plain `calibrated_cost_min` often pushes recall up at too much accuracy cost.

## Stop 3B Queue

Run:

- datasets: `spider_target_minority`, `spider_target_majority`, `breakhis_natural`
- models: `vit`, `timm_dinov3_vit_lora`
- methods: all six implemented methods
- seeds: `42,43,44`
- decision modes: `argmax`, `calibrated_cost_min`, `calibrated_threshold`

Projected runtime: about `5-6` hours on the RTX 5090.

## Stop 4 Queue Revision

Stop 4A should run the missing balanced-data backbone ablation:

- datasets: `spider_balanced`, `breakhis_balanced`
- models: `convnext`, `timm_dinov3_convnext_lora`
- methods: `ce_calibrated_cost_min`, `nicme_logit_adjustment`, `nicme_hybrid`
- seeds: `42,43,44`

The original plan only emphasized `nicme_hybrid`, but Stop 3A makes it scientifically necessary to include `nicme_logit_adjustment` as the current strongest NICME candidate.

Stop 4B cost-ratio sensitivity should wait until Stop 3B and Stop 4A summaries identify the best model/method pair per application.

## Live Execution Note

Started 2026-04-28 in tmux session `nicme_stop3b_stop4a`.

Artifacts:

- launcher: `results/stop3b_stop4a_chain/launch_stop3b_stop4a.sh`
- chain logs: `results/stop3b_stop4a_chain/chain.stdout.log`, `results/stop3b_stop4a_chain/chain.stderr.log`
- Stop 3B root: `results/stop3b_imbalance_decoupling`
- Stop 4A root: `results/stop4a_backbone_ablation`

Initial check passed: Stop 3B generated `108` configs, run 1 completed successfully with return code `0` in `67.73` seconds, checkpoint cleanup worked, and run 2 started automatically.

Monitoring additions:

- `scripts/analyze_stop_results.py` now aggregates stop logs and metrics into full decision rows, aggregate summaries, ranked summaries, and a scientific read. It groups by cost ratio/cost matrix for Stop 4B.
- `scripts/run_stop4b_cost_ratio.py` now supports Stop 4B balanced cost-ratio sweeps with target-class false-negative ratios such as `1,2,5,10,20` and reverse error cost fixed at `1`.

## Interpretation Guardrails

- Do not claim NICME hybrid is the best method unless later evidence changes the current read.
- It is fairer to refer to the proposed method family as NICME logit adjustment / NICME hybrid until the final ablations are complete.
- Balanced results remain the central evidence for decoupling cost sensitivity from class imbalance.
- Natural and target-imbalanced results are deployment-realism tests, not substitutes for balanced evidence.
- `timm_dinov3_convnext_lora` uses ConvNeXt MLP adapter targets, not attention q/v LoRA.
