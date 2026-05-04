# NICME Stop-Gated Experiment Master Memory

Last updated: 2026-04-28

Full plan:

- `docs/experiment_plans/STOP_GATED_MASTER_PLAN.md`

Mandatory protocol:

- After each completed stop, create `docs/experiment_plans/STOP_N_results_and_next_plan.md`.
- Also create `memory/experiment_plan_stop_N.md`.
- Preserve the prior plan verbatim in the stop report.
- Summarize observed data/results.
- Explain what changed and why.
- Write a decision-complete plan for the next stop.
- Ask: `Are the results satisfactory enough to continue, revise, or pause?`
- Do not begin the next stop until the user approves.

Command:

```bash
nicme-experiment-stop --stop N --archived-plan PATH_TO_PRIOR_PLAN --split-dir name=PATH --results-dir name=PATH --notes "What changed and why"
```

Current state:

- The master stop-gated plan is archived.
- No stop has been executed by this memory note.
- Do not assume BreaKHis has been downloaded, any GPU training has run, or any Stop 0/1/2/3/4 report exists until the corresponding `STOP_N_results_and_next_plan.md` file is present.

Stop sequence:

- Stop 0: data audit.
- Stop 1: smoke runs.
- Stop 2: prototype runs.
- Stop 3: main binary paper runs.
- Stop 4: ablations and robustness.

Paper-claim guardrail:

- Balanced-dataset gains are required before claiming NICME decouples cost sensitivity from class imbalance.
- Natural/imbalanced gains alone are not enough.
