# NICME Memory Index

This folder is a Codex/GPT memory pack for this repository, similar in purpose to a `CLAUDE.md` onboarding file but split into focused documents.

Start here:

- `repository_overview.md` - what this repo is, its layout, stack, and source-of-truth files.
- `commands_and_environment.md` - environment setup, common commands, run directories, and verification notes.
- `training_pipeline.md` - parent repository training flow, datasets, models, losses, metrics, and outputs.
- `sweeps_hpo_and_playground.md` - cost-matrix sweeps, HPO, comparison scripts, and the nested playground project.
- `data_configs_and_artifacts.md` - local data, config files, weights, results, checkpoints, and generated artifacts.
- `maintenance_notes.md` - coding conventions, known pitfalls, stale docs, and safety notes for future agents.
- `nicme_binary_extension_2026.md` - Research Proposal extension memory: literature gate, binary data plan, cost convention, BreaKHis prep, metrics, baselines, model backends, experiment tiers, and verification status.

## Standing Accuracy Instruction

The user explicitly asked that the following instruction be remembered and followed:

> "Ensure that your response is absolutely correct, thoroughly investigated, up to date for 2026, and does not contain any errors or hallucinations.
> Be as thorough and detailed as possible in your research. Cite your sources if needed.
> Be unbiased in your response and show no inherent lean towards agreeing or disagreeing with me. "

## Validation Scope

These notes were created from local repository inspection on 2026-04-27 and extended with the binary-first NICME Research Proposal implementation record on 2026-04-28. Source files, configs, docs, generated result manifests, dataset counts, git status, and the nested playground project were inspected. Large binary artifacts such as image files, checkpoint tensors, PNG/PDF/HTML plots, and model weight binaries were summarized by path, size, file count, or adjacent metadata rather than read byte-for-byte.
