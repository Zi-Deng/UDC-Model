# NICME Workspace Migration - 2026-05-01

## Summary

The active NICME workspace was consolidated into the canonical
repository path:

- Canonical root after migration: `/mnt/storage/github/NICME`
- Active source before migration: `/mnt/storage/github/NICME`
- Older NICME clone before migration: `/mnt/storage/github/NICME`

Both source directories were already on `/mnt/storage`; this migration fixes the
split-workspace problem rather than a root-filesystem placement problem.

## Preserved External Backups

- Previous `/mnt/storage/github/NICME` clone:
  `/mnt/storage/github/archive/NICME_pre_migration_20260501_063437`
- Quarantined local repo caches from the active tree:
  `/mnt/storage/github/archive/NICME_repo_caches_20260501_063437`

The old clone was preserved outside the canonical repository rather than
overwritten.

## Git State

- Old NICME clone HEAD: `0a93186eb4d1a2f1ff7f9a0e7074802504eebb8a`
- Active tree HEAD now at canonical root:
  `762a868de2036b03f02493f9a119da72432ab553`

The active tree retained its existing uncommitted multiclass implementation,
docs, memory, results, data, and checkpoint state.

## Preserved Old Untracked Files

The previous NICME clone had five untracked EyePACS helper files. Copies are
preserved here for inspection:

- `old_nicme_untracked/config/eyepacs_5class.json`
- `old_nicme_untracked/config/eyepacs_5class_reg.json`
- `old_nicme_untracked/scripts/create_dr_csvs.py`
- `old_nicme_untracked/scripts/preprocess_dr_data.py`
- `old_nicme_untracked/scripts/setup_dr_data.sh`

These are archived historical helpers, not the active multiclass data-prep
implementation.

## Size Snapshot After Migration

- `data/`: 145G
- `checkpoints/`: 9.0G
- `results/`: 625M
- `memory/`: 420K
- `docs/`: 792K
- `nicme/`: 132K
- `scripts/`: 204K
- `tests/`: 56K
- In-repo migration archive: 60K
- Old NICME clone backup: 26G
- Quarantined repo-local caches: 409M

## Cache Policy

The following repo-local caches were removed from the canonical root and
quarantined externally:

- `.cache/`
- `.pytest_cache/`
- `.ruff_cache/`
- all `__pycache__/` directories

Shared large-artifact caches remain outside the repo as intended:

- `HF_HOME=/mnt/storage/huggingface`
- `HF_HUB_CACHE=/mnt/storage/huggingface/hub`
- `HF_DATASETS_CACHE=/mnt/storage/huggingface/datasets`
- `TORCH_HOME=/mnt/storage/torch`
- `TMPDIR=/mnt/storage/tmp/nicme`
- `XDG_CACHE_HOME=/mnt/storage/.cache`
- `MPLCONFIGDIR=/mnt/storage/.cache/matplotlib`

## Path Reference Policy

Active code and current memory were updated to use
`/mnt/storage/github/NICME`. Historical logs and archived helper files should
use this canonical path for repository-root references.

## Validation

Post-migration checks completed from `/mnt/storage/github/NICME`:

- Canonical root verified with `git rev-parse --show-toplevel`.
- The previous sibling workspace path is no longer used.
- Active scripts/docs/memory/results launch files no longer reference the old
  workspace path.
- MC0 readiness for EyePACS DR and PMI Pills passed for both `balanced` and
  `natural` variants. The validation report is in
  `validation/mc0_readiness/`.
- Storage preflight for training passed with all large caches under
  `/mnt/storage`.
- Full test suite passed with `PYTHONPATH=. micromamba run -n ml pytest -p no:cacheprovider`:
  `49 passed`.
- Repo-local caches created during validation were quarantined externally.
