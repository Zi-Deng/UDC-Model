# External Artifact Migration Plan

The repository is self-contained for now. Before public camera-ready release, move large artifacts out of git/workspace distribution and replace them with stable external references plus checksums.

## Target Hosting

| Artifact | Preferred host | Reason |
|---|---|---|
| Datasets | Zenodo or Hugging Face Datasets | Persistent DOI or dataset card support |
| Pretrained weights and final checkpoints | Hugging Face Hub or GitHub Releases | Common ML model hosting and versioning |
| Full sweep/result archives | Zenodo | DOI-backed reproducibility bundle |
| Compact CSV/JSON summaries | Git repository | Small and useful for review |

## Migration Steps

1. Freeze final artifact set after paper experiments are complete.
2. Compute SHA256 checksums for every file included in the external bundle.
3. Upload datasets, weights, checkpoints, and full result archives to the selected hosts.
4. Update `docs/artifact_manifest.md` with external URLs, checksums, sizes, and retrieval commands.
5. Update `README.md`, release-view READMEs, and `docs/reproducibility.md` so all commands download or reference external artifacts.
6. Remove large artifacts from git tracking if any remain tracked.
7. Verify a fresh clone can reproduce the compact result tables after downloading artifacts.

## Required Acceptance Check

A fresh clone plus documented artifact downloads must support:

```bash
micromamba env update -n ml -f environment.yml
micromamba activate ml
python scripts/train_reg.py --config config/nicme_2class_spiders_regularized.json
python scripts/compare_sweeps.py
```

