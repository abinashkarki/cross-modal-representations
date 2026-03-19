# Results Layout

This repository keeps only small canonical result artifacts in git.

## Current Scale250 Artifacts In Git

- `results/scale250_full/baseline/robustness_opt_full/robustness_stats.json`
- `results/scale250_full/aligned5/robustness/robustness_stats.json`
- `results/scale250_full/baseline25_extension/architecture_analysis/architecture_summary.json`
- Paper-facing figures under `results/scale250_full/**`

## Heavy Artifacts Excluded From Git

- Scale250 compiled `replication_results.json`
- raw per-model embedding dumps
- caches, logs, smoke runs, and partial experiments

Those are indexed in `artifacts/release_manifest.json`. In a local working copy, the excluded files
have been moved under `.local_artifacts/results/`.

## Legacy Comparison Artifacts Kept In Git

- `results/baseline/`
- `results/aligned5/`
- `results/v2_change_assets/`
