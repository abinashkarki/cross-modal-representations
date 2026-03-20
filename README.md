# Cross-Modal Representations

Public repository for the Scale250 cross-modal representation study and paper:
*Small Benchmarks Overstate Cross-Modal Convergence: Evidence from a 250-Concept Confirmatory Benchmark*.

This repo is intentionally lean. It keeps the canonical manuscript, figure pack, manifests, core
scripts, and small paper-facing summary artifacts in git. Heavy compiled outputs, raw embedding
dumps, caches, and image payloads are excluded from git and described in
[artifacts/release_manifest.json](./artifacts/release_manifest.json).

## Reproducibility Modes

This repo supports three different use cases:

1. Read the canonical paper and inspect the small release-facing artifacts already checked into git.
2. Reproduce the paper analyses from compiled artifacts after those heavy bundles have been restored locally.
3. Reconstruct the dataset from source using the public roster, provenance tables, and curation/build scripts.

These modes are intentionally separate. The checked-in
[data/data_manifest_250.json](./data/data_manifest_250.json) is the canonical analysis manifest and
references image payloads that are excluded from git. For a fresh public rebuild, start instead from
[data/concept_roster_250_scaffold.json](./data/concept_roster_250_scaffold.json) and
[docs/dataset_reconstruction.md](./docs/dataset_reconstruction.md).

## Canonical Paper

- Manuscript source: [manuscript/paper.md](./manuscript/paper.md)
- HTML render: [manuscript/paper.html](./manuscript/paper.html)
- LaTeX export: [manuscript/paper.tex](./manuscript/paper.tex)
- Figure pack: [manuscript/figures/scale250](./manuscript/figures/scale250)

## What Is In Git

- Canonical Scale250 manuscript and figure pack
- Canonical manifests and provenance metadata
- Core extraction, compilation, robustness, scaling, and figure scripts
- Small canonical result artifacts for the paper:
  - [results/scale250_full/baseline/robustness_opt_full/robustness_stats.json](./results/scale250_full/baseline/robustness_opt_full/robustness_stats.json)
  - [results/scale250_full/aligned5/robustness/robustness_stats.json](./results/scale250_full/aligned5/robustness/robustness_stats.json)
  - [results/scale250_full/baseline25_extension/architecture_analysis/architecture_summary.json](./results/scale250_full/baseline25_extension/architecture_analysis/architecture_summary.json)

## What Is Not In Git

- Image payloads
- Full per-model raw embedding outputs
- Large compiled `replication_results.json` artifacts for Scale250
- Local run logs, caches, smoke runs, and partial experiments
- Model weights, HF caches, and `safetensors`

Those assets are indexed in [artifacts/release_manifest.json](./artifacts/release_manifest.json). In a
local working copy, they can also be restored from `.local_artifacts/` without re-downloading.

## Start Here

1. Read the release/repro guide: [docs/release_reproducibility.md](./docs/release_reproducibility.md)
2. If you need the full dataset-build path, read [docs/dataset_reconstruction.md](./docs/dataset_reconstruction.md)
3. Create the environment from [environment.yml](./environment.yml) or [requirements.txt](./requirements.txt)
4. Run the release guardrail check:

```bash
python src/release_checks.py
```

5. If you are working from the author's cleaned local workspace and need the excluded heavy compiled artifacts locally, materialize them:

```bash
python src/materialize_release_artifacts.py --from-local-archive --all
```

6. Regenerate paper figures and renders if desired:

```bash
make paper-figures
make paper-render
```

## Repository Layout

- [data](./data): canonical analysis manifest, dataset-build inputs, and provenance metadata
- [docs](./docs): model panels, runbook, and release/repro documentation
- [manuscript](./manuscript): canonical paper plus archival material
- [results](./results): canonical paper artifacts plus archived comparison outputs
- [src](./src): core pipeline, analysis, release, and figure-generation scripts
- [artifacts](./artifacts): manifest and checksums for large excluded artifacts

## Canonical Files

- [data/data_manifest_250.json](./data/data_manifest_250.json)
- [data/concept_roster_250_scaffold.json](./data/concept_roster_250_scaffold.json)
- [data/data_manifest_250_skeleton.json](./data/data_manifest_250_skeleton.json)
- [docs/model_panel_core22.txt](./docs/model_panel_core22.txt)
- [docs/dataset_reconstruction.md](./docs/dataset_reconstruction.md)
- [docs/scale250_modular_runbook.md](./docs/scale250_modular_runbook.md)
- [src/main_replication.py](./src/main_replication.py)
- [src/compile_results.py](./src/compile_results.py)
- [src/robustness_analysis.py](./src/robustness_analysis.py)
- [src/scaling_analysis.py](./src/scaling_analysis.py)
- [src/visualize_replication_results.py](./src/visualize_replication_results.py)
- [src/analyze_arvlm_extension.py](./src/analyze_arvlm_extension.py)
- [src/generate_scale250_paper_figures.py](./src/generate_scale250_paper_figures.py)

## Archive

Earlier small-benchmark materials are retained for comparison context, but the canonical public
study in this repo is the Scale250 manuscript above.

- Archived earlier paper: [manuscript/legacy/where_representations_diverge](./manuscript/legacy/where_representations_diverge)
- Archived comparison results: [results/baseline](./results/baseline), [results/aligned5](./results/aligned5), [results/v2_change_assets](./results/v2_change_assets)

## Citation

Citation metadata is in [CITATION.cff](./CITATION.cff).
