# Cross-Modal Representations

Public release repository for the Scale250 cross-modal representation study and paper:
*Small Benchmarks Overstate Cross-Modal Convergence: Evidence from a 250-Concept Confirmatory Benchmark*.

This repo is intentionally lean. It keeps the manuscript, figure pack, manifests, core scripts, and
small paper-facing summary artifacts in git. Heavy compiled outputs, raw embedding dumps, caches, and
image payloads are excluded from git and described in [artifacts/release_manifest.json](./artifacts/release_manifest.json).

## Current Paper

- Manuscript source: [manuscript/paper.md](./manuscript/paper.md)
- HTML render: [manuscript/paper.html](./manuscript/paper.html)
- LaTeX export: [manuscript/paper.tex](./manuscript/paper.tex)
- Figure pack: [manuscript/figures/scale250](./manuscript/figures/scale250)

## What Is In Git

- Current Scale250 manuscript and figure pack
- Canonical manifests and provenance metadata
- Core extraction, compilation, robustness, scaling, and figure scripts
- Small canonical result artifacts for the paper:
  - [results/scale250_full/baseline/robustness_opt_full/robustness_stats.json](./results/scale250_full/baseline/robustness_opt_full/robustness_stats.json)
  - [results/scale250_full/aligned5/robustness/robustness_stats.json](./results/scale250_full/aligned5/robustness/robustness_stats.json)
  - [results/scale250_full/baseline25_extension/architecture_analysis/architecture_summary.json](./results/scale250_full/baseline25_extension/architecture_analysis/architecture_summary.json)
- Legacy comparison artifacts from the earlier local release under [results/baseline](./results/baseline) and [results/aligned5](./results/aligned5)

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
2. Create the environment from [environment.yml](./environment.yml) or [requirements.txt](./requirements.txt)
3. Run the release guardrail check:

```bash
python src/release_checks.py
```

4. If you need the excluded heavy compiled artifacts locally, materialize them:

```bash
python src/materialize_release_artifacts.py --from-local-archive --all
```

5. Regenerate paper figures and renders if desired:

```bash
make paper-figures
make paper-render
```

## Repository Layout

- [data](./data): canonical manifests and provenance metadata only
- [docs](./docs): model panels, runbook, and release/repro documentation
- [manuscript](./manuscript): current paper plus legacy manuscript archive
- [results](./results): small canonical paper artifacts and legacy comparison outputs
- [src](./src): core pipeline, analysis, release, and figure-generation scripts
- [artifacts](./artifacts): manifest and checksums for large excluded artifacts

## Canonical Current Files

- [data/data_manifest_250.json](./data/data_manifest_250.json)
- [docs/model_panel_core22.txt](./docs/model_panel_core22.txt)
- [docs/scale250_modular_runbook.md](./docs/scale250_modular_runbook.md)
- [src/main_replication.py](./src/main_replication.py)
- [src/compile_results.py](./src/compile_results.py)
- [src/robustness_analysis.py](./src/robustness_analysis.py)
- [src/scaling_analysis.py](./src/scaling_analysis.py)
- [src/visualize_replication_results.py](./src/visualize_replication_results.py)
- [src/analyze_arvlm_extension.py](./src/analyze_arvlm_extension.py)
- [src/generate_scale250_paper_figures.py](./src/generate_scale250_paper_figures.py)

## Legacy Material

The previous paper is preserved under
[manuscript/legacy/where_representations_diverge](./manuscript/legacy/where_representations_diverge).
The earlier local-release comparison results remain in:

- [results/baseline](./results/baseline)
- [results/aligned5](./results/aligned5)
- [results/v2_change_assets](./results/v2_change_assets)

## Citation

Citation metadata is in [CITATION.cff](./CITATION.cff).
