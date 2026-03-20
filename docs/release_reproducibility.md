# Release Reproducibility Guide

This repository is the public paper release, not the full working directory.
The canonical study in this repo is the Scale250 manuscript and its supporting result surface.
Earlier small-benchmark materials remain only as archival comparison context.

## Scope

The repo includes:

- manuscript sources and figure pack
- canonical manifests and provenance metadata
- core extraction / analysis scripts
- small paper-facing result artifacts
- a manifest for excluded heavy artifacts

The repo excludes:

- image payloads
- full compiled Scale250 `replication_results.json`
- raw per-model output dumps
- caches, logs, smoke runs, and partial experiments
- model weights and HF caches

## Reproducibility Modes

### 1. Paper Reading

Use the checked-in manuscript, figures, and small result summaries:

- `manuscript/paper.md`
- `manuscript/figures/scale250/`
- `results/scale250_full/**`

### 2. Artifact-Based Analysis Reproduction

This mode reproduces the paper analyses from compiled artifacts without rebuilding the dataset from
source.

Important constraint:

- the current public release does **not** publish external artifact URLs
- `src/materialize_release_artifacts.py` therefore works only with `--from-local-archive`
- that path is useful in the author's cleaned workspace, but external users need separately hosted bundles before this mode works outside the local archive

### 3. Dataset Reconstruction From Source

This mode rebuilds the dataset inputs from the public roster, sourcing logic, and provenance rules.
Use [docs/dataset_reconstruction.md](./dataset_reconstruction.md) for that workflow.

Important scope note:

- `data/data_manifest_250.json` is the canonical analysis manifest and references image payloads excluded from git
- for a fresh rebuild, start from `data/concept_roster_250_scaffold.json` and the fresh-build path documented in `docs/dataset_reconstruction.md`

## Environment

Preferred:

```bash
conda env create -f environment.yml
conda activate cross-modal-representations
```

Alternative:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon / MLX note:

- The extraction pipeline expects Apple Silicon for the `mlx`, `mlx-lm`, and `mlx-vlm` paths.
- `pandoc` is required separately for `make paper-render`.

## Release Guardrail Check

Run this first:

```bash
python src/release_checks.py
```

It verifies that the public tree contains the required canonical files and does not accidentally track
forbidden heavy payloads.

## Restoring Excluded Heavy Artifacts

The heavy compiled Scale250 artifacts are indexed in [artifacts/release_manifest.json](../artifacts/release_manifest.json).

If you are working from this local cleaned workspace, restore them from `.local_artifacts/`:

```bash
python src/materialize_release_artifacts.py --from-local-archive --all
```

This copies the excluded files back into their canonical public checkout paths, for example:

- `results/scale250_full/baseline/replication_results.json`
- `results/scale250_full/aligned5/replication_results.json`
- `results/scale250_full/baseline25_extension/replication_results.json`

If you are not working from the local cleaned workspace, this release currently has no published
artifact URLs to download from.

## Regenerating Paper Figures

Once the heavy compiled artifacts are available locally:

```bash
make paper-figures
make paper-render
```

## Dataset Reconstruction

For the public from-source path, start with:

- `docs/dataset_reconstruction.md`
- `data/concept_roster_250_scaffold.json`
- `src/init_scale250_fresh_build.py`
- `src/source_scale250_manifest.py`
- `src/sync_manifest_curation.py`

## Experiment Extraction

These entry points are only relevant **after** you have either restored the heavy artifacts locally
or reconstructed a runnable dataset with image payloads present.

Core 22-model panel:

- model list: `docs/model_panel_core22.txt`
- run script: `docs/run_modular_panel.sh`
- full workflow: `docs/scale250_modular_runbook.md`

Autoregressive VLM extension:

- Qwen pair: `docs/model_panel_qwen_arvlm2.txt`
- full AR-VLM panel scaffold: `docs/model_panel_arvlm4.txt`

Optional local model overrides:

- set `LOCAL_MODELS_DIR=/path/to/local/models` to point MLX Qwen overrides at a local cache
- if `LOCAL_MODELS_DIR` is unset, the code falls back to in-repo `models/` when present and then to the legacy workspace-level `models/` path

## What To Cite / What To Use

- canonical manuscript: `manuscript/paper.md`
- canonical manifest: `data/data_manifest_250.json`
- archival earlier paper: `manuscript/legacy/where_representations_diverge/`
