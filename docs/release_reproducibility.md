# Release Reproducibility Guide

This repository is the public paper release, not the full working directory.

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

## Regenerating Paper Figures

Once the heavy compiled artifacts are available locally:

```bash
make paper-figures
make paper-render
```

## Main Experiment Entry Points

Core 22-model panel:

- model list: `docs/model_panel_core22.txt`
- run script: `docs/run_modular_panel.sh`
- runbook: `docs/scale250_modular_runbook.md`

Autoregressive VLM extension:

- Qwen pair: `docs/model_panel_qwen_arvlm2.txt`
- full AR-VLM panel scaffold: `docs/model_panel_arvlm4.txt`

## What To Cite / What To Use

- current manuscript: `manuscript/paper.md`
- current manifest: `data/data_manifest_250.json`
- legacy paper: `manuscript/legacy/where_representations_diverge/`
