# Dataset Reconstruction Guide

This guide describes the **from-source** path for rebuilding the Scale250 dataset inputs.

It is intentionally separate from artifact-based paper reproduction.

## What This Guide Does And Does Not Promise

This repo supports a defensible public reconstruction workflow, not a byte-identical guarantee for
every image payload.

What you can reproduce:

- the 250-concept roster
- the source-balance policy
- the candidate-sourcing logic
- the CLIP-gated acceptance path
- the provenance and curation schema
- the strict source-balance and manifest validation steps

What may vary:

- exact candidate availability over time
- Unsplash results and public endpoint behavior
- external dataset hosting details
- manual review decisions if you rebuild from source rather than restoring a local archive

## Public Build Inputs Shipped In Git

- `data/concept_roster_250_scaffold.json`
- `data/data_manifest_250_skeleton.json`
- `data/scale250_seed_provenance.csv`
- `data/scale250_fresh_provenance.csv`
- `data/scale250_expanded_reserve_catalog.json`

The checked-in `data/data_manifest_250.json` is the canonical analysis manifest and references image
payloads that are excluded from git. Do not use it as the starting point for a fresh public rebuild.

## External Dependencies And Access

Install the normal environment plus curation extras:

```bash
pip install -r requirements.txt
pip install -r requirements-curation.txt
```

The sourcing path assumes:

- internet access
- ImageNet-1k validation access through the Hugging Face dataset `mrm8488/ImageNet1K-val`
- public Open Images metadata and object access
- CLIP model downloads for screening
- NLTK WordNet data

Unsplash note:

- `src/source_scale250_manifest.py` uses Unsplash public search endpoints
- `src/topup_images_to_target.py` optionally accepts `UNSPLASH_ACCESS_KEY`
- public Unsplash access may change over time, so exact rebuilds are not guaranteed from public search alone

## Quick Smoke Fixture

The public repo ships a tiny smoke fixture under `fixtures/dataset_rebuild_smoke/`.

Use it first:

```bash
python src/init_scale250_fresh_build.py \
  --roster-path fixtures/dataset_rebuild_smoke/roster.json \
  --manifest-path /tmp/scale250_fixture_manifest.json \
  --image-root fixtures/dataset_rebuild_smoke/images \
  --candidate-root /tmp/scale250_fixture_candidates \
  --provenance-path /tmp/scale250_fixture_provenance.csv \
  --inventory-output /tmp/scale250_fixture_inventory.csv \
  --tracker-output /tmp/scale250_fixture_tracker.csv \
  --force
```

```bash
python src/sync_manifest_curation.py \
  --manifest-path /tmp/scale250_fixture_manifest.json \
  --image-root fixtures/dataset_rebuild_smoke/images \
  --sync-image-paths true \
  --infer-image-sources-from-filenames true \
  --strict-image-sources true \
  --strict-source-balance true \
  --write
```

```bash
python src/preflight_replication.py \
  --phase pre \
  --manifest /tmp/scale250_fixture_manifest.json \
  --min-images-per-concept 3 \
  --require-image-source-metadata true
```

If this passes, the public build path is wired correctly on your machine.

## Full Scale250 Reconstruction

### 1. Initialize A Fresh Shadow Build

```bash
python src/init_scale250_fresh_build.py
```

This creates:

- `data/data_manifest_250_fresh.json`
- `data/scale250_fresh_provenance.csv`
- `data/scale250_fresh_curation_inventory.csv`
- `data/scale250_fresh_concept_tracker.csv`

The fresh build keeps the canonical checked-in analysis manifest untouched.

### 2. Source Candidate Images

Run the candidate sourcing step against the fresh manifest:

```bash
python src/source_scale250_manifest.py \
  --manifest-path data/data_manifest_250_fresh.json \
  --write
```

Useful narrower runs:

```bash
python src/source_scale250_manifest.py \
  --manifest-path data/data_manifest_250_fresh.json \
  --concepts cat dog horse \
  --sources imagenet openimages unsplash \
  --write
```

### 3. Sync Accepted Images Into The Manifest

```bash
python src/sync_manifest_curation.py \
  --manifest-path data/data_manifest_250_fresh.json \
  --image-root data/images_250_fresh \
  --sync-image-paths true \
  --infer-image-sources-from-filenames true \
  --write
```

### 4. Enforce Strict Quotas

```bash
python src/sync_manifest_curation.py \
  --manifest-path data/data_manifest_250_fresh.json \
  --image-root data/images_250_fresh \
  --sync-image-paths true \
  --infer-image-sources-from-filenames true \
  --strict-image-sources true \
  --strict-source-balance true
```

### 5. Run Manifest Preflight

```bash
python src/preflight_replication.py \
  --phase pre \
  --manifest data/data_manifest_250_fresh.json \
  --min-images-per-concept 15 \
  --require-image-source-metadata true \
  --require-clip-scores
```

Only after this should you treat the fresh manifest as extraction-ready.

## Moving From Dataset Build To Model Extraction

Once a fresh manifest is fully curated and passes preflight, the extraction workflow lives in:

- `docs/scale250_modular_runbook.md`
- `docs/run_modular_panel.sh`
- `docs/model_panel_core22.txt`

If you use local MLX model caches for the Qwen overrides, set:

```bash
export LOCAL_MODELS_DIR=/path/to/local/models
```
