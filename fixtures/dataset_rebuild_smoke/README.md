# Dataset Rebuild Smoke Fixture

This fixture is a tiny public sanity check for the dataset-build path.

It is not part of the Scale250 benchmark itself. The included images are synthetic, tiny, and only
exist to verify that the public rebuild workflow is wired correctly.

## What It Tests

- roster-driven manifest initialization
- curator inventory generation
- manifest sync from an image root
- filename-based source inference
- strict source-balance validation
- manifest preflight on a minimal runnable fixture

## Fixture Layout

- `roster.json`: two concepts with a `1/1/1` source-balance target
- `images/`: six tiny JPEG files using the same source-encoded naming scheme as the full dataset

## Expected Smoke Commands

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
