# Data Layout

This public repo keeps metadata, not image payloads.

## Canonical Files

- `data_manifest_250.json`: current Scale250 confirmatory manifest
- `data_manifest_multi.json`: legacy manifest from the earlier local release
- `scale250_seed_provenance.csv`: provenance for the seeded Scale250 build
- `scale250_fresh_provenance.csv`: provenance for fresh additions/replacements
- `scale250_expanded_reserve_catalog.json`: reserve concept metadata

## Excluded From Git

- `images_multi/`
- `images_250/`
- `images_250_fresh/`
- candidate image payload directories

Those live only in local working state and are intentionally moved under `.local_artifacts/data/`
for this release.
