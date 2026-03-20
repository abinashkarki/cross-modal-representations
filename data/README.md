# Data Layout

This public repo keeps metadata, manifests, and provenance, not image payloads.

## Canonical Files

- `data_manifest_250.json`: canonical analysis manifest for the Scale250 study; references image payloads excluded from git
- `concept_roster_250_scaffold.json`: public a priori concept roster used to build the 250-concept benchmark
- `data_manifest_250_skeleton.json`: public empty-image skeleton generated from the shipped roster
- `data_manifest_multi.json`: legacy manifest from the earlier local release
- `scale250_seed_provenance.csv`: provenance for the seeded Scale250 build
- `scale250_fresh_provenance.csv`: provenance for fresh additions/replacements
- `scale250_expanded_reserve_catalog.json`: reserve concept metadata

## Recommended Public Starting Points

- For paper-facing analysis, use `data_manifest_250.json`.
- For a fresh dataset rebuild, start from `concept_roster_250_scaffold.json` and `src/init_scale250_fresh_build.py`.
- For generic roster-to-manifest generation, use `src/generate_manifest_from_roster.py`.

## Excluded From Git

- `images_multi/`
- `images_250/`
- `images_250_fresh/`
- candidate image payload directories

Those live only in local working state and are intentionally moved under `.local_artifacts/data/`
for this release.
