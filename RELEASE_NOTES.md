# Release Notes

## v2.0.0

Scale250 canonical paper release and repository cleanup for public upload.

### Included

- Canonical Scale250 manuscript in canonical paths:
  - `manuscript/paper.md`
  - `manuscript/paper.html`
  - `manuscript/paper.tex`
- Full Scale250 figure pack under `manuscript/figures/scale250/`
- Canonical Scale250 manifest and provenance metadata
- Core extraction, compilation, robustness, scaling, and figure scripts
- Lean paper-facing result artifacts under `results/scale250_full/`
- Release artifact manifest and checksums under `artifacts/`
- Release guardrail script and CI scaffold

### Excluded From Git

- Image payloads
- Full Scale250 compiled `replication_results.json` files
- Raw per-model embedding outputs
- Logs, caches, smoke runs, and partial experiments
- Model weights and local HF caches

### Main Findings Reflected In This Release

- In supporting comparison to the earlier small benchmark, small benchmarks materially overstated
  language-image convergence on this panel.
- Within-family structure remains strong.
- Both contrastive and autoregressive VLMs are substantially closer to vision than to language in the current selected-layer analysis.
- Mid-to-late aligned layers carry stronger cross-family structure than the terminal selected layer.
- No clean language-model size law explains cross-modal alignment in this local panel.

### Release Cleanup

- Promoted the Scale250 paper to canonical `manuscript/paper.*` paths.
- Archived the earlier small-benchmark paper under `manuscript/legacy/where_representations_diverge/`.
- Moved heavy local-only data and run state into `.local_artifacts/`.
- Rebuilt `results/scale250_full/` as a lean public tree containing only small canonical paper-facing outputs.
- Added artifact indexing, local materialization support, and release guardrails.
- Added a public dataset-reconstruction path with shipped roster/skeleton inputs and a tiny runnable smoke fixture.

## v1.0.0

Initial standalone local prototype release before the Scale250 study was promoted as the canonical
public paper.
