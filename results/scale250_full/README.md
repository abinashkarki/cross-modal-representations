# Scale250 Result Surface

This directory is the lean public result surface for the canonical Scale250 paper.

## Included In Git

- baseline robustness stats and summary plot
- aligned5 robustness stats and summary plot
- baseline and aligned5 paper-facing RSA/scaling figures
- autoregressive VLM architecture-summary JSON and figures
- healthcheck probe output

## Excluded From Git

- compiled `replication_results.json`
- raw per-model outputs
- logs and caches
- smoke runs and partial experiments

Use `artifacts/release_manifest.json` plus `src/materialize_release_artifacts.py` if you need the
heavy compiled artifacts restored into their canonical paths locally. In the current public release,
that restore path is local-archive only until external artifact URLs are published.
