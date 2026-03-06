# Cross-Modal Representations

Final release repository for the cross-modal representation study comparing language, vision, and
vision-language models under a robust local evaluation protocol. The repository is organized as an
evidence package first: data, scripts, results, and release summaries are primary; the manuscript
lives under `manuscript/`.

## Release Status

- Release: `v1.0.0`
- Scope: final paper, final figures, reproducible analysis scripts, curated concept dataset, and
  signed-off baseline/aligned findings
- Remote: [abinashkarki/cross-modal-representations](https://github.com/abinashkarki/cross-modal-representations)
- License: [MIT](./LICENSE)

## Main Findings

- Within-family convergence is strong for language-language, vision-vision, and
  vision-language-to-vision pairs.
- Vision-language encoders remain structurally closer to vision than to language.
- Cross-modal geometry exists but is materially weaker than within-family agreement.
- Image source, prompt template, and representational depth all change the measured geometry enough
  to matter for interpretation.

## Start Here

- Paper source: [`manuscript/paper.md`](./manuscript/paper.md)
- LaTeX source: [`manuscript/paper.tex`](./manuscript/paper.tex)
- PDF export source: [`manuscript/paper_arxiv_ready.html`](./manuscript/paper_arxiv_ready.html)
- Release notes: [`RELEASE_NOTES.md`](./RELEASE_NOTES.md)

## Repository Layout

- [`manuscript/`](./manuscript): manuscript sources and print-ready HTML
- [`src/`](./src): replication, compilation, visualization, robustness, and healthcheck scripts
- [`data/`](./data): final manifest and 30-image-per-concept dataset
- [`results/baseline/`](./results/baseline): final baseline run artifacts
- [`results/aligned5/`](./results/aligned5): final aligned-layer run artifacts
- [`results/v2_change_assets/`](./results/v2_change_assets): representative figures used in the paper
- [`results/summaries/`](./results/summaries): signed-off summary reports from the final run
- [`docs/`](./docs): run scripts and supporting notes

## Canonical Artifacts

- Main manuscript: [`manuscript/paper.md`](./manuscript/paper.md)
- Browser-to-PDF manuscript: [`manuscript/paper_arxiv_ready.html`](./manuscript/paper_arxiv_ready.html)
- Baseline robustness stats: [`results/baseline/robustness/robustness_stats.json`](./results/baseline/robustness/robustness_stats.json)
- Aligned robustness stats: [`results/aligned5/robustness/robustness_stats.json`](./results/aligned5/robustness/robustness_stats.json)
- Final signoff summary: [`results/summaries/V2_FINAL_SIGNOFF.md`](./results/summaries/V2_FINAL_SIGNOFF.md)

## Key Figures

- Global RSA structure: [`results/baseline/heatmaps/rsa_matrix.png`](./results/baseline/heatmaps/rsa_matrix.png)
- Source-holdout sensitivity: [`results/v2_change_assets/source_holdout_loso.png`](./results/v2_change_assets/source_holdout_loso.png)
- Bootstrap uncertainty: [`results/v2_change_assets/ci_width_hist.png`](./results/v2_change_assets/ci_width_hist.png)
- Prompt sensitivity: [`results/v2_change_assets/prompt_sensitivity.png`](./results/v2_change_assets/prompt_sensitivity.png)
- Depth trend: [`results/v2_change_assets/depth_trend.png`](./results/v2_change_assets/depth_trend.png)

## Large Result Files

The two compiled replication artifacts are stored compressed to stay under GitHub file limits:

- [`results/baseline/replication_results.json.gz`](./results/baseline/replication_results.json.gz)
- [`results/aligned5/replication_results.json.gz`](./results/aligned5/replication_results.json.gz)

To inspect them locally:

```bash
gunzip -k results/baseline/replication_results.json.gz
gunzip -k results/aligned5/replication_results.json.gz
```

## Exporting the Paper to PDF

Open [`manuscript/paper_arxiv_ready.html`](./manuscript/paper_arxiv_ready.html) in a browser and
use Print to PDF with:

- Paper size: A4
- Margins: default or browser-managed
- Background graphics: enabled

The HTML is already styled for clean single-column manuscript export.

## Environment

Create a local environment with:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Core dependencies:

- `torch`
- `transformers`
- `pillow`
- `numpy`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `mlx`
- `mlx-lm`

## Citation

Citation metadata is provided in [`CITATION.cff`](./CITATION.cff).

## Notes

- This standalone repo intentionally excludes local model caches, virtual environments, runtime
  logs, and intermediate cache directories from the source workspace.
- Historical summary artifacts have been path-normalized for this standalone repo, but the
  `manuscript/` folder and the `results/baseline` and `results/aligned5` folders are the primary
  references.
