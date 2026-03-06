# Cross-Modal Representations

Standalone backup repository for the final cross-modal representation study extracted from the
larger Platonic Convergence Lab workspace.

## What This Repo Contains

- `paper.md`: canonical manuscript source
- `paper.tex`: standalone LaTeX generated from the manuscript
- `paper_arxiv_ready.html`: print-first HTML paper for browser export to PDF
- `src/`: replication and analysis scripts
- `data/`: final concept manifest and 30-image-per-concept dataset
- `results/`: final baseline/aligned artifacts, summary reports, and representative figures
- `docs/`: run scripts and supporting experiment notes

## Canonical Artifacts

- Main paper: `paper.md`
- PDF export source: `paper_arxiv_ready.html`
- Final baseline results: `results/baseline/`
- Final aligned-layer results: `results/aligned5/`
- Summary sign-off and impact notes: `results/summaries/`

## Result Files

The two compiled replication artifacts are stored compressed to stay under GitHub file limits:

- `results/baseline/replication_results.json.gz`
- `results/aligned5/replication_results.json.gz`

To inspect them locally:

```bash
gunzip -k results/baseline/replication_results.json.gz
gunzip -k results/aligned5/replication_results.json.gz
```

## Exporting the Paper to PDF

Open `paper_arxiv_ready.html` in a browser and use Print to PDF with:

- Paper size: A4
- Margins: default / browser-managed
- Background graphics: enabled

The HTML is already styled for clean single-column manuscript export.

## Environment

Install dependencies with:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Core packages:

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

## Notes

- This repo intentionally excludes local model caches, virtual environments, logs, and intermediate
  caches from the source workspace.
- The included summary reports are historical artifacts from the final signed-off run and may still
  reference absolute source-workspace paths. The canonical paths for this repo are the ones in this
  `README`.
