# Release Notes

## v1.0.0

Initial public backup release of the final cross-modal representation study.

### Included

- Final manuscript in `manuscript/` as Markdown, LaTeX, and print-ready HTML
- Curated 30-image-per-concept dataset and manifest
- Reproducible analysis scripts
- Final baseline and aligned-layer result artifacts
- Signed-off summary reports and representative paper figures

### Main Results

- Language-language median RSA: `0.661`
- Vision-vision median RSA: `0.682`
- Vision-to-vision-language median RSA: `0.702`
- Language-to-vision median RSA: `0.259`
- Language-to-vision-language median RSA: `0.230`
- LOSO ImageNet mean delta: `-0.1564`
- Prompt sensitivity mean max-absolute delta: `0.1960`
- Aligned-layer mean RSA trend: `0.3424 -> 0.4149 -> 0.4345 -> 0.4403 -> 0.4639`

### Release Cleanup

- Extracted the study into a standalone repository outside the larger lab workspace
- Removed virtual environments, local model caches, runtime logs, and intermediate cache folders
- Compressed oversized compiled result JSON artifacts for GitHub compatibility
- Normalized historical absolute paths inside summary and metadata artifacts
- Added a release-oriented README and a print-ready HTML manuscript

### Canonical Files

- `manuscript/paper.md`
- `manuscript/paper.tex`
- `manuscript/paper_arxiv_ready.html`
- `results/baseline/`
- `results/aligned5/`
- `results/summaries/V2_FINAL_SIGNOFF.md`
