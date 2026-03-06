# V2 Final Sign-Off (30 Images/Concept, Core 22)

Generated at: 2026-03-05T13:47:51

## 1) Environment Info
- Platform: macOS-26.0.1-arm64-arm-64bit
- Python: 3.11.5
- torch: 2.10.0
- transformers: 5.2.0
- mlx: 0.31.0

## 2) Manifest + Data Contract Summary
- Manifest path: `data/data_manifest_multi.json`
- Concepts: 28
- images_per_concept_target: 30
- Image counts per concept: min=30, max=30
- CLIP coverage (from `concept_metadata[concept].clip_scores`): full=28, partial=0, missing=0
- Source distribution:
- imagenet: 10 concepts
- openimages: 6 concepts
- unsplash: 12 concepts

## 3) Baseline Pass (selected / -1)
- Profile root: `results/baseline`
- Extraction log: `results/baseline/logs/pipeline_baseline_20260305_123853.log`
- Final strict compile artifact timestamp: 2026-03-05T12:56:49
- Final robustness artifact timestamp: 2026-03-05T13:09:02
- Command used:
```bash
HF_HUB_ENABLE_HF_TRANSFER=0 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTHON_BIN=<local-python-path> \
RUN_PROFILE=baseline \
LOCAL_FILES_ONLY=false \
RUN_ROBUSTNESS=1 \
FORCE_CACHE_REBUILD=true \
CACHE_IMAGE_EMBEDDINGS=true \
MODEL_TIMEOUT=2400 \
./run_all_models.sh
```

## 4) Aligned5 Pass (full 22-model sweep)
- Profile root: `results/aligned5`
- Full sweep log: `results/aligned5/logs/pipeline_aligned5_20260305_130935.log`
- Status line: `Done.` present in pipeline log.
- Final strict compile artifact timestamp: 2026-03-05T13:25:06
- Final robustness artifact timestamp: 2026-03-05T13:35:40
- Command used:
```bash
HF_HUB_ENABLE_HF_TRANSFER=0 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTHON_BIN=<local-python-path> \
RUN_PROFILE=aligned5 \
LOCAL_FILES_ONLY=false \
RUN_ROBUSTNESS=1 \
FORCE_CACHE_REBUILD=true \
CACHE_IMAGE_EMBEDDINGS=true \
MODEL_TIMEOUT=2400 \
./run_all_models.sh
```

## 5) Artifact Paths
### Baseline
- Compiled: `results/baseline/replication_results.json.gz`
- Heatmap: `results/baseline/heatmaps/rsa_matrix.png`
- Compositionality: `results/baseline/compositionality/all_models_compositionality.png`
- Scaling: `results/baseline/scaling/scaling_analysis.png`
- Robustness JSON: `results/baseline/robustness/robustness_stats.json`

### Aligned5
- Compiled: `results/aligned5/replication_results.json.gz`
- Heatmap: `results/aligned5/heatmaps/rsa_matrix.png`
- Compositionality: `results/aligned5/compositionality/all_models_compositionality.png`
- Scaling: `results/aligned5/scaling/scaling_analysis.png`
- Robustness JSON: `results/aligned5/robustness/robustness_stats.json`

## 6) Determinism and Integrity Checks
- Post-preflight (`--require-cache`) passed for baseline and aligned5.
- Robustness determinism check (seed=42 rerun): identical after excluding metadata timestamp/output_dir.
- Robustness variability check (seed=43): stochastic outputs changed as expected.

## 7) Skips / Failures
- Skipped items: none.
- Failed models: none.
- Timed out models: none.
