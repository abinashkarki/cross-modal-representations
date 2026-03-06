# V2 Baseline vs Aligned5 Delta Brief

Generated: 2026-03-05T13:52:15

## Executive Summary
- Pairwise RSA drift from baseline to aligned is very small: mean delta `+0.0006`, median `+0.0000`, mean |delta| `0.0007` across `231` pairs.
- Statistical significance profile is unchanged: BH-FDR q<0.05 pairs remained `157` in both runs.
- Largest positive pair delta: `+0.020`; largest negative pair delta: `-0.005`.

## Meaningful Pairwise Deltas (|delta| >= 0.001)
Increases:
- `SmolLM3-3B-8bit` vs `ViT-MAE-base` (causal × vision): `0.430 -> 0.451` (`+0.020`)
- `Qwen3-1.7B-MLX-8bit` vs `ViT-MAE-base` (causal × vision): `0.396 -> 0.416` (`+0.019`)
- `DINOv2-base` vs `ViT-MAE-base` (vision × vision): `0.604 -> 0.621` (`+0.017`)
- `Qwen3-0.6B-MLX-8bit` vs `ViT-MAE-base` (causal × vision): `0.201 -> 0.215` (`+0.014`)
- `ViT-MAE-base` vs `data2vec-vision` (vision × vision): `0.553 -> 0.565` (`+0.012`)
- `I-JEPA` vs `ViT-MAE-base` (vision × vision): `0.817 -> 0.828` (`+0.012`)
- `DINOv2-small` vs `ViT-MAE-base` (vision × vision): `0.634 -> 0.643` (`+0.010`)
- `Granite-3.3-2B-Instruct-8bit` vs `ViT-MAE-base` (causal × vision): `0.326 -> 0.336` (`+0.009`)
Decreases:
- `MetaCLIP-B32-400m` vs `ViT-MAE-base` (vision × vision_language): `0.719 -> 0.713` (`-0.005`)
- `SigLIP` vs `ViT-MAE-base` (vision × vision_language): `0.696 -> 0.692` (`-0.004`)

## Modality-Group Drift
- `vision × vision`: n=45, mean delta `+0.0015`, mean |delta| `0.0015`
- `causal × vision`: n=80, mean delta `+0.0010`, mean |delta| `0.0010`
- `causal × vision_language`: n=32, mean delta `+0.0000`, mean |delta| `0.0000`
- `vision_language × vision_language`: n=6, mean delta `+0.0000`, mean |delta| `0.0000`
- `causal × causal`: n=28, mean delta `+0.0000`, mean |delta| `0.0000`
- `vision × vision_language`: n=40, mean delta `-0.0002`, mean |delta| `0.0003`

## Source Holdout Stability
- `LOSO` / `imagenet`: mean delta `-0.1564 -> -0.1573`, mean |delta| `0.2358 -> 0.2358`
- `LOSO` / `openimages`: mean delta `-0.0792 -> -0.0789`, mean |delta| `0.0865 -> 0.0863`
- `LOSO` / `unsplash`: mean delta `+0.0900 -> +0.0904`, mean |delta| `0.1295 -> 0.1294`
- `Source-only` / `imagenet`: mean delta `-0.0194 -> -0.0184`, mean |delta| `0.1912 -> 0.1904`
- `Source-only` / `openimages`: skipped (insufficient concepts (6 < 8))
- `Source-only` / `unsplash`: skipped (insufficient concepts (4 < 8))

## Prompt Sensitivity
- Mean `max_abs_delta_vs_baseline`: baseline `0.1960`, aligned `0.1960`.
- Max per-model shift in `max_abs_delta_vs_baseline`: `0.000000` (effectively unchanged).

## Aligned-Layer Fraction Means
- `d00`: baseline `0.4633`, aligned `0.3424`, delta `-0.1209` (n=231)
- `d25`: baseline `0.4633`, aligned `0.4149`, delta `-0.0483` (n=231)
- `d50`: baseline `0.4633`, aligned `0.4345`, delta `-0.0288` (n=231)
- `d75`: baseline `0.4633`, aligned `0.4403`, delta `-0.0230` (n=231)
- `d100`: baseline `0.4633`, aligned `0.4639`, delta `+0.0006` (n=231)

## Files
- Brief: `/Users/hi/projects/platonic_representation/experiments/01_embeddings_convergence_basics/results/replication/V2_BASELINE_VS_ALIGNED_DELTA_BRIEF.md`
- Baseline robustness: `/Users/hi/projects/platonic_representation/experiments/01_embeddings_convergence_basics/results/replication/baseline/robustness/robustness_stats.json`
- Aligned robustness: `/Users/hi/projects/platonic_representation/experiments/01_embeddings_convergence_basics/results/replication/aligned5/robustness/robustness_stats.json`
