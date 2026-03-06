# V2 Change Impact Report (vs V1)

Generated: 2026-03-05

## Comparison Scope
- V1 reference (original experiment): `legacy V1 artifact (not included in this standalone repo)` (18 models).
- V2 final baseline: `results/baseline/replication_results.json.gz` (22 models).
- V2 final aligned pass: `results/aligned5/replication_results.json.gz` (22 models).
- Robustness artifacts used:
  - baseline: `results/baseline/robustness/robustness_stats.json`
  - aligned5: `results/aligned5/robustness/robustness_stats.json`

## Executive Call
- Most impactful scientifically: `30 images/concept`, `source holdout`, `aligned-layer protocol`.
- Most impactful operationally: `per-image cache`.
- Most impactful for statistical credibility: `image-bootstrap CI + Mantel/FDR`.
- Still missing for completion against your original list: `q4 quantization sensitivity sweep`.

## Change-by-Change Impact

### 1) Increase to 30 images/concept
Status: **Done**

Observed impact:
- On overlapping V1/V2 models (10 shared models), same-model representational continuity is moderate rather than near-identical:
  - mean Spearman continuity: `0.839`
  - min/max: `0.595` / `0.954`
- On shared-model inter-model RSA pairs (45 pairs), V1->V2 shift is substantial:
  - mean delta: `+0.104`
  - mean absolute delta: `0.115`
  - largest increase: `+0.284` (`I-JEPA` vs `ViT-MAE-base`)

Runtime/cost impact:
- Vision model extraction cost is close to the predicted ~3x increase:
  - mean V2/V1 runtime ratio on shared vision models: `2.90x`
  - median ratio: `3.09x`
  - examples:
    - `DINOv2-small`: `7.4s -> 25s` (`3.36x`)
    - `DINOv2-base`: `14.5s -> 45s` (`3.09x`)

Worth it?
- **Yes (high)**. It materially changes conclusions on model relationships and improves image sampling robustness.

Paper placement:
- **Findings** (primary).
- Also reference in **Methods** (dataset density increase).

### 2) Per-image embedding cache persisted for resampling
Status: **Done**

Observed impact:
- Enables exact image-level resampling and source-split analysis without re-running forward passes.
- Strong restartability behavior (post-preflight `--require-cache` passes on both profiles).

Runtime/cost impact:
- Major rerun speedup demonstrated on cached vision model:
  - `DINOv2-small` cached rerun: `~5.0s`
  - same model fresh extraction in sweep: `25s`
  - approx speedup: `~5x`

Worth it?
- **Yes (very high operational ROI)**.

Paper placement:
- **Methods/Engineering**.
- Not a scientific finding by itself, but essential for reproducibility and robust diagnostics.

### 3) Image-bootstrap RSA CIs (resample image subsets)
Status: **Done (exact method now implemented)**

Observed impact:
- CI method in metadata: `image_bootstrap_percentile`.
- Configuration: `300 draws`, `sample_size=10`, `replacement=true`.
- Pair behavior is correct:
  - language-language pairs deterministic with zero-width CI (28 pairs).
  - image-involved pairs have non-zero uncertainty (203 pairs).
- CI width summary (image-involved pairs):
  - mean width: `0.2034`
  - median width: `0.1980`
  - max width: `0.3703`

Worth it?
- **Yes (high statistical value)**. This is the core credibility upgrade over the old non-image bootstrap behavior.

Paper placement:
- **Methods** (primary statistical protocol).
- Include key CI-width summary in **Findings** appendix or robustness subsection.

### 4) Source holdout split by image source
Status: **Done**

Observed impact (baseline):
- LOSO (leave-one-source-out) shows meaningful source dependence:
  - exclude `imagenet`: mean delta `-0.1564`, mean |delta| `0.2358`, max |delta| `0.7080`
  - exclude `openimages`: mean delta `-0.0792`
  - exclude `unsplash`: mean delta `+0.0900`
- Source-only:
  - `imagenet` analyzable (mean delta `-0.0194`, mean |delta| `0.1912`)
  - `openimages` and `unsplash` skipped by threshold (`<8` concepts), as designed.

Worth it?
- **Yes (high)**. This exposed dataset-source sensitivity that would otherwise be hidden.

Paper placement:
- **Findings** (robustness/bias finding).
- Also include skip-threshold caveat in **Limitations**.

### 5) Prompt sensitivity with 3 templates
Status: **Done** (`baseline3`)

Observed impact (8 language models):
- Mean `max_abs_delta_vs_baseline`: `0.1960`
- Range: `0.1446` to `0.2785`
- Largest sensitivity: `Qwen3-0.6B-MLX-8bit` (`0.2785`)
- This is non-trivial variance; template choice can alter cross-modal comparisons.

Worth it?
- **Yes (moderate-high)**. Strong control variable; improves interpretation confidence.

Paper placement:
- **Ablations/Controls** (primary).
- Mention in Findings only briefly as “language template variance is non-negligible.”

### 6) Dedicated aligned-layer protocol (aligned5)
Status: **Done** (full 22-model pass)

Observed impact:
- Depth trend is strong in aligned run:
  - mean pairwise RSA by depth bin:
    - `d00`: `0.3424`
    - `d25`: `0.4149`
    - `d50`: `0.4345`
    - `d75`: `0.4403`
    - `d100`: `0.4639`
  - clear increase toward deeper layers (`+0.1215` from d00 to d100).
- Baseline-vs-aligned final outputs are nearly unchanged at terminal depth:
  - mean pairwise delta (aligned-baseline): `+0.0006`
  - significance profile unchanged (`157` BH-FDR significant pairs both).

Runtime/cost impact:
- Full extraction-stage cost remained similar to baseline in this implementation:
  - baseline extraction sum: `916s`
  - aligned extraction sum: `926s`
  - delta: `+1.1%`
- This is much better than naive 1.5x–2x expectations because multi-layer outputs are captured in one pass per forward call.

Worth it?
- **Yes (high scientific value, low additional extraction cost in current implementation)**.

Paper placement:
- **Findings** (depth-dependent convergence pattern).
- Also in **Ablations** as a structured protocol variant.

### 7) q4 quantization sensitivity
Status: **Not done (excluded from final sign-off gate)**

Observed impact:
- Not measured in final 22-model completion package.

Worth it?
- **Still worth doing** as a targeted ablation if quantization robustness is part of claims.

Paper placement:
- **Ablations** (future/remaining).

## What Goes in Findings vs Ablations

### Findings (main text)
- 30-image expansion changes inter-model RSA materially on overlapping models.
- Source holdout reveals strong source-dependent shifts (especially Imagenet exclusion).
- Layer depth strongly modulates convergence (d00->d100 monotonic increase).
- Terminal aligned vs baseline similarity is effectively unchanged (stability at end depth).

### Ablations / Controls
- Prompt template sensitivity (3-template variance, per-model spread).
- Quantization sensitivity (q8 vs q4) once run.
- Optional bootstrap parameter sensitivity (draws/sample-size stress test).
- Source-only split limitations due concept-count thresholds.

## Recommendation
- Lock current V2 core findings as robust.
- Add one focused q4 ablation sweep as the remaining high-value gap.
