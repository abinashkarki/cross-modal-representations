# Paper Claim Outline (Single-Run Framing)

## 1. Research Question
- What determines representational geometry across language, vision, and vision-language models: modality family, depth, or dataset composition?

## 2. Methods Contract
- 22 models: 8 language, 10 vision SSL, 4 vision-language.
- 28 concepts (20 base + 8 compounds), 30 images per concept.
- Metrics/protocol: Spearman RSA, image-bootstrap CI (300 draws, sample size 10), Mantel (3000 permutations), BH-FDR.
- Robustness controls: source holdout (LOSO + source-only), prompt sensitivity (3 templates), aligned5 depth profile.

## 3. Exact Main Claims
1. Modality asymmetry:
- Vision-VLM median rho = 0.702 vs Language-VLM median rho = 0.230 (3.06x).
2. Dataset-conditional geometry:
- LOSO ImageNet mean delta = -0.1564, max |delta| = 0.7080.
3. Depth-dependent convergence:
- Aligned means: d00 0.3424 -> d25 0.4149 -> d50 0.4345 -> d75 0.4403 -> d100 0.4639.
4. Terminal stability:
- Aligned-minus-baseline mean delta = +0.0006, mean |delta| = 0.0007.

## 4. Evidence Tables/Figures
- Figure 1 (`baseline/heatmaps/rsa_matrix.png`): Global RSA structure.
- Figure 2 (`v2_change_assets/ci_width_hist.png`): CI width distribution.
- Figure 3 (`v2_change_assets/source_holdout_loso.png`): LOSO source effects.
- Figure 4 (`v2_change_assets/prompt_sensitivity.png`): Prompt sensitivity spread.
- Figure 5 (`v2_change_assets/depth_trend.png`): Depth trend in aligned pass.
- Figure 6 (`v2_change_assets/pairwise_delta_hist.png`): Baseline vs aligned delta distribution.
- Figure 7 (`v2_change_assets/runtime_ratio.png`): Runtime/restartability context.

## 5. Findings vs Ablations Map
- Findings (main text): modality asymmetry, dataset-conditionality, depth trend, terminal stability.
- Ablations/controls (supporting): image-bootstrap uncertainty envelope, prompt sensitivity, source-only threshold skips, q4 out-of-scope statement.

## 6. Guardrails on Interpretation
- Supported: dataset-conditional measured geometry.
- Not supported: causal claim that training data dominates architecture (no controlled retraining).
