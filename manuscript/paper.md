---
title: "Small Benchmarks Overstate Cross-Modal Convergence: Evidence from a 250-Concept Confirmatory Benchmark"
author: "Abinash Karki (Independent Research)"
date: "March 2026"
---

---

## Abstract

The Platonic Representation Hypothesis (PRH) proposes that sufficiently capable models trained on
different modalities converge toward a shared representational geometry. This paper studies that
claim using Scale250, a 250-concept benchmark evaluated on a fixed 22-model core panel: 8 language
models, 10 self-supervised vision models, and 4 contrastive vision-language encoders evaluated
through their image towers. Each concept contains 15 images with strict within-concept source
balance (5 ImageNet, 5 Open Images, 5 Unsplash). The analysis combines a primary selected-layer
evaluation, an aligned five-layer depth analysis, image bootstrap confidence intervals, Mantel
permutation tests with Benjamini-Hochberg FDR correction, prompt-sensitivity analysis, and metric
triangulation with linear Centered Kernel Alignment (CKA). To test whether the bridge-model
interpretation depends on contrastive dual encoders alone, the paper also adds a secondary 25-model
architecture extension with three autoregressive VLMs: `Qwen3.5-2B`, `Qwen3.5-4B`, and
`Phi-3.5-vision`. The main result is a strongly family-structured geometry rather than broad
language-image convergence. In the Scale250 baseline, language-language RSA remains strong (mean
0.6143) and image-side agreement remains substantial (mean 0.4681), while language-image RSA is
weak (mean 0.0199). The contrastive vision-language encoders cluster strongly with the image side
(vision-VLM mean RSA 0.5003; VLM-VLM mean RSA 0.8938) and only weakly with language models
(language-VLM mean RSA 0.0308). The autoregressive VLMs show the same pattern even more clearly:
mean RSA to vision is 0.5122 versus 0.0404 to language, with a vision-minus-language gap of
0.4718. Aligned-layer analysis shows that mean pairwise RSA peaks in mid-to-late layers (0.3096 at
`d75`) rather than the terminal selected layer (0.2683), but the headline selected-layer
conclusion is unchanged. A scaling analysis across 0.6B-4B language models shows no monotonic size
law for cross-modal alignment (Spearman size vs. mean vision RSA = -0.0952). CKA preserves the
same qualitative ordering of family structure and agrees with pairwise RSA at Pearson 0.7229 and
Spearman 0.7050. A supporting comparison to an earlier 20-concept local benchmark indicates that
small benchmarks can materially overstate cross-modal agreement. What survives the stronger
measurement regime is robust within-family structure, vision-anchored bridge models across both
contrastive and autoregressive VLMs, and a mid-layer depth effect rather than broad language-image
convergence.

**Keywords**: Platonic Representation Hypothesis, representational similarity analysis, centered
kernel alignment, cross-modal convergence, benchmark design, layer alignment

---

## 1. Introduction

### 1.1 The Question Behind the Hypothesis

The Platonic Representation Hypothesis (PRH) asks whether models trained on different modalities
converge toward a shared internal geometry because they are all modeling the same world. In its
strong form, the hypothesis predicts that modality should matter less and less as capability grows: a
language model and a vision model, trained on different data with different objectives, should still
come to organize concepts similarly.

That is a stronger claim than task transfer or interface compatibility. It is a claim about
inevitability in representation itself. The relevant question is therefore not whether some
cross-modal signal can be detected on a favorable benchmark, but whether that signal survives a
harder and more balanced measurement regime.

### 1.2 Why Measurement Design Matters

The central risk in this literature is not only false positives. It is effect-size inflation from
narrow, source-skewed, or otherwise convenient benchmarks. A study can detect genuine cross-modal
signal while still overstating how broad or durable that signal is. The relevant question is
therefore not whether some language-image agreement can be found, but how much agreement survives
once the benchmark is broadened, balanced, and analyzed under a clearly confirmatory frame.

This is where the present study is deliberately stricter. The model roster is held fixed in the core
analysis so that the experimental work is done by measurement design rather than by changing the
panel. The concept benchmark expands to 250 base concepts, compounds are removed from the main
confirmatory claim, source balance is enforced within concept, and depth, robustness, and
metric-triangulation analyses are specified as part of the study rather than as an afterthought.

### 1.3 Scale250 Study Design

The paper is organized around one primary benchmark and one secondary scope test.

The primary benchmark is Scale250: a 250-concept, source-balanced study evaluated on a fixed
22-model core panel consisting of 8 language models, 10 self-supervised vision models, and 4
contrastive vision-language encoders. The primary evidence package combines a selected-layer
analysis, an aligned five-layer depth profile, image bootstrap confidence intervals, Mantel
permutation tests with FDR correction, prompt-sensitivity analysis, and CKA triangulation.

The secondary scope test is a 25-model architecture extension that adds three autoregressive VLMs.
Its role is narrow but important: it asks whether the bridge-model interpretation that emerges in the
core panel is specific to contrastive image-language encoders or whether it persists when native
autoregressive multimodal models are added.

### 1.4 Contributions

The study makes five contributions:

1. It provides a larger and more balanced measurement of cross-modal geometry on a fixed local
   22-model panel.
2. It characterizes family structure directly, separating within-family reproducibility from
   cross-modal agreement.
3. It tests whether multimodal bridge models behave as modality-neutral intermediates or as
   vision-anchored models across both contrastive and autoregressive architectures.
4. It asks whether shared structure is more visible in aligned mid-to-late layers than in selected
   final-layer readouts.
5. It uses metric triangulation and a supporting historical comparison to clarify which conclusions
   are specific to a stronger benchmark and which are robust to measurement choice.

### 1.5 Research Questions

The paper is organized around six questions:

- **RQ1**: On the 250-concept benchmark, how much geometry is shared within and across modality
  families?
- **RQ2**: Do contrastive VLM image towers and autoregressive VLMs behave as modality-neutral
  bridges, or do they remain more closely tied to the vision family?
- **RQ3**: Does aligned-layer analysis reveal stronger shared structure than selected-layer
  evaluation?
- **RQ4**: Do the main conclusions survive an alternate representational similarity metric?
- **RQ5**: Does language-model size predict stronger vision alignment in this panel?
- **RQ6**: How does the Scale250 benchmark compare with the earlier small local benchmark when the
  core 22-model roster is held fixed?

---

## 2. Related Work

The closest conceptual antecedent is the Platonic Representation Hypothesis of Huh et al. (2024),
which argues that sufficiently capable language and vision models exhibit increasing geometric
agreement across modalities. That work is valuable because it turns a philosophical intuition into an
empirical one: convergence becomes a measurement problem rather than a slogan. The present paper
addresses a narrower question. Instead of asking whether convergence can be shown on large curated
setups, it asks how much of the effect survives when a small local benchmark is replaced by a much
broader confirmatory one.

Our measurement frame sits inside the broader literature on representational similarity. RSA
(Kriegeskorte et al., 2008) compares the relational geometry of concept representations rather than
their raw coordinates, which makes it especially useful when embedding dimensions differ across model
families. But RSA is not the only lens. Canonical-correlation approaches such as SVCCA (Raghu et al.,
2017) and PWCCA-style analyses (Morcos et al., 2018) ask whether two representations span similar
subspaces even when their coordinates are not directly aligned. CKA (Kornblith et al., 2019) adds a
kernel-based alternative that is widely used because it is invariant to isotropic scaling and robust
to differences in representation width. Reviewers are right to be skeptical of metric monoculture. A
claim that rests only on Spearman RSA over cosine-similarity matrices is a weaker claim than one
that survives more than one representational comparison family.

This paper therefore treats metric triangulation as part of the contribution rather than an optional
appendix. RSA remains the main inferential target because the research question is about shared
concept geometry, but linear CKA is used as a second lens over the same concept-by-feature matrices.
The point is not to claim that the metrics are interchangeable. They are not. RSA compares
rank-ordered similarity structure in representational dissimilarity matrices; CKA compares centered
alignment between feature spaces. Agreement between them therefore strengthens trust in the direction
of the result, not in any single numeric magnitude.

The bridge-model question also sits in a specific multimodal literature. Contrastive image-language
models such as CLIP (Radford et al., 2021) and SigLIP (Zhai et al., 2023) are designed to make
language and image embeddings interoperable. But interoperable interfaces do not automatically imply
modality-neutral internals. A dual-encoder can align tasks while still preserving strong
vision-anchored structure in its image tower. This distinction is central to the present paper
because the four vision-language models in the current panel are all contrastive image-language
encoders evaluated through their image-side representations.

Finally, this paper contributes to a quieter but important tradition of benchmark design and
measurement clarification. Weaker-than-expected results matter when they arise from a stronger
measurement regime rather than from degraded engineering. Here the point is not that cross-modal
structure disappears. It is that its estimated strength depends materially on benchmark
construction. That is scientifically useful even if it is less glamorous than a broad positive
claim.

---

## 3. Methods

### 3.1 Confirmatory Design Overview

The full study is built around a simple contract: keep the model panel fixed, scale the concept
benchmark aggressively, and separate the primary confirmatory benchmark from supporting analyses.

**Table 1: Scale250 study design.**

| Component | Current experiment |
|-----------|--------------------|
| Core model panel | 22 models |
| Model families | 8 language, 10 vision SSL, 4 vision-language |
| Secondary architecture extension | +3 autoregressive VLMs (25 models complete overall) |
| Primary concept set | 250 base concepts |
| Concept design | stratified a priori, 10 strata x 25 concepts |
| Compound concepts in primary set | none |
| Images per concept | 15 |
| Per-concept source balance | 5 ImageNet, 5 Open Images, 5 Unsplash |
| Text templates extracted | `baseline3` (`t0`, `t1`, `t2`) |
| Main layer protocols | selected-layer baseline and `aligned5` |
| Bootstrap | 300 draws, 10 images per concept, with replacement |
| Significance test | 3,000-permutation Mantel, BH FDR |
| Secondary metric | linear CKA on concept-by-feature matrices |

![Figure 1. Scale250 study design. Left: the primary benchmark consists of 250 stratified base concepts and excludes compounds from the main confirmatory claim. Center: every concept is balanced within concept across ImageNet, Open Images, and Unsplash at 5/5/5 images. Right: the model panel comprises 8 language, 10 vision SSL, and 4 vision-language encoders.](figures/scale250/design_overview.png)

### 3.2 Model Panel and Architecture Extension

The core 22-model roster is fixed throughout the main study. This is deliberate. Holding the panel
constant lets benchmark design and analysis protocol do the experimental work.

**Table 2: Model roster.**

| Family | Models |
|-------|--------|
| Language (8) | Falcon3-1B-Instruct-8bit, Granite-3.3-2B-Instruct-8bit, LFM2-2.6B-Exp-8bit, Qwen2.5-1.5B-Instruct-8bit, Qwen3-0.6B-MLX-8bit, Qwen3-1.7B-MLX-8bit, Qwen3-4B-MLX-8bit, SmolLM3-3B-8bit |
| Vision SSL (10) | BEiT-base, ConvNeXt-v2, DINOv2-base, DINOv2-small, DINOv3-ConvNeXt-tiny, Hiera-base, I-JEPA, ViT-MAE-base, ViT-MSN-base, data2vec-vision |
| Vision-Language (4) | CLIP-ViT-B32, MetaCLIP-B32-400m, SigLIP, SigLIP2 |

The language models span roughly 0.6B-4B parameters. The vision side deliberately spans multiple
self-supervised paradigms. The vision-language models require a specific caveat: the current panel
contains **contrastive dual-encoder models**, and the analysis uses their **image towers**. Any
claim about bridge-model behavior in the core 22-model analysis is therefore scoped to contrastive
image-encoder-side representations, not to autoregressive interleaved VLMs in general.

To answer that scope critique directly, the paper adds a secondary architecture extension on the same
250-concept benchmark with three completed autoregressive VLMs: `Qwen3.5-2B-Base`,
`Qwen3.5-4B-Base`, and `Phi-3.5-vision-instruct`. These models are not folded into the aligned5
analysis; they are used as a selected-layer architecture stress test against the existing bridge-model
story. The comparison therefore asks a narrow question: when we add native autoregressive multimodal
models, do they sit closer to language, closer to vision, or in between?

### 3.3 Benchmark Construction

The new benchmark is not a larger random sample. It is a structured benchmark. The manifest in
`data/data_manifest_250.json` specifies:

- a **stratified a priori** primary set (`base250`)
- **10 semantic strata** with **25 concepts each**
- **5 reserve concepts per stratum**
- **no compounds in the primary confirmatory set**
- **drop-concept-if-unbalanced** enforcement

The 10 target strata are:

- animals
- plants and fungi
- food and drink
- clothing and accessories
- tools and household objects
- furniture, appliances, and containers
- vehicles and machines
- buildings and infrastructure
- natural landforms and waterscapes
- musical instruments

Each concept has exactly 15 accepted images. The source policy is within-concept balanced:
5 ImageNet, 5 Open Images, and 5 Unsplash images per concept, with no substitution allowed. That is
a major improvement over the earlier project state because source diversity is now built into the
benchmark rather than treated mainly as a post hoc nuisance variable.

### 3.4 Embedding Extraction

Language-side concept embeddings are extracted from short definitional prompts using the three
`baseline3` templates:

- `The concept of {concept}`
- `An example of {concept}`
- `The meaning of {concept}`

The selected-layer baseline uses the model's default terminal representation. The aligned-layer pass
uses five standardized depth fractions:

- `d00`
- `d25`
- `d50`
- `d75`
- `d100`

For image-side models, per-image embeddings are extracted first and aggregated to concept-level
representations. Image-level embeddings are cached, which allows image bootstrap and layer analysis
without rerunning the forward passes.

For the autoregressive VLM extension, image-conditioned representations are extracted through the
MLX-VLM `get_input_embeddings(...)` path after multimodal processor preparation. The representation
used for analysis is the final inserted image-token embedding aggregated over the image tokens for a
single image, then averaged to the concept level in the same way as the other image-side models. This
path provides one comparable selected-layer representation per image but does not yet expose a stable
cross-family internal layer stack. The autoregressive VLM extension is therefore selected-layer only.

### 3.5 RSA and Statistical Inference

For each model:

1. construct a concept-by-concept cosine similarity matrix,
2. extract the upper triangle,
3. compare model pairs with Spearman RSA.

The primary inferential package has two parts:

- **image bootstrap**: 300 draws, 10 images per concept, sampling with replacement
- **Mantel significance**: 3,000 permutations per model pair, two-sided p-values, BH FDR over 231
  pairs

We also compute prompt sensitivity from the three extracted templates and run the same robustness
package on the aligned5 pass.

### 3.6 CKA Triangulation

RSA is the main analysis because the paper is about shared concept geometry, but we add a second
metric to test whether the qualitative ordering survives outside the RSA family. For the selected
layer baseline, we build one concept-by-feature matrix per model using the same 250 concept
representations used in the RSA analysis. Each matrix is row-normalized across concepts before
comparison, and we compute pairwise **linear CKA** between all model pairs.

This CKA analysis is intentionally used as triangulation rather than as a second significance layer.
The inferential package remains tied to RSA, bootstrap, and Mantel tests. The CKA question is
qualitative: does the same family structure appear when we compare centered feature spaces rather
than ranked similarity matrices?

### 3.7 Supporting Historical Comparison to the Earlier Small Benchmark

For interpretive context, a later results section compares Scale250 with an earlier local benchmark
that used the same 22-model roster but only 20 base concepts for RSA plus 8 compound probes for
exploratory analysis. The purpose of that comparison is secondary rather than primary. It is used to
show how benchmark scope changes the estimated cross-modal effect size while holding the core panel
fixed.

---

## 4. Results

This section presents the canonical Scale250 findings first and uses comparison to the earlier
small-benchmark study as supporting context rather than as the main analytical frame.

### 4.1 Family Structure in the Scale250 Baseline

The baseline run in `results/scale250_full/baseline/replication_results.json` and
`results/scale250_full/baseline/robustness_opt_full/robustness_stats.json` shows clear family
structure.

**Table 3: Fine-grained pairwise RSA summary for the Scale250 baseline.**

| Pair type | Pairs | Mean $\rho$ | Median $\rho$ | Significant after FDR |
|----------|------:|------------:|--------------:|----------------------:|
| Language-Language | 28 | 0.6143 | 0.5994 | 28 / 28 |
| Vision-Vision | 45 | 0.3827 | 0.4248 | 43 / 45 |
| VLM-VLM | 6 | 0.8938 | 0.8921 | 6 / 6 |
| Vision-VLM | 40 | 0.5003 | 0.5584 | 40 / 40 |
| Language-Vision | 80 | 0.0155 | 0.0149 | 15 / 80 |
| Language-VLM | 32 | 0.0308 | 0.0372 | 6 / 32 |

![Figure 2. Baseline RSA heatmap for the 250-concept benchmark. The family block structure is visible directly: language models cluster together, CLIP-family encoders cluster tightly, and the strongest bridge-model affinities are on the image side rather than between language and image.](figures/scale250/baseline_rsa_heatmap.png)

Three conclusions follow immediately.

First, language models agree strongly with one another. The mean language-language RSA is 0.6143 and
all 28 pairs survive FDR correction.

Second, the contrastive vision-language encoders do not behave like modality-neutral bridges. Their
image towers agree extremely strongly with one another (mean 0.8938) and strongly with the pure
vision models (mean 0.5003), but only weakly with language models (mean 0.0308). On the current
evidence, these bridge models are better interpreted as **vision-anchored interfaces** than as proof
of a shared cross-modal geometry.

Third, strict language-vision agreement is weak. The mean is 0.0155, the median is 0.0149, and only
15 of 80 pairs survive FDR correction. The strongest positive pairs in the entire panel are all on
the image side:

- `SigLIP` vs `SigLIP2`: 0.9397
- `CLIP-ViT-B32` vs `MetaCLIP-B32-400m`: 0.9306
- `MetaCLIP-B32-400m` vs `SigLIP2`: 0.9100

The most negative pairs are all cross-modal and modest in magnitude:

- `DINOv2-base` vs `Granite-3.3-2B-Instruct-8bit`: -0.1271
- `Granite-3.3-2B-Instruct-8bit` vs `SigLIP2`: -0.0941
- `DINOv2-base` vs `LFM2-2.6B-Exp-8bit`: -0.0927

### 4.2 Autoregressive VLMs Preserve the Same Bridge-Model Story

The main architecture-scope question is whether the bridge-model interpretation only holds for
contrastive dual encoders such as CLIP and SigLIP. To test that, the paper adds three
autoregressive VLMs on the same 250-concept benchmark and merges them with the 22-model baseline
into a 25-model selected-layer panel.

**Table 4: Bridge-model family summary after adding three autoregressive VLMs.**

| Bridge family | Mean to language | Mean to vision | Mean within family | Mean to contrastive VLM | Vision minus language |
|--------------|-----------------:|---------------:|-------------------:|------------------------:|----------------------:|
| Contrastive VLM (4) | 0.0308 | 0.5003 | 0.8938 | - | 0.4695 |
| Autoregressive VLM (3) | 0.0404 | 0.5122 | 0.9011 | 0.7202 | 0.4718 |

![Figure 3. Bridge-model comparison after adding three autoregressive VLMs. Every bridge model is much closer to the vision family than to the language family. The new autoregressive VLMs do not collapse toward the language block; they preserve the same vision-anchored pattern as the contrastive encoders.](figures/scale250/arvlm_bridge_language_vs_vision.png)

![Figure 4. Family-block mean RSA in the 25-model extension. The strongest off-diagonal structure remains on the image side: contrastive VLMs and autoregressive VLMs are both much closer to vision than to language, and the two bridge families are strongly aligned with one another.](figures/scale250/arvlm_family_block_mean_rsa.png)

This extension does not rescue a modality-neutral bridge interpretation. It strengthens the opposite
one. The three autoregressive VLMs have mean RSA 0.5122 to the vision family and only 0.0404 to the
language family, which is effectively the same vision-language gap seen in the contrastive VLMs
(0.4718 versus 0.4695). The per-model pattern is also consistent:

- `Qwen3.5-2B-Base-MLX-8bit`: 0.5357 to vision, 0.0399 to language
- `Qwen3.5-4B-Base-MLX-8bit`: 0.5272 to vision, 0.0410 to language
- `Phi-3.5-vision-instruct-MLX-8bit`: 0.4737 to vision, 0.0403 to language

The bridge-model result is therefore no longer limited to contrastive image towers. On this
benchmark and under a selected-layer extraction regime, both contrastive VLMs and autoregressive
VLMs look far more vision-like than language-like. The architecture extension does not answer every
multimodal question, but it materially widens the scope of the bridge-model claim.

### 4.3 Depth Helps More Than Final-Layer Choice

The aligned-layer analysis is where the main positive result of the study appears. The headline
conclusion barely changes when the primary selected-layer analysis is compared with the terminal
aligned-layer readout in `aligned5`, but the depth profile is informative.

**Table 5: Mean RSA by aligned depth fraction in the 250-concept run.**

| Depth | Overall | Language-Language | Image-Image | Language-Image |
|------|--------:|------------------:|------------:|---------------:|
| d00 | 0.2381 | 0.6143 | 0.3859 | 0.0240 |
| d25 | 0.3053 | 0.6143 | 0.5317 | 0.0441 |
| d50 | 0.3094 | 0.6143 | 0.5372 | 0.0481 |
| d75 | 0.3096 | 0.6143 | 0.5450 | 0.0420 |
| d100 | 0.2683 | 0.6143 | 0.4679 | 0.0197 |

![Figure 5. Aligned5 depth profile. Language-language agreement is essentially flat across depth, but image-image and language-image agreement rise strongly into the middle and late layers before dropping back at the terminal selected layer.](figures/scale250/aligned5_depth_profile.png)

This depth pattern matters for interpretation. It says that whatever shared geometry exists in this
panel is strongest in mid-to-late aligned layers, not uniquely in the terminal layer. The
language-language line stays flat, while image-image and language-image comparisons do the moving.

At the same time, aligned layers do **not** rescue a strong PRH reading. The selected-layer baseline
and selected-layer `aligned5` summaries are nearly identical: both have 138/231 significant
model-pair results, the mean absolute pairwise shift is only 0.0002, and 210 of 231 pairwise RSA
values are unchanged exactly. So aligned layers change *where* the convergence signal is most
visible, not *whether* the final high-level conclusion changes.

### 4.4 Metric Triangulation With CKA Preserves the Same Ordering

The main metric-triangulation question is whether the RSA story survives outside the RSA family. The
answer is yes at the qualitative level.

**Table 6: Broad family comparison under linear CKA.**

| Pair type | Pairs | Mean CKA | Median CKA |
|----------|------:|---------:|-----------:|
| Language-Language | 28 | 0.7429 | 0.7350 |
| Image-Image | 91 | 0.6338 | 0.6407 |
| Language-Image | 112 | 0.4657 | 0.4766 |

For this CKA summary, the four contrastive VLM image encoders are grouped with the image side,
because the extracted representations are image-tower features. Under that definition, the same broad
ordering survives: language-language is highest, image-image remains next, and language-image is the
weakest category. Pairwise agreement between RSA and CKA over all 231 model pairs is substantial
(Pearson 0.7229, Spearman 0.7050), which is exactly what we want from metric triangulation: not
identical magnitudes, but the same structural ranking.

![Figure 6. Baseline CKA heatmap for the same 22-model selected-layer run. The block structure closely matches the RSA view, with the strongest alignments concentrated among image-side and contrastive VLM encoders.](figures/scale250/baseline_cka_heatmap.png)

![Figure 7. Pairwise RSA vs. CKA across all 231 model pairs in the Scale250 baseline. The metrics are not numerically interchangeable, but the positive correlation shows that the family structure is not an artifact of RSA alone.](figures/scale250/rsa_vs_cka_scatter.png)

The same top families dominate under CKA. The strongest pairs are again `SigLIP` vs `SigLIP2`
(0.9749), `CLIP-ViT-B32` vs `MetaCLIP-B32-400m` (0.9712), and `MetaCLIP-B32-400m` vs `SigLIP2`
(0.9605). The weakest CKA pairs still concentrate in cross-modal comparisons involving weaker
language-image alignment, especially around `ViT-MSN-base`.

The numeric gap is less stark under CKA than under RSA. That difference is expected rather than
concerning. RSA is a rank correlation over similarity matrices; CKA is a centered alignment on
feature matrices and is bounded positive in this implementation. The important result is not that
the numbers match exactly, but that the qualitative family ordering survives a second metric family.

### 4.5 Prompt Sensitivity Is Real but Manageable

Prompt sensitivity is materially smaller on Scale250 than on the earlier small benchmark, but it is
not negligible.

Across the 8 language models in the Scale250 baseline:

- mean maximum absolute shift vs. baseline template: 0.0766
- median maximum absolute shift: 0.0818
- maximum shift: 0.1238

By model, the least prompt-sensitive and most prompt-sensitive cases are:

- `Granite-3.3-2B-Instruct-8bit`: 0.0446
- `Qwen3-1.7B-MLX-8bit`: 0.0460
- `SmolLM3-3B-8bit`: 0.1238

![Figure 8. Prompt sensitivity across language models. Values are the maximum absolute shift in cross-modal RSA relative to the baseline template over the two alternative templates in the `baseline3` family.](figures/scale250/prompt_sensitivity.png)

The image-bootstrap confidence intervals are also much tighter than in the previous benchmark:

- previous benchmark mean CI width: 0.1787
- Scale250 mean CI width: 0.0552

That improvement matters. The Scale250 benchmark is not only harder; it is also more statistically
stable. The broader concept set shrinks uncertainty enough that small cross-modal effects can be
interpreted more cautiously and more credibly.

### 4.6 There Is No Clean Language-Model Scaling Law Here

The earlier small-benchmark study entertained a tentative scale story. Scale250 does not support it.

Using the baseline scaling analysis in `results/scale250_full/baseline/scaling/scaling_analysis.png`,
the Spearman correlation between language-model size and mean vision RSA is -0.0952. That is
effectively no monotonic scaling law.

The per-model pattern is mixed:

- best mean vision alignment: `Qwen3-0.6B-MLX-8bit` at 0.0455
- worst mean vision alignment: `Granite-3.3-2B-Instruct-8bit` at -0.0333
- `Qwen3-4B-MLX-8bit` improves over the 2B and 1.5B cases, but not over the 0.6B case

![Figure 9. Baseline scaling analysis. Cross-modal RSA with vision models does not increase monotonically with language-model size across the 0.6B-4B panel.](figures/scale250/baseline_scaling.png)

This does not mean size never matters. It means size is not the dominant variable in this local
panel once the benchmark is broadened. Architecture family, training data, and objective appear to
matter more than raw parameter count.

### 4.7 Supporting Historical Comparison to the Earlier Small Benchmark

For context, the earlier 20-base-concept local benchmark reported much stronger cross-modal
agreement. When the same 22-model panel is evaluated on Scale250, that estimate drops substantially.

**Table 7: Broad family comparison, earlier small benchmark vs. Scale250.**

| Benchmark | Language-Language mean $\rho$ | Image-Image mean $\rho$ | Language-Image mean $\rho$ | Significant pairs |
|-----------|------------------------------:|------------------------:|---------------------------:|------------------:|
| Earlier small benchmark | 0.6703 | 0.6721 | 0.2418 | 157 / 231 |
| Scale250 baseline | 0.6143 | 0.4681 | 0.0199 | 138 / 231 |

![Figure 10. Benchmark scale changes the cross-modal story. Left: mean RSA by broad family pairing. Right: FDR-significant fraction by family pairing. The large reduction is specifically in language-image agreement, while language-language agreement remains strong and image-image agreement remains substantial.](figures/scale250/old_vs_new_family_summary.png)

The scale-up does not merely shrink everything uniformly. The strongest change is in cross-modal
agreement. Language-image mean RSA drops from 0.2418 in the earlier benchmark to 0.0199 in
Scale250, and the significant fraction falls from 40/112 to 21/112. That is the clearest evidence
in the project that small benchmarks can materially overstate cross-modal convergence.

At the same time, the broader benchmark does **not** erase all structure. Language-language
agreement stays high, and image-image agreement stays well above zero. The comparison therefore does
not replace the Scale250 result. It clarifies why the Scale250 estimates should be treated as the
canonical ones for this panel.

---

## 5. Discussion

### 5.1 What This Study Establishes

The strongest interpretation of the Scale250 study is not that convergence disappears. It is that
convergence becomes more selective and more conditional under a broader and more balanced benchmark.

The canonical result is family structure, not universal convergence. Language models agree strongly
with one another, image-side models agree strongly with one another, and multimodal bridge models sit
much closer to the image side than to the language side. Cross-modal language-image agreement is
present, but weak as a general statement on this benchmark.

The architecture extension sharpens that story further. Adding three autoregressive VLMs does not
pull the bridge-model block toward language. It leaves the geometry on the image side. Weak
language-image agreement is therefore not just a quirk of contrastive dual encoders in the original
22-model panel.

That is not a trivial negative result. It clarifies the boundary of the phenomenon:

- language models converge strongly with language models
- image-side models converge strongly with image-side models
- contrastive bridge models behave more like image models than like modality-neutral intermediates
- language-image convergence is weak as a general statement on this benchmark

### 5.2 Why the Bridge Models Matter Across Architectures

The bridge-model result remains conceptually important, but it needs to be stated carefully. In the
core 22-model panel, the bridge models are contrastive dual encoders evaluated through their image
towers. In the architecture extension, the bridge models include three native autoregressive VLMs evaluated
through their image-conditioned selected-layer embeddings. Under both regimes, they cluster strongly
with vision.

That does **not** prove that all multimodal models are vision-anchored under every extraction
choice. It does support a narrower and useful claim: effective multimodal interfaces can be built
without erasing modality-shaped internal structure, and the present evidence points consistently
toward vision-anchored bridge geometry rather than modality-neutral convergence.

### 5.3 What the Depth Result Adds

The aligned5 result is the most constructive positive finding in the study. It suggests that shared
geometry, such as it is, is better understood as a developmental pattern across depth than as a
property of final-layer readouts alone. Mid-to-late aligned layers carry stronger cross-family
structure than the terminal selected layer.

That finding keeps a weaker Platonic story alive. One could argue that convergence begins inside the
network before being distorted by task-specific final readout behavior. But on the present evidence,
that weaker story remains a plausible interpretation rather than a confirmed universal law.

### 5.4 Benchmark Design and Historical Context

The supporting historical comparison matters because it shows that benchmark design is not a minor
implementation detail. On the same 22-model panel, the earlier 20-base-concept benchmark yielded a
much stronger language-image estimate than the 250-concept Scale250 benchmark. The difference is not
just numerical. It changes the level of confidence we should place in broad cross-modal claims on
this panel.

That does not mean the paper should be read mainly as a correction note to earlier internal work.
The primary contribution is the standalone characterization of what survives on the stronger
Scale250 benchmark. The historical comparison is useful because it explains why the current study is
the canonical reference point and why small benchmarks deserve caution when they are used to support
large theoretical claims.

### 5.5 What This Study Can and Cannot Refute About PRH

This study does **not** refute the broadest possible version of PRH. It does not test frontier
language models, it does not include a broad sweep of autoregressive multimodal architectures, and it
does not cover a rich abstract-concept regime. Stronger cross-modal effects could still emerge under
those conditions.

What the study *does* support is narrower and more concrete:

1. On this benchmark, robust agreement is much stronger within families than across language and
   image.
2. The bridge-model result survives an architecture extension from contrastive VLMs to three
   autoregressive VLMs.
3. Any surviving cross-family structure is more visible in aligned mid-to-late layers than in the
   final selected layer.
4. The supporting historical comparison indicates that small benchmarks can materially overstate
   language-image geometric convergence on this panel.

That is already a meaningful boundary on the strongest reading of PRH in this setup. The right claim
is not "this panel is broadly cross-modally aligned." It is "this panel shows weak cross-modal
alignment once measured on a stronger benchmark."

### 5.6 Limitations

The study is substantially better grounded than the earlier small-benchmark version, but it is not
complete.

1. **No primary compositionality claim.** The primary Scale250 benchmark intentionally excludes
   compounds, so compositionality is no longer part of the main evidence package.
2. **Source-holdout remains incomplete.** The new benchmark improves source balance by design, but
   the current robustness implementation collapses source holdout to a single `mixed_balanced`
   regime, so this paper does not claim a strong source-ablation result.
3. **Scale still tops out at 4B on the language side.** A stronger cross-modal effect could emerge
   at materially larger scale.
4. **The concept set is concrete-heavy.** The benchmark is broader than before, but not yet a strong
   test of abstraction.
5. **The AR-VLM extension is still narrow and selected-layer only.** The architecture stress test
   now includes three completed autoregressive VLMs, but it does not yet include a broader family
   sweep or aligned-layer internal hooks for those models.

### 5.7 Implications for the Next Experiment

The next high-value experiments are now clearer than they were before Scale250.

- Complete a broader autoregressive VLM panel, including one or two additional non-Qwen families and
  a lighter-weight completion of the `SmolVLM2` run.
- Add a dedicated abstract-concept tranche to test whether abstraction increases or decreases
  cross-modal agreement.
- Repair the per-image-source holdout path so the balanced benchmark supports a real source-ablation
  analysis.
- Implement aligned-layer hooks for at least one autoregressive VLM family so the depth result can be
  tested beyond the current selected-layer extension.
- Expand same-family language-model scaling to separate architecture from parameter count.
- Run aligned-layer analyses on larger language models rather than only the 0.6B-4B regime.

The study therefore narrows the open question. It no longer asks, "Do different modalities ever
agree?" They do, weakly. The real question is now: under what benchmark, architecture, and depth
conditions does that agreement become strong enough to count as evidence for a shared geometry?

---

## 6. Conclusion

The Scale250 study provides a broader and more balanced measurement of cross-modal geometry on this
local panel.

It combines a structured 250-concept benchmark, strict within-concept source balance, aligned-layer
analysis, robustness controls, CKA triangulation, and a focused autoregressive-VLM architecture
extension. The result is a stronger test of the hypothesis than a small or loosely balanced
benchmark can provide.

The resulting picture is more conservative and more informative:

1. **Within-modality convergence is robust.**
2. **Both contrastive and autoregressive bridge models are strongly vision-like, not
   modality-neutral.**
3. **Cross-modal language-image geometry is weak on a broad balanced benchmark.**
4. **Mid-to-late layers are more informative than the final selected layer.**
5. **Language-model size does not explain cross-modal alignment in this panel.**
6. **The same qualitative result survives a second metric family, linear CKA.**

The right high-level takeaway is therefore not that PRH has been disproved in general. It is that
the strongest supported structure on this panel is within-family alignment, vision-anchored bridge
behavior, and depth-dependent overlap rather than broad language-image convergence. A supporting
comparison to the earlier small-benchmark study suggests that narrow benchmarks can materially
overstate cross-modal agreement, but the main contribution of the paper is the standalone
characterization of what survives under the stronger Scale250 regime.

---

## Code and Data Availability

Code, manifests, small paper-facing result artifacts, figure-generation code, and manuscript
sources are available in this repository and are intended to be published alongside the paper at
`https://github.com/abinashkarki/cross-modal-representations`. Heavy compiled Scale250 artifacts are
excluded from git and indexed in `artifacts/release_manifest.json` with checksums in
`artifacts/SHA256SUMS.txt`. The canonical in-repo artifacts for this paper are:

- `data/data_manifest_250.json`
- `results/scale250_full/baseline/robustness_opt_full/robustness_stats.json`
- `results/scale250_full/aligned5/robustness/robustness_stats.json`
- `results/scale250_full/baseline25_extension/architecture_analysis/architecture_summary.json`
- `src/generate_scale250_paper_figures.py`
- `src/analyze_arvlm_extension.py`
- `artifacts/release_manifest.json`
- `manuscript/figures/scale250/`

---

## References

Andreas, J. (2019). Measuring Compositionality in Representation Learning. *ICLR 2019*.

Huh, M., Cheung, B., Wang, T., and Isola, P. (2024). The Platonic Representation Hypothesis.
*arXiv preprint arXiv:2405.07987*.

Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. (2019). Similarity of Neural Network
Representations Revisited. *ICML 2019*.

Kriegeskorte, N., Mur, M., and Bandettini, P. A. (2008). Representational similarity analysis:
connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.

Morcos, A. S., Raghu, M., and Bengio, S. (2018). Insights on representational similarity in neural
networks with canonical correlation. *NeurIPS 2018*.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013). Distributed
representations of words and phrases and their compositionality. *NeurIPS 2013*.

Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J. (2017). SVCCA: Singular Vector
Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. *arXiv preprint
arXiv:1706.05806*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A.,
Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. (2021). Learning Transferable Visual Models
From Natural Language Supervision. *ICML 2021*.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. (2023). Sigmoid Loss for Language Image
Pre-Training. *ICCV 2023*.

---

## Appendix A: Reproducibility Summary

- Supporting historical comparison benchmark: 20 base concepts for RSA, 8 compound probes, same
  22-model panel
- Primary Scale250 benchmark: 250 base concepts, no compounds in the primary set
- Secondary architecture extension: 3 completed autoregressive VLMs merged into a 25-model selected-layer panel
- Images per concept: 15
- Source policy: 5 ImageNet, 5 Open Images, 5 Unsplash per concept
- Model pairs tested: 231
- Bootstrap draws: 300
- Bootstrap image sample size: 10
- Mantel permutations per pair: 3,000
- Prompt templates extracted: 3
- Layer protocols: selected-layer baseline and aligned5
- Secondary metric: linear CKA on selected-layer concept matrices

## Appendix B: Files Used For This Paper

- `data/data_manifest_250.json`
- `results/scale250_full/baseline/robustness_opt_full/robustness_stats.json`
- `results/scale250_full/aligned5/robustness/robustness_stats.json`
- `results/scale250_full/baseline25_extension/architecture_analysis/architecture_summary.json`
- `artifacts/release_manifest.json`
- `artifacts/SHA256SUMS.txt`
- `src/analyze_arvlm_extension.py`
- `results/baseline/robustness/robustness_stats.json`
- `src/generate_scale250_paper_figures.py`
- `manuscript/figures/scale250/`
