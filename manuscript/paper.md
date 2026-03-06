---
title: "Where Representations Diverge: Robust Evidence on Modality, Dataset, and Depth in Cross-Modal Geometry"
author: "Abinash Karki (Independent Research)"
date: "March 2026"
---

---

## Abstract

The Platonic Representation Hypothesis proposes that sufficiently capable models trained on
different modalities converge toward a shared representational geometry. We test that claim using
self-contained local evaluation of 22 models: 8 language models, 10
self-supervised vision models, and 4 vision-language models. The final evaluation uses 28
concepts (20 base + 8 compounds), 30 images per concept, image-level caching, image-bootstrap
confidence intervals, Mantel permutation tests with Benjamini-Hochberg FDR correction, source
holdout analysis, prompt-sensitivity controls, and an aligned-layer protocol. Across 231 model
pairs, within-family agreement is strong for vision-language pairs (median Spearman RSA
$\rho = 0.702$), vision-vision pairs ($\rho = 0.682$), and language-language pairs
($\rho = 0.661$), but much weaker for language-vision pairs ($\rho = 0.259$) and
language-vision-language pairs ($\rho = 0.230$). Only 35/80 language-vision pairs and 5/32
language-vision-language pairs survive FDR correction, versus 43/45 vision-vision pairs and
40/40 vision-vision-language pairs. Robustness controls materially affect interpretation:
leave-one-source-out removal of ImageNet shifts pairwise RSA by a mean of -0.1564
(max $|\Delta| = 0.7080$), language prompt choice changes cross-modal RSA by a mean maximum
absolute delta of 0.1960, and aligned-layer analysis shows monotonic growth in mean pairwise RSA
from 0.3424 at early depth to 0.4639 at terminal depth. Yet terminal aligned and baseline results
are nearly identical (mean delta +0.0006), indicating that the depth effect is about when
convergence appears, not whether the final layer changes the conclusion. The strongest supported
claim is therefore not universal convergence. Rather, this local-scale stress test of strong PRH
finds modality-asymmetric and strongly source-conditional geometry in this setup: vision-language
encoders remain structurally closer to vision than to language, and cross-modal geometry is
sensitive to image source, prompt framing, and representational depth.

**Keywords**: Platonic Representation Hypothesis, representational similarity analysis,
cross-modal convergence, multimodal models, source holdout, prompt sensitivity, layer alignment

---

## 1. Introduction

### 1.1 The Question Behind the Hypothesis

When distributional embeddings were found to encode relational regularities such as "king - man +
woman = queen" (Mikolov et al., 2013), they suggested that representation may discover geometry
rather than merely store labels. The Platonic Representation Hypothesis (Huh et al., 2024) pushes
that intuition further: if different model families are all trained on rich evidence about the
same world, then their internal spaces should converge toward the same underlying structure.

That is a strong claim. It says representational geometry is not primarily a byproduct of
modality, architecture, or objective design, but a consequence of modeling reality itself. Under
that view, a pure language model and a pure vision model should increasingly agree on how concepts
relate, even if they were never trained together.

### 1.2 Why This Matters Beyond AI

The question matters because it reframes convergence as evidence about inevitability. If very
different systems repeatedly discover similar structures, then some representational solutions may
be mathematically favored rather than historically accidental. That matters not only for AI, but
for broader questions about perception, abstraction, and biological constraint.

A useful analogy is convergent evolution. Foveae, sparse coding, and selective attention recur in
biological systems because they solve recurring problems under recurring constraints. The Platonic
hypothesis asks whether an analogous statement holds for neural representation: do distinct model
families discover the same geometry because the world itself imposes it?

### 1.3 From Positive Results to Boundary Conditions

The hypothesis is easy to overstate. Apparent convergence can arise from multiple weaker causes:
shared internet distributions, architectural bias, language-alignment objectives, prompt choices,
or evaluation artifacts. A useful experiment therefore does not only ask whether convergence
exists. It asks what survives after the obvious confounds are stressed.

This paper is explicitly a stress test of the strong form of PRH, not a full adjudication of
every weaker Platonic view.

This paper is organized around that stricter standard. We retain the original V1 question, but we
answer it using the final 22-model replication package rather than the earlier prototype runs. The
goal is not to extract the most optimistic reading of the data. It is to identify the most
defensible one.

### 1.4 This Paper

We evaluate 22 models across three families:

- 8 language models
- 10 self-supervised vision models
- 4 vision-language image encoders

The final dataset contains 20 base concepts, 8 compound concepts, and 30 curated images per
concept. We measure cross-model agreement with Representational Similarity Analysis (RSA) over the
20 base concepts and supplement it with image-bootstrap confidence intervals, Mantel permutation
tests, source holdout analysis, prompt sensitivity, and an aligned-layer depth profile.

Our research questions are:

- **RQ1**: How much representational geometry is shared within and across modality families?
- **RQ2**: Do vision-language encoders bridge language and vision, or do they remain
  predominantly vision-like?
- **RQ3**: How stable are the conclusions under changes in image source, text prompt, and layer
  depth?
- **RQ4**: Does compound-concept behavior add independent evidence for universal convergence?

Our central claim is narrow: this study stress-tests the strong form of PRH under local-scale
evaluation. The final replication supports robust within-family convergence and strong
vision-to-vision-language coupling, but it does not support a strong version of universal
cross-modal convergence at the scales tested here.

---

## 2. Related Work

Huh et al. (2024) introduced the Platonic Representation Hypothesis and argued that stronger
models show increasing representational agreement across language and vision. Their framing is
useful because it turns a philosophical intuition into a measurable claim about geometry.

Representational Similarity Analysis (RSA) provides the relevant tool (Kriegeskorte et al., 2008).
RSA compares the geometry of pairwise relationships inside two representational spaces rather than
the coordinates of the embeddings themselves, making it appropriate for comparing models with
different dimensionalities and architectures.

The present work differs from frontier-scale Platonic evaluations in three ways. First, it is
deliberately constrained to a local compute regime, forcing the question toward minimum viable
scale rather than massive capacity. Second, it emphasizes robustness controls that can materially
change interpretation: image bootstrap, source holdout, prompt sensitivity, and aligned-layer
analysis. Third, it uses a broad family comparison rather than a single bridge-model success case.

This produces a more conservative but more interpretable result: not simply whether some
cross-modal signal exists, but how that signal compares to within-family agreement and how stable
it remains once the obvious nuisance variables are perturbed.

---

## 3. Methods

### 3.1 Model Selection

The final core roster contains 22 models chosen to remain within the lab's hardware constraint
while spanning distinct modality families and training objectives.

**Table 1: Final model roster.**

| Family | Models |
|-------|--------|
| Language (8) | Falcon3-1B-Instruct-8bit, Granite-3.3-2B-Instruct-8bit, LFM2-2.6B-Exp-8bit, Qwen2.5-1.5B-Instruct-8bit, Qwen3-0.6B-MLX-8bit, Qwen3-1.7B-MLX-8bit, Qwen3-4B-MLX-8bit, SmolLM3-3B-8bit |
| Vision SSL (10) | BEiT-base, ConvNeXt-v2, DINOv2-base, DINOv2-small, DINOv3-ConvNeXt-tiny, Hiera-base, I-JEPA, ViT-MAE-base, ViT-MSN-base, data2vec-vision |
| Vision-Language (4) | CLIP-ViT-B32, MetaCLIP-B32-400m, SigLIP, SigLIP2 |

The language models cover roughly 0.6B to 4B parameters and include both MLX and Transformer
backends. The vision group spans multiple self-supervised paradigms, including masked prediction,
self-distillation, joint-embedding prediction, and ConvNeXt-based encoders. The vision-language
group provides the key bridge condition: models trained with explicit image-language alignment but
evaluated here using their image encoders alone.

### 3.2 Concept Set and Image Construction

The evaluation uses 28 concepts:

- 20 base concepts for RSA
- 8 compound concepts for exploratory compositionality analysis

The 20 base concepts are:

- cat, dog, bird, fish, elephant
- car, bridge, airplane, mountain, building
- fire, water, forest, ocean, road
- city, space, sun, moon, storm

The 8 compound concepts are:

- forest fire, space city, water city, city forest
- mountain road, ocean bridge, city bridge, mountain forest

Each concept has exactly 30 images. The final manifest contains full CLIP-score coverage for all
28 concepts and allocates source coverage at the concept level across ImageNet (10 concepts),
OpenImages (6 concepts), and Unsplash (12 concepts). This source diversity matters empirically,
because later holdout analysis shows that image source is not a cosmetic detail but a major driver
of measured geometry.

### 3.3 Embedding Extraction and Layer Protocol

For language models, concept embeddings are extracted from a short textual prompt. The canonical
template is:

`The concept of {concept}`

Prompt sensitivity is evaluated using two alternatives:

- `An example of {concept}`
- `The meaning of {concept}`

For image models, per-image embeddings are extracted and then aggregated to concept-level
representations. Image-level embeddings are cached so that bootstrap resampling and source
holdouts can be run without recomputing forward passes.

Two layer protocols are used:

- **Baseline**: the selected terminal representation (`-1` / final-layer equivalent)
- **Aligned5**: five depth fractions (`d00`, `d25`, `d50`, `d75`, `d100`)

The aligned protocol is important because it distinguishes final-output agreement from depth-wise
emergence. A convergence pattern that only appears at terminal depth means something different
from one that accumulates monotonically across the network.

### 3.4 Representational Similarity Analysis

RSA is computed over the 20 base concepts only.

For each model:

1. Construct a 20 x 20 cosine-similarity matrix over concept embeddings.
2. Extract the upper triangle (190 unique concept pairs).
3. Compute Spearman rank correlation between model-pair vectors.

Spearman correlation is used because the relevant question is rank agreement in concept geometry,
not absolute similarity scale, which differs substantially across model families.

### 3.5 Statistical Inference and Robustness

The final replication supplements raw RSA with four robustness procedures.

**Mantel significance.**

- 3,000 permutations per model pair
- Two-sided p-values
- Benjamini-Hochberg FDR correction across all 231 model pairs

**Image bootstrap.**

- 300 bootstrap draws
- sample size = 10 images per concept
- sampling with replacement

**Source holdout.**

- leave-one-source-out (LOSO)
- source-only analysis when enough concepts remain

**Prompt sensitivity.**

- three text templates for all language models

These controls are not afterthoughts. They materially shift conclusions. The paper's main claims
are based on what remains stable after these checks.

---

## 4. Results

### 4.1 Global Geometry Is Family-Structured, Not Universally Shared

The most stable finding in the final replication is modality asymmetry.

**Table 2: Pairwise RSA summary by family.**

| Pair type | Pairs | Median $\rho$ | Mean $\rho$ | Significant after FDR |
|----------|------:|--------------:|------------:|----------------------:|
| Language-Language | 28 | 0.661 | 0.670 | 28 / 28 |
| Vision-Vision | 45 | 0.682 | 0.630 | 43 / 45 |
| Vision-Vision-Language | 40 | 0.702 | 0.681 | 40 / 40 |
| Language-Vision | 80 | 0.259 | 0.246 | 35 / 80 |
| Language-Vision-Language | 32 | 0.230 | 0.232 | 5 / 32 |
| Vision-Language-Vision-Language | 6 | 0.939 | 0.927 | 6 / 6 |

Three features stand out.

First, within-family agreement is strong in both language and vision. Language models agree with
each other at roughly the same level as vision models agree with each other. This supports a weak
form of convergence: once modality and objective family are held roughly fixed, representational
geometry becomes reproducible.

Second, the vision-language encoders are much closer to vision than to language. The median
vision-to-vision-language RSA (0.702) is more than three times the median
language-to-vision-language RSA (0.230). That is the opposite of what one would expect if
vision-language encoders served as genuinely intermediate representations between the two
modalities.

Third, cross-modal signal exists but remains partial. Language-vision pairs are not centered at
zero; many are significant. But they are far below within-family agreement. The strongest
language-vision pair is I-JEPA with SmolLM3-3B-8bit ($\rho = 0.500$), while the strongest
vision-vision pair is ConvNeXt-v2 with Hiera-base ($\rho = 0.929$). The scale of the difference
matters more than the existence of a few positive pairs.

### 4.2 Vision-Language Models Behave Like Vision Models

The bridge-model question is central because multimodal models are often used as intuitive support
for Platonic convergence. The final replication does not support treating multimodal success alone
as evidence for a shared modality-neutral geometry.

All 40 vision-to-vision-language pairs are significant after FDR correction, with a median
$\rho = 0.702$. By contrast, only 5 of 32 language-to-vision-language pairs survive correction,
with a median $\rho = 0.230$.

Representative high-similarity bridge pairs include:

- CLIP-ViT-B32 with I-JEPA: $\rho = 0.823$
- I-JEPA with SigLIP2: $\rho = 0.815$
- CLIP-ViT-B32 with Hiera-base: $\rho = 0.776$
- SigLIP2 with ViT-MAE-base: $\rho = 0.774$

In this setup, the best-supported interpretation is that contrastive training creates a useful
cross-modal interface while leaving the image encoder predominantly vision-like. That is a more
defensible reading than claiming that language and vision have converged onto one shared geometry
internally.

### 4.3 Measured Geometry Is Strongly Source-Conditional

The final V2 package added explicit source holdout analysis. This was scientifically important,
because source dependence turned out to be large enough to alter interpretation.

**Table 3: Source holdout effects relative to the full baseline run.**

| Holdout condition | Concepts retained | Mean $\Delta \rho$ | Mean $|\Delta \rho|$ | Max $|\Delta \rho|$ |
|------------------|------------------:|-------------------:|---------------------:|--------------------:|
| Leave out ImageNet | 10 | -0.1564 | 0.2358 | 0.7080 |
| Leave out OpenImages | 14 | -0.0792 | 0.0865 | 0.2654 |
| Leave out Unsplash | 16 | +0.0900 | 0.1295 | 0.4170 |
| ImageNet only | 10 | -0.0194 | 0.1912 | 0.5931 |

The ImageNet result is the most consequential. Excluding ImageNet reduces pairwise RSA by
-0.1564 on average, with some model pairs shifting by more than 0.70. The asymmetry is also
family-structured: mean $\Delta \rho$ is +0.0382 for vision-vision pairs and +0.0323 for
vision-vision-language pairs, but -0.3526 for language-vision pairs and -0.4518 for
language-vision-language pairs. The 20 largest absolute shifts are all cross-family.

These results are consistent with dataset-conditional geometry in this setup, but they do not
isolate dataset source as a unique causal factor. Leaving out ImageNet also changes retained
concepts, image style, and likely object typicality. The safer claim is therefore that measured
geometry is strongly source-conditional in this benchmark, not that dataset identity alone has
been uniquely isolated as the cause.

![Figure 2. ImageNet holdout impact by family pair and largest absolute pairwise shifts. Removing ImageNet leaves vision-side pairs comparatively stable but drives large negative shifts in language-vision and language-vision-language pairs; the top 20 absolute shifts are all cross-family.](../results/v2_change_assets/imagenet_holdout_impact.png)

### 4.4 Prompt Choice and Uncertainty Are Not Negligible

Language-side prompt choice also matters more than a casual evaluation might suggest.

Across the 8 language models, the mean maximum absolute shift in cross-modal RSA under the two
alternative templates is 0.1960, with a range from 0.1446 to 0.2785. The most prompt-sensitive
model in the final run is Qwen3-0.6B-MLX-8bit; the least sensitive is
Granite-3.3-2B-Instruct-8bit.

Bootstrap uncertainty confirms that the image side contributes substantial variance. Across the
231 model pairs, image-bootstrap confidence intervals have:

- mean width: 0.1787
- median width: 0.1900
- maximum width: 0.3703

The implication is methodological. Small cross-modal differences should not be overinterpreted
without uncertainty estimates, because both prompt framing and image resampling can move the
results by amounts comparable to many headline pairwise gaps.

### 4.5 Deeper Layers Converge More, but the Final Answer Stays the Same

The aligned-layer analysis shows a clear depth trend:

**Table 4: Mean pairwise RSA by aligned depth fraction.**

| Depth fraction | Mean pairwise $\rho$ | Median pairwise $\rho$ |
|---------------|---------------------:|-----------------------:|
| d00 | 0.3424 | 0.2491 |
| d25 | 0.4149 | 0.3356 |
| d50 | 0.4345 | 0.3857 |
| d75 | 0.4403 | 0.4067 |
| d100 | 0.4639 | 0.4346 |

Convergence strengthens monotonically with depth. This is one of the clearest positive findings in
the entire project. It suggests that representational agreement is accumulated, not merely read out
at the end.

But the second half of the depth result is equally important: terminal aligned and baseline runs
are almost identical. The mean aligned-minus-baseline pairwise delta is +0.0006, and the FDR
significance profile is unchanged. So depth changes when the geometry becomes visible, but it does
not reverse the final high-level conclusion. Deeper layers amplify the same family structure
rather than revealing a hidden universal geometry that the baseline missed.

### 4.6 Compositionality Is Exploratory, Not a Main Claim

The earlier small-scale analyses encouraged a strong story about compositionality. The final 22-model
package does not support such strong wording.

Using the current V2 metrics:

- mean additive compositionality is 0.948 for vision-language models
- mean additive compositionality is 0.873 for vision models
- mean additive compositionality is 0.831 for language models

The balance-style score is even less separating, with family means compressed into the range
0.907 to 0.965. In other words, compound-concept behavior in the final pipeline does not cleanly
partition by modality. It overlaps broadly and depends strongly on metric design.

This does not make the compositionality analysis useless. It makes it secondary. The final paper
therefore treats compositionality as exploratory evidence rather than a core pillar of the
conclusion.

---

## 5. Discussion

### 5.1 What Converges, and What Does Not

The final replication supports two statements and rejects one.

Supported:

- models converge strongly within family
- vision-language encoders converge strongly with vision encoders

Not supported:

- all sufficiently capable modalities converge toward one shared geometry

This is not a null result. It narrows the hypothesis. The data are consistent with modality-shaped
structure that remains partially coupled across modalities but does not collapse into one shared
space at the tested scale.

### 5.2 Why Bridge Models Do Not Rescue the Strong Hypothesis

One tempting reply is that the vision-language models already demonstrate cross-modal convergence.
But that reading confuses interface alignment with internal geometry.

The bridge models are trained precisely to connect images and text. If their image encoders still
end up substantially closer to vision than to language, then multimodal success cannot be taken as
independent evidence for a strong Platonic claim. It demonstrates engineered compatibility, which
is useful, but conceptually different.

### 5.3 Why Robustness Controls Change the Interpretation

The most important difference between the early prototype and the final replication is not model
count. It is methodological pressure.

Three controls matter especially:

- source holdout shows that image provenance materially changes geometry
- prompt sensitivity shows that language-side framing materially changes geometry
- aligned-layer analysis shows that convergence is depth-dependent but terminally stable

This is why the current paper is more conservative than the early narrative. Once those controls
are in place, the strongest claims are about boundary conditions, not about universal agreement.

### 5.4 Limitations

The paper still has clear limits.

1. **No controlled retraining.** We compare pretrained models; we do not intervene on data or
   objective while holding architecture fixed.
2. **Scale is modest.** The language side is capped at 4B parameters by design. A stronger
   Platonic effect may emerge later.
3. **Concepts are still concrete-heavy.** The evaluation does not yet include enough abstract
   concepts to test whether abstraction strengthens or weakens cross-modal agreement.
4. **Prompting is narrow.** We use short definitional prompts, not long contextual or sensory
   prompts.
5. **Quantization sensitivity is incomplete.** A dedicated q4 vs q8 final ablation was identified
   as still worth doing but is not part of the sign-off package.

### 5.5 Future Directions

The most informative next experiments are not simply "more models." They are more controlled
models.

High-value next steps are:

- same-family scaling to isolate size from architecture
- intervention on image source composition
- abstract-concept expansion
- richer text prompt families
- controlled language-alignment training to distinguish engineering from emergence

If the Platonic hypothesis is true in a strong sense, it should survive those interventions. If it
does not, then the field needs a weaker and more precise formulation.

---

## 6. Conclusion

We tested the Platonic Representation Hypothesis using the final local 22-model replication rather
than the earlier prototype reports. The result is not that cross-modal geometry is absent. It is
that the stable signal is weaker and more conditional than an optimistic reading suggests.

The final evidence supports four conclusions:

1. **Within-family convergence is robust.** Language models agree strongly with language models,
   and vision models agree strongly with vision models.
2. **Vision-language encoders are structurally vision-like in this setup.** Their geometry is much
   closer to vision than to language.
3. **Cross-modal geometry is real but partial.** It is substantially weaker than within-family
   agreement and often disappears under stricter significance thresholds.
4. **Interpretation depends on controls.** Image source, prompt template, and layer depth all
   matter enough to change the story.

The strong Platonic claim is therefore not supported in this local-scale stress test. The more
defensible statement is weaker: representational geometry is shaped by both world structure and
modality-specific training pressures, and the balance between those forces remains an open
empirical question.

### What We Still Do Not Know

Two possibilities remain live.

- **Weak Platonic view**: the present runs are below the scale where cross-modal geometry becomes
  dominant, but the monotonic depth trend and non-zero cross-modal signal are early evidence.
- **Family-structured view**: each modality learns a partially overlapping but fundamentally
  distinct geometry, and explicit alignment is required to bridge them.

The current project resolves neither possibility completely. What it does provide is a cleaner map
of the boundary: where convergence is already robust, where it is still fragile, and which controls
must be passed before stronger claims become credible.

---

## Code and Data Availability

Code, curated data manifests, final result artifacts, and the print-ready manuscript are available
at [https://github.com/abinashkarki/cross-modal-representations](https://github.com/abinashkarki/cross-modal-representations).

The canonical public artifacts for this release are the manuscript in `manuscript/`, the curated
dataset in `data/`, and the signed-off baseline and aligned outputs in `results/baseline/` and
`results/aligned5/`.

---

## References

Andreas, J. (2019). Measuring Compositionality in Representation Learning. *ICLR 2019*.

Bao, H., Dong, L., Piao, S., and Wei, F. (2022). BEiT: BERT Pre-Training of Image
Transformers. *ICLR 2022*.

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, R. (2022). Masked Autoencoders Are
Scalable Vision Learners. *CVPR 2022*.

Huh, M., Cheung, B., Wang, T., and Isola, P. (2024). The Platonic Representation Hypothesis.
*arXiv preprint arXiv:2405.07987*.

Kriegeskorte, N., Mur, M., and Bandettini, P. A. (2008). Representational similarity analysis:
connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013). Distributed
representations of words and phrases and their compositionality. *NeurIPS 2013*.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., and Bojanowski, P.
(2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv preprint
arXiv:2304.07193*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A.,
Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. (2021). Learning Transferable Visual
Models From Natural Language Supervision. *ICML 2021*.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. (2023). Sigmoid Loss for Language Image
Pre-Training. *ICCV 2023*.

---

## Appendix A: Statistical Summary

- Models tested: 22
- Model pairs tested: 231
- Base concepts used for RSA: 20
- Compound concepts used for exploratory analysis: 8
- Images per concept: 30
- Total concept images: 840
- Mantel permutations per pair: 3,000
- Bootstrap draws: 300
- Bootstrap image sample size: 10
- Execution context: local commodity hardware

## Appendix B: Reproducibility Notes

The final paper is based on the completed standalone release artifacts in `results/baseline/`,
`results/aligned5/`, and the corresponding robustness outputs. The final claims in this draft are
aligned with the signed-off baseline, aligned5, and robustness artifacts rather than with the
earlier prototype summaries.
