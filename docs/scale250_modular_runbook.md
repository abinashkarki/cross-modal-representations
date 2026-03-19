# 250-Concept Modular Runbook

This runbook turns the 250-concept scale-up into a staged 2-3 day workflow on the existing repo.

## Recommended Confirmatory Design

- `250` concepts
- `15` images per concept
- balanced source coverage within concept (`5/5/5` across the three sources)
- full `22`-model baseline pass
- `baseline3` language templates retained at extraction time
- final baseline robustness with cached image resampling
- aligned-layer pass (`aligned5`) as stage-two confirmatory analysis

## Important Repo Detail

The extractor now accepts `--manifest-path`, and `docs/run_all_models.sh` now forwards `MANIFEST_PATH`
into extraction. This means the 250-concept run does not require swapping the default manifest file.

Preflight now also enforces exact per-concept source balance whenever a manifest declares
`source_balance_policy.mode = within_concept_balanced`.

## Build The Manifest From The Roster

Start from:

- roster scaffold: `data/concept_roster_250_scaffold.json`
- generator: `src/generate_manifest_from_roster.py`

Generate an editable manifest skeleton with empty image lists:

```bash
python src/generate_manifest_from_roster.py \
  --roster-path data/concept_roster_250_scaffold.json \
  --output-path data/data_manifest_250_skeleton.json
```

If you want expected storage paths prefilled while curating:

```bash
python src/generate_manifest_from_roster.py \
  --roster-path data/concept_roster_250_scaffold.json \
  --output-path data/data_manifest_250_placeholder.json \
  --populate-placeholder-paths true
```

Recommended workflow:

1. Keep the roster scaffold as the a priori concept-selection record.
2. Generate the skeleton manifest from the roster.
3. Populate real image paths, `clip_scores`, and `image_sources`.
4. Update `source_mix_actual` as curation finishes.
5. Rename the filled file to `data/data_manifest_250.json` only once it is genuinely runnable.

Use the curation sync helper during dataset construction:

```bash
python src/sync_manifest_curation.py \
  --manifest-path data/data_manifest_250.json \
  --image-root data/images_250 \
  --sync-image-paths true \
  --infer-image-sources-from-filenames true \
  --write
```

For a strict gate on source coverage while curating:

```bash
python src/sync_manifest_curation.py \
  --manifest-path data/data_manifest_250.json \
  --image-root data/images_250 \
  --sync-image-paths true \
  --infer-image-sources-from-filenames true \
  --strict-image-sources true \
  --strict-source-balance true
```

Before curation starts, generate the curator-facing inventory and tracker:

```bash
python src/generate_curation_inventory.py \
  --manifest-path data/data_manifest_250_skeleton.json
```

The detailed pre-image workflow is documented in `docs/scale250_curation_protocol.md`.

## Directory Convention

Use separate roots for baseline and aligned5 so cache manifests cannot collide:

- `results/scale250/baseline`
- `results/scale250/aligned5`

Each root will contain:

- `raw_data/`
- `logs/`
- `cache/`
- compiled outputs and robustness outputs created later

## Day 0: Freeze The Protocol

Before the long run starts, freeze:

- the 250-concept manifest
- source balancing rules
- the model panel
- the prompt protocol
- the primary confirmatory metrics

Suggested prompt rule:

- extract with `baseline3`
- decide in advance whether the paper's primary language condition is:
  - the prompt range as a fixed factor, or
  - a fixed prompt ensemble built from the extracted templates

## Day 1: Smoke Panel Then Full Baseline Extraction

### 1) Manifest preflight

```bash
python src/preflight_replication.py \
  --phase pre \
  --manifest data/data_manifest_250.json \
  --raw-dir results/scale250/baseline/raw_data \
  --models-file docs/run_all_models.sh \
  --cache-dir results/scale250/baseline/cache \
  --min-images-per-concept 15 \
  --require-image-source-metadata true \
  --require-clip-scores
```

### 2) Smoke panel

The smoke panel intentionally spans language, vision, and VLM:

- `Qwen3-1.7B-MLX-8bit`
- `Granite-3.3-2B-Instruct-8bit`
- `DINOv2-base`
- `I-JEPA`
- `SigLIP2`

Run it with:

```bash
MANIFEST_PATH=data/data_manifest_250.json \
RESULTS_ROOT=results/scale250/baseline \
RUN_PROFILE=baseline \
MODELS_FILE=docs/model_panel_smoke5.txt \
TEXT_TEMPLATE_SET=baseline3 \
bash docs/run_modular_panel.sh
```

What you are checking:

- manifest integrity
- image-path correctness
- prompt/template extraction behavior
- cache creation
- per-model runtime sanity

### 3) Full baseline extraction

If the smoke panel is clean, run the full 22-model baseline panel:

```bash
MANIFEST_PATH=data/data_manifest_250.json \
RESULTS_ROOT=results/scale250/baseline \
RUN_PROFILE=baseline \
MODELS_FILE=docs/model_panel_core22.txt \
TEXT_TEMPLATE_SET=baseline3 \
bash docs/run_modular_panel.sh
```

Notes:

- reruns are restartable because extraction is per model
- leave `FORCE_OVERWRITE=false` during the long run so completed models are skipped
- keep `CACHE_IMAGE_EMBEDDINGS=true`

## Day 2: Compile, Quick Robustness, Start Aligned5

### 1) Post-extraction preflight

```bash
python src/preflight_replication.py \
  --phase post \
  --manifest data/data_manifest_250.json \
  --raw-dir results/scale250/baseline/raw_data \
  --models-file docs/run_all_models.sh \
  --cache-dir results/scale250/baseline/cache \
  --min-images-per-concept 15 \
  --require-image-source-metadata true \
  --require-cache \
  --require-clip-scores
```

### 2) Strict compile

```bash
python src/compile_results.py \
  --strict \
  --raw-dir results/scale250/baseline/raw_data \
  --output-file results/scale250/baseline/replication_results.json \
  --manifest data/data_manifest_250.json \
  --min-models 22
```

### 3) Quick robustness gate

Run a fast gate before the final overnight pass:

```bash
python src/robustness_analysis.py \
  --data-file results/scale250/baseline/replication_results.json \
  --manifest-path data/data_manifest_250.json \
  --layer selected \
  --bootstrap-draws 100 \
  --bootstrap-sample-size 10 \
  --bootstrap-replacement true \
  --mantel-permutations 1000 \
  --min-concepts-for-rsa 8 \
  --seed 42 \
  --cache-dir results/scale250/baseline/cache \
  --output-dir results/scale250/baseline/robustness_quick
```

Decision rule:

- if family medians, CI widths, prompt effects, and source-holdout outputs look sane, proceed
- if something is clearly broken, fix it before launching the final overnight robustness job

### 4) Start aligned5 extraction

```bash
MANIFEST_PATH=data/data_manifest_250.json \
RESULTS_ROOT=results/scale250/aligned5 \
RUN_PROFILE=aligned5 \
MODELS_FILE=docs/model_panel_core22.txt \
TEXT_TEMPLATE_SET=baseline3 \
bash docs/run_modular_panel.sh
```

## Day 2 Overnight: Final Baseline Robustness

```bash
python src/robustness_analysis.py \
  --data-file results/scale250/baseline/replication_results.json \
  --manifest-path data/data_manifest_250.json \
  --layer selected \
  --bootstrap-draws 300 \
  --bootstrap-sample-size 10 \
  --bootstrap-replacement true \
  --mantel-permutations 3000 \
  --min-concepts-for-rsa 8 \
  --seed 42 \
  --cache-dir results/scale250/baseline/cache \
  --output-dir results/scale250/baseline/robustness
```

## Day 3: Compile Aligned5 And Finish Extensions

If aligned5 extraction completed:

```bash
python src/compile_results.py \
  --strict \
  --raw-dir results/scale250/aligned5/raw_data \
  --output-file results/scale250/aligned5/replication_results.json \
  --manifest data/data_manifest_250.json \
  --min-models 22
```

Then run:

- `visualize_replication_results.py`
- `scaling_analysis.py`
- `robustness_analysis.py` for aligned5

Suggested priority order for Day 3:

1. aligned5 compile and depth analyses
2. final prompt-protocol decision (`range` vs `ensemble`)
3. quantization sensitivity
4. extra model families or extra prompt families

## Confirmatory vs Exploratory Split

Confirmatory:

- full 22-model baseline
- 250 concepts
- 15 images per concept
- within-concept source balancing
- baseline3 extraction
- full baseline robustness

Secondary confirmatory:

- full aligned5 pass on the same concept set

Exploratory:

- quantization sensitivity
- richer prompt families
- extra model additions
- alternative image-count ablations
