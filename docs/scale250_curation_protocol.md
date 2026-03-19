# Scale250 Curation Protocol

This protocol is for the stage before model extraction, when the dataset is still being assembled.

## Goal

Build a `250`-concept benchmark with exact within-concept source balance:

- `5` ImageNet images
- `5` OpenImages images
- `5` Unsplash images

Every confirmatory concept must satisfy the exact `5/5/5` quota. If a concept cannot satisfy it, replace that concept only from the reserve list in the same stratum.

## Core Files

- roster: `data/concept_roster_250_scaffold.json`
- manifest skeleton: `data/data_manifest_250_skeleton.json`
- inventory generator: `src/generate_curation_inventory.py`
- manifest sync helper: `src/sync_manifest_curation.py`

## Directory And Filename Convention

Store images in:

```text
data/images_250/<storage_slug>/
```

Use filenames that encode the source:

```text
<storage_slug>_<source>_<NN>.jpg
```

Examples:

```text
data/images_250/cat/cat_imagenet_01.jpg
data/images_250/cat/cat_openimages_01.jpg
data/images_250/cat/cat_unsplash_01.jpg
```

Why this matters:

- the source is visible in the filename
- `sync_manifest_curation.py` can infer `image_sources` automatically from filenames
- quota auditing becomes trivial

## Curator Workflow

1. Freeze the concept roster.
2. Generate the curation inventory and concept tracker:

```bash
python src/generate_curation_inventory.py \
  --manifest-path data/data_manifest_250_skeleton.json
```

3. Work from `data/scale250_curation_inventory.csv` while collecting images.
4. Save accepted images into the correct concept folder using the standard filename pattern.
5. After each curation batch, sync the manifest:

```bash
python src/sync_manifest_curation.py \
  --manifest-path data/data_manifest_250_skeleton.json \
  --image-root data/images_250 \
  --sync-image-paths true \
  --infer-image-sources-from-filenames true \
  --write
```

6. Run the strict quota gate periodically:

```bash
python src/sync_manifest_curation.py \
  --manifest-path data/data_manifest_250_skeleton.json \
  --image-root data/images_250 \
  --sync-image-paths true \
  --infer-image-sources-from-filenames true \
  --strict-image-sources true \
  --strict-source-balance true
```

Only rename the manifest to `data/data_manifest_250.json` when it is fully curated and passes the strict gate.

## Inclusion Rules

- The image should depict the target concept clearly and centrally enough that a human would name the concept without relying on context.
- Use natural photographs, not icons, diagrams, screenshots, or synthetic graphic compositions.
- Avoid text-dominant images, meme-like images, watermarked promotional images, and collages.
- Avoid highly ambiguous subtype drift. Example: `dog` can include breed variation, but not toy figurines or logos.
- Avoid repeated near-duplicates within a concept, especially the same object instance, scene, camera burst, or photographer series.

## Diversity Rules

- Within each concept/source cell, prefer variation in viewpoint, background, and instance.
- Do not let all `5` images in one cell come from the same visual subtype unless the concept itself is that subtype.
- For scene-like concepts, vary weather, composition, and geography when possible.

## Replacement Rule

If a concept is proving hard to balance:

- do not silently accept `5/5/3` or similar
- mark it as blocked in the tracker
- replace it with a reserve concept from the same stratum

That keeps the confirmatory set stratified and prevents source confounding from re-entering the design.
