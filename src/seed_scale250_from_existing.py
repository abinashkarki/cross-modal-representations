import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from typing import Any, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_SOURCE_MANIFEST = os.path.join(REPO_ROOT, "data", "data_manifest_multi.json")
DEFAULT_TARGET_MANIFEST = os.path.join(REPO_ROOT, "data", "data_manifest_250_skeleton.json")
DEFAULT_TARGET_IMAGE_ROOT = os.path.join(REPO_ROOT, "data", "images_250")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_seed_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def copy_seed_images(
    source_manifest: Dict[str, Any],
    target_manifest: Dict[str, Any],
    target_image_root: str,
    per_concept: int,
    concepts: List[str],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    source_concept_to_images = source_manifest.get("concept_to_images", {})
    source_metadata = source_manifest.get("concept_metadata", {})
    target_concept_to_images = target_manifest.setdefault("concept_to_images", {})
    target_metadata = target_manifest.setdefault("concept_metadata", {})

    copied_rows: List[Dict[str, Any]] = []
    for concept in concepts:
        source_images = source_concept_to_images.get(concept, [])
        if not source_images:
            continue
        source_md = source_metadata.get(concept, {})
        source_name = str(source_md.get("source", "")).strip().lower()
        if source_name not in {"imagenet", "openimages", "unsplash"}:
            continue

        target_md = target_metadata.setdefault(concept, {})
        storage_slug = target_md.get("storage_slug")
        if not storage_slug:
            continue

        clip_scores = target_md.setdefault("clip_scores", {})
        image_sources = target_md.setdefault("image_sources", {})
        dest_dir = os.path.join(target_image_root, storage_slug)
        os.makedirs(dest_dir, exist_ok=True)

        picked = source_images[:per_concept]
        for idx, rel_source_path in enumerate(picked, start=1):
            src_abs = os.path.join(REPO_ROOT, rel_source_path)
            if not os.path.exists(src_abs):
                continue
            dest_name = f"{storage_slug}_{source_name}_{idx:02d}.jpg"
            dest_abs = os.path.join(dest_dir, dest_name)
            if os.path.exists(dest_abs) and not overwrite:
                continue
            shutil.copy2(src_abs, dest_abs)

            src_basename = os.path.basename(rel_source_path)
            source_score = source_md.get("clip_scores", {}).get(src_basename)
            if source_score is not None:
                clip_scores[dest_name] = source_score
            image_sources[dest_name] = source_name

        current_relpaths = sorted(
            os.path.join("data", "images_250", storage_slug, name)
            for name in os.listdir(dest_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        target_concept_to_images[concept] = current_relpaths

        counts = Counter(image_sources.get(os.path.basename(path), "") for path in current_relpaths)
        target_md["num_images"] = len(current_relpaths)
        target_md["source_mix_actual"] = {
            "imagenet": int(counts.get("imagenet", 0)),
            "openimages": int(counts.get("openimages", 0)),
            "unsplash": int(counts.get("unsplash", 0)),
        }
        copied_rows.append(
            {
                "concept": concept,
                "seed_source": source_name,
                "copied": len(current_relpaths),
                "storage_slug": storage_slug,
            }
        )
    return copied_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the 250-concept manifest from overlapping concepts in the existing multi-concept dataset."
    )
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--target-manifest", default=DEFAULT_TARGET_MANIFEST)
    parser.add_argument("--target-image-root", default=DEFAULT_TARGET_IMAGE_ROOT)
    parser.add_argument("--per-concept", type=int, default=5)
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="Optional subset of overlapping concepts to seed.",
    )
    parser.add_argument("--overwrite", type=str, default="false")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    overwrite = parse_bool(args.overwrite)
    source_manifest = load_json(os.path.abspath(args.source_manifest))
    target_manifest = load_json(os.path.abspath(args.target_manifest))

    source_concepts = set(source_manifest.get("concept_to_images", {}))
    target_concepts = set(target_manifest.get("concept_to_images", {}))
    overlap = sorted(source_concepts & target_concepts)
    concepts = overlap if not args.concepts else [c for c in args.concepts if c in overlap]

    copied_rows = copy_seed_images(
        source_manifest=source_manifest,
        target_manifest=target_manifest,
        target_image_root=os.path.abspath(args.target_image_root),
        per_concept=args.per_concept,
        concepts=concepts,
        overwrite=overwrite,
    )

    if args.write:
        atomic_write_json(os.path.abspath(args.target_manifest), target_manifest)

    print(f"Overlapping concepts available: {len(overlap)}")
    print(f"Seeded concepts: {len(copied_rows)}")
    for row in copied_rows:
        print(
            f"  - {row['concept']}: "
            f"source={row['seed_source']} copied={row['copied']} "
            f"storage_slug={row['storage_slug']}"
        )
    if args.write:
        print(f"\nUpdated manifest: {os.path.abspath(args.target_manifest)}")


if __name__ == "__main__":
    main()
