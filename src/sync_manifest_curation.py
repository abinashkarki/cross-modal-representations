import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Set


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp_path, path)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower())
    return slug.strip("_")


def infer_source_from_filename(image_name: str, required_sources: List[str]) -> str:
    stem, _ext = os.path.splitext(image_name)
    if not required_sources:
        return ""
    alternation = "|".join(re.escape(source) for source in sorted(required_sources, key=len, reverse=True))
    match = re.match(rf"^.+_(?P<source>{alternation})_(?P<slot>\d+)$", stem)
    if not match:
        return ""
    return str(match.group("source"))


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def scan_image_dir(image_root: str, storage_slug: str) -> List[str]:
    concept_dir = os.path.join(image_root, storage_slug)
    if not os.path.isdir(concept_dir):
        return []
    files = sorted(
        name for name in os.listdir(concept_dir)
        if name.lower().endswith(IMAGE_EXTS)
    )
    return [os.path.relpath(os.path.join(concept_dir, name), REPO_ROOT) for name in files]


def concept_names_in_order(
    concept_to_images: Dict[str, List[str]],
    concept_metadata: Dict[str, Any],
) -> List[str]:
    names = list(concept_to_images.keys())
    seen: Set[str] = set(names)
    for concept in concept_metadata.keys():
        if concept not in seen:
            names.append(concept)
            seen.add(concept)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync manifest curation metadata and optionally rebuild image path lists."
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--image-root",
        default="data/images_250_fresh",
        help="Image root relative to repo root or as an absolute path.",
    )
    parser.add_argument(
        "--sync-image-paths",
        type=str,
        default="false",
        help="Scan image directories and replace concept_to_images entries (true|false).",
    )
    parser.add_argument(
        "--prune-stale-metadata",
        type=str,
        default="false",
        help="Drop clip_scores/image_sources keys that do not correspond to current image files.",
    )
    parser.add_argument(
        "--strict-image-sources",
        type=str,
        default="false",
        help="Fail if any current image is missing an image_sources entry.",
    )
    parser.add_argument(
        "--strict-source-balance",
        type=str,
        default="false",
        help="Fail if any concept violates the declared source_balance_policy.",
    )
    parser.add_argument(
        "--infer-image-sources-from-filenames",
        type=str,
        default="false",
        help=(
            "Fill missing image_sources entries from filenames like "
            "<storage_slug>_<source>_<NN>.jpg (true|false)."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write updated manifest back to disk.",
    )
    args = parser.parse_args()

    sync_image_paths = parse_bool(args.sync_image_paths)
    prune_stale_metadata = parse_bool(args.prune_stale_metadata)
    strict_image_sources = parse_bool(args.strict_image_sources)
    strict_source_balance = parse_bool(args.strict_source_balance)
    infer_image_sources_from_filenames = parse_bool(args.infer_image_sources_from_filenames)

    manifest_path = os.path.abspath(args.manifest_path)
    image_root = (
        os.path.abspath(args.image_root)
        if os.path.isabs(args.image_root)
        else os.path.join(REPO_ROOT, args.image_root)
    )

    manifest = load_json(manifest_path)
    concept_to_images = manifest.setdefault("concept_to_images", {})
    concept_metadata = manifest.setdefault("concept_metadata", {})
    policy = manifest.get("source_balance_policy", {})
    required_sources = list(policy.get("required_sources", []))
    target_per_source = {
        source: int(count) for source, count in policy.get("target_per_source", {}).items()
    }

    concepts = concept_names_in_order(concept_to_images, concept_metadata)
    missing_source_entries = []
    balance_violations = []
    summary_rows = []

    for concept in concepts:
        md = concept_metadata.setdefault(concept, {})
        storage_slug = md.get("storage_slug") or slugify(concept)
        md["storage_slug"] = storage_slug

        if sync_image_paths:
            concept_to_images[concept] = scan_image_dir(image_root, storage_slug)

        images = concept_to_images.get(concept, [])
        image_names = [os.path.basename(path) for path in images]
        image_name_set = set(image_names)

        clip_scores = md.get("clip_scores")
        if not isinstance(clip_scores, dict):
            clip_scores = {}
        image_sources = md.get("image_sources")
        if not isinstance(image_sources, dict):
            image_sources = {}

        if prune_stale_metadata:
            clip_scores = {name: value for name, value in clip_scores.items() if name in image_name_set}
            image_sources = {name: value for name, value in image_sources.items() if name in image_name_set}

        source_counts = Counter()
        missing_for_concept = []
        unknown_source_values = []
        for image_name in image_names:
            if infer_image_sources_from_filenames and image_name not in image_sources:
                inferred_source = infer_source_from_filename(image_name, required_sources)
                if inferred_source:
                    image_sources[image_name] = inferred_source
            source = image_sources.get(image_name)
            if source is None:
                missing_for_concept.append(image_name)
                continue
            source_counts[source] += 1
            if required_sources and source not in required_sources:
                unknown_source_values.append(f"{image_name}={source}")

        if missing_for_concept:
            missing_source_entries.append((concept, missing_for_concept))
        if unknown_source_values:
            balance_violations.append(
                f"{concept}: unknown image_sources values {unknown_source_values[:5]}"
            )

        md["clip_scores"] = clip_scores
        md["image_sources"] = image_sources
        md["num_images"] = len(images)
        if required_sources:
            md["source_mix_actual"] = {
                source: int(source_counts.get(source, 0)) for source in required_sources
            }
        else:
            md["source_mix_actual"] = dict(sorted(source_counts.items()))

        if target_per_source:
            for source, target in target_per_source.items():
                actual = int(source_counts.get(source, 0))
                if actual != target:
                    balance_violations.append(
                        f"{concept}: source '{source}' actual={actual} target={target}"
                    )

        summary_rows.append(
            {
                "concept": concept,
                "num_images": len(images),
                "missing_image_sources": len(missing_for_concept),
                "clip_scores_present": len(clip_scores),
                "source_mix_actual": md["source_mix_actual"],
            }
        )

    if args.write:
        atomic_write_json(manifest_path, manifest)

    print(f"Manifest: {manifest_path}")
    print(f"Image root: {image_root}")
    print(f"Concept count: {len(concepts)}")
    print(f"Sync image paths: {sync_image_paths}")
    print(f"Prune stale metadata: {prune_stale_metadata}")
    print(f"Infer image_sources from filenames: {infer_image_sources_from_filenames}")
    print(f"Write changes: {args.write}")
    print("")

    totals = Counter()
    for row in summary_rows:
        totals["images"] += row["num_images"]
        totals["missing_image_sources"] += row["missing_image_sources"]
        totals["clip_scores_present"] += row["clip_scores_present"]
    print(
        "Totals: "
        f"images={totals['images']}, "
        f"clip_scores_present={totals['clip_scores_present']}, "
        f"missing_image_sources={totals['missing_image_sources']}"
    )

    if summary_rows:
        print("")
        print("Sample concepts:")
        for row in summary_rows[:10]:
            print(
                f"  - {row['concept']}: "
                f"images={row['num_images']}, "
                f"missing_image_sources={row['missing_image_sources']}, "
                f"source_mix_actual={row['source_mix_actual']}"
            )

    if missing_source_entries:
        print("")
        print(f"Concepts with missing image_sources entries: {len(missing_source_entries)}")
        for concept, missing in missing_source_entries[:15]:
            print(f"  - {concept}: missing {len(missing)} entries (sample {missing[:5]})")

    if balance_violations:
        print("")
        print(f"Source-balance issues: {len(balance_violations)}")
        for row in balance_violations[:30]:
            print(f"  - {row}")

    if strict_image_sources and missing_source_entries:
        raise SystemExit(1)
    if strict_source_balance and balance_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
