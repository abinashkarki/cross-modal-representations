import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from typing import Any, Dict, List, Sequence


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
DEFAULT_INVENTORY_PATH = os.path.join(REPO_ROOT, "data", "scale250_curation_inventory.csv")
DEFAULT_TRACKER_PATH = os.path.join(REPO_ROOT, "data", "scale250_concept_tracker.csv")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_csv(path: str, fieldnames: Sequence[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_curation_", suffix=".csv", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def concept_names_in_order(
    concept_to_images: Dict[str, List[str]],
    concept_metadata: Dict[str, Any],
) -> List[str]:
    names = list(concept_to_images.keys())
    seen = set(names)
    for concept in concept_metadata.keys():
        if concept not in seen:
            names.append(concept)
            seen.add(concept)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate curator-facing inventory and tracker CSVs from the scale-up manifest."
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--inventory-output", default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--tracker-output", default=DEFAULT_TRACKER_PATH)
    parser.add_argument(
        "--image-root-relpath",
        default="",
        help=(
            "Override the image root used in suggested_relpath. "
            "Defaults to manifest.shadow_build.active_image_root when present."
        ),
    )
    args = parser.parse_args()

    manifest = load_json(os.path.abspath(args.manifest_path))
    concept_to_images = manifest.get("concept_to_images", {})
    concept_metadata = manifest.get("concept_metadata", {})
    policy = manifest.get("source_balance_policy", {})
    required_sources = list(policy.get("required_sources", []))
    target_per_source = {
        source: int(count) for source, count in policy.get("target_per_source", {}).items()
    }
    image_root_relpath = args.image_root_relpath or manifest.get("shadow_build", {}).get(
        "active_image_root", "data/images_250_fresh"
    )
    if not required_sources or not target_per_source:
        raise ValueError("Manifest is missing source_balance_policy.required_sources/target_per_source.")

    inventory_rows: List[Dict[str, Any]] = []
    tracker_rows: List[Dict[str, Any]] = []

    for concept in concept_names_in_order(concept_to_images, concept_metadata):
        md = concept_metadata.get(concept, {})
        storage_slug = md.get("storage_slug", "")
        images = concept_to_images.get(concept, [])
        image_sources = md.get("image_sources", {})
        source_to_files: Dict[str, List[str]] = {source: [] for source in required_sources}

        for image_path in images:
            image_name = os.path.basename(image_path)
            source = image_sources.get(image_name, "")
            if source in source_to_files:
                source_to_files[source].append(image_name)

        for source in required_sources:
            source_to_files[source].sort()

        current_counts = Counter({source: len(files) for source, files in source_to_files.items()})
        tracker_rows.append(
            {
                "concept": concept,
                "stratum": md.get("stratum", ""),
                "semantic_type": md.get("semantic_type", ""),
                "selection_status": md.get("selection_status", ""),
                "source_feasibility": md.get("source_feasibility", ""),
                "storage_slug": storage_slug,
                "target_total": sum(target_per_source.values()),
                "current_total": len(images),
                "current_imagenet": current_counts.get("imagenet", 0),
                "current_openimages": current_counts.get("openimages", 0),
                "current_unsplash": current_counts.get("unsplash", 0),
                "remaining_imagenet": max(0, target_per_source.get("imagenet", 0) - current_counts.get("imagenet", 0)),
                "remaining_openimages": max(0, target_per_source.get("openimages", 0) - current_counts.get("openimages", 0)),
                "remaining_unsplash": max(0, target_per_source.get("unsplash", 0) - current_counts.get("unsplash", 0)),
                "concept_status": "complete" if all(
                    current_counts.get(source, 0) >= target_per_source.get(source, 0)
                    for source in required_sources
                ) else "pending",
                "replacement_policy": "replace only from reserve list in same stratum if quota cannot be met",
            }
        )

        for source in required_sources:
            target = target_per_source[source]
            existing_files = source_to_files[source]
            for slot_idx in range(1, target + 1):
                suggested_filename = f"{storage_slug}_{source}_{slot_idx:02d}.jpg"
                inventory_rows.append(
                    {
                        "concept": concept,
                        "stratum": md.get("stratum", ""),
                        "semantic_type": md.get("semantic_type", ""),
                        "selection_status": md.get("selection_status", ""),
                        "source_feasibility": md.get("source_feasibility", ""),
                        "storage_slug": storage_slug,
                        "source": source,
                        "source_slot": slot_idx,
                        "suggested_filename": suggested_filename,
                        "suggested_relpath": os.path.join(image_root_relpath, storage_slug, suggested_filename),
                        "slot_status": "filled" if slot_idx <= len(existing_files) else "pending",
                        "actual_filename": existing_files[slot_idx - 1] if slot_idx <= len(existing_files) else "",
                        "candidate_record_id": "",
                        "candidate_url": "",
                        "license_or_terms": "",
                        "curator_initials": "",
                        "notes": "",
                    }
                )

    atomic_write_csv(
        os.path.abspath(args.inventory_output),
        [
            "concept",
            "stratum",
            "semantic_type",
            "selection_status",
            "source_feasibility",
            "storage_slug",
            "source",
            "source_slot",
            "suggested_filename",
            "suggested_relpath",
            "slot_status",
            "actual_filename",
            "candidate_record_id",
            "candidate_url",
            "license_or_terms",
            "curator_initials",
            "notes",
        ],
        inventory_rows,
    )
    atomic_write_csv(
        os.path.abspath(args.tracker_output),
        [
            "concept",
            "stratum",
            "semantic_type",
            "selection_status",
            "source_feasibility",
            "storage_slug",
            "target_total",
            "current_total",
            "current_imagenet",
            "current_openimages",
            "current_unsplash",
            "remaining_imagenet",
            "remaining_openimages",
            "remaining_unsplash",
            "concept_status",
            "replacement_policy",
        ],
        tracker_rows,
    )

    print(f"Wrote inventory -> {os.path.abspath(args.inventory_output)}")
    print(f"Wrote concept tracker -> {os.path.abspath(args.tracker_output)}")
    print(f"Concept count: {len(tracker_rows)}")
    print(f"Inventory rows: {len(inventory_rows)}")


if __name__ == "__main__":
    main()
