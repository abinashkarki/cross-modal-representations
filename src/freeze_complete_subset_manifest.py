import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
DEFAULT_OUTPUT_MANIFEST = os.path.join(REPO_ROOT, "data", "data_manifest_250_complete_subset.json")
DEFAULT_OUTPUT_TRACKER = os.path.join(REPO_ROOT, "data", "scale250_complete_subset_concept_tracker.csv")
DEFAULT_OUTPUT_INVENTORY = os.path.join(REPO_ROOT, "data", "scale250_complete_subset_curation_inventory.csv")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_subset_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def complete_concepts(manifest: Dict[str, Any]) -> List[str]:
    result = []
    for concept, images in manifest.get("concept_to_images", {}).items():
        md = manifest["concept_metadata"][concept]
        actual = md.get("source_mix_actual", {})
        if len(images) == 15 and actual == {"imagenet": 5, "openimages": 5, "unsplash": 5}:
            result.append(concept)
    return result


def build_subset_manifest(manifest: Dict[str, Any], concepts: List[str]) -> Dict[str, Any]:
    output = dict(manifest)
    output["manifest_status"] = "complete_subset_frozen"
    output["description"] = (
        "Frozen subset of the fresh balanced build containing only concepts that currently satisfy "
        "the exact 5/5/5 source contract."
    )
    output["subset_scope"] = {
        "concept_count": len(concepts),
        "selection_rule": "include only concepts with exactly 15 images and exact 5/5/5 source balance",
        "selected_concepts": concepts,
    }
    output["concept_to_images"] = {concept: list(manifest["concept_to_images"][concept]) for concept in concepts}
    output["concept_metadata"] = {concept: dict(manifest["concept_metadata"][concept]) for concept in concepts}
    return output


def regenerate_curator_sheets(manifest_path: str, inventory_path: str, tracker_path: str) -> None:
    generator_path = os.path.join(SCRIPT_DIR, "generate_curation_inventory.py")
    subprocess.run(
        [
            sys.executable,
            generator_path,
            "--manifest-path",
            manifest_path,
            "--inventory-output",
            inventory_path,
            "--tracker-output",
            tracker_path,
            "--image-root-relpath",
            "data/images_250_fresh",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the currently complete 5/5/5 subset from the fresh scale250 manifest.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-tracker", default=DEFAULT_OUTPUT_TRACKER)
    parser.add_argument("--output-inventory", default=DEFAULT_OUTPUT_INVENTORY)
    args = parser.parse_args()

    manifest = load_json(os.path.abspath(args.manifest_path))
    concepts = complete_concepts(manifest)
    subset = build_subset_manifest(manifest, concepts)

    output_manifest = os.path.abspath(args.output_manifest)
    atomic_write_json(output_manifest, subset)
    regenerate_curator_sheets(output_manifest, os.path.abspath(args.output_inventory), os.path.abspath(args.output_tracker))

    print(f"Manifest: {output_manifest}")
    print(f"Concept count: {len(concepts)}")
    print(f"Tracker: {os.path.abspath(args.output_tracker)}")
    print(f"Inventory: {os.path.abspath(args.output_inventory)}")


if __name__ == "__main__":
    main()
