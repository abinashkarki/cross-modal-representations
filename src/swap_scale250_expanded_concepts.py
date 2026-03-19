import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
DEFAULT_TRACKER_PATH = os.path.join(REPO_ROOT, "data", "scale250_fresh_concept_tracker.csv")
DEFAULT_INVENTORY_PATH = os.path.join(REPO_ROOT, "data", "scale250_fresh_curation_inventory.csv")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_xswap_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def atomic_write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_xswap_", suffix=".csv", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    return "_".join(part for part in cleaned.split("_") if part)


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
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def build_expanded_metadata(
    manifest: Dict[str, Any],
    concept: str,
    catalog_row: Dict[str, Any],
) -> Dict[str, Any]:
    target_per_source = manifest["source_balance_policy"]["target_per_source"]
    return {
        "source": "mixed_balanced",
        "stratum": catalog_row["stratum"],
        "semantic_type": catalog_row.get("semantic_type", "entity"),
        "selection_status": "expanded_reserve_swap",
        "source_feasibility": catalog_row.get("source_feasibility", "high"),
        "storage_slug": slugify(concept),
        "num_images": 0,
        "description": concept,
        "source_mix_target": dict(target_per_source),
        "source_mix_actual": {source: 0 for source in target_per_source},
        "clip_scores": {},
        "image_sources": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap active empty concepts for expanded reserve concepts.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--tracker-path", default=DEFAULT_TRACKER_PATH)
    parser.add_argument("--inventory-path", default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--catalog-path", required=True)
    parser.add_argument("--swap", nargs="+", required=True, help="Pairs in old:new form")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest_path)
    tracker_path = os.path.abspath(args.tracker_path)
    inventory_path = os.path.abspath(args.inventory_path)
    catalog_path = os.path.abspath(args.catalog_path)

    manifest = load_json(manifest_path)
    catalog = load_json(catalog_path)
    provenance_rel = manifest.get("shadow_build", {}).get("provenance_ledger", "data/scale250_fresh_provenance.csv")
    provenance_path = os.path.join(REPO_ROOT, provenance_rel)
    provenance_rows = load_csv(provenance_path)

    concept_to_images = manifest["concept_to_images"]
    concept_metadata = manifest["concept_metadata"]
    old_concepts = list(concept_to_images.keys())

    swaps: List[Tuple[str, str]] = []
    for raw in args.swap:
        if ":" not in raw:
            raise ValueError(f"Invalid --swap entry '{raw}'. Expected old:new")
        old_concept, new_concept = raw.split(":", 1)
        swaps.append((old_concept, new_concept))

    swap_map = dict(swaps)
    for old_concept, new_concept in swaps:
        if old_concept not in concept_to_images:
            raise KeyError(f"Active concept not found: {old_concept}")
        if new_concept in concept_to_images:
            raise ValueError(f"Expanded reserve concept already present in manifest: {new_concept}")
        if new_concept not in catalog:
            raise KeyError(f"Expanded reserve concept missing from catalog: {new_concept}")

    new_concept_to_images: Dict[str, List[str]] = {}
    new_concept_metadata: Dict[str, Any] = {}
    for concept in old_concepts:
        replacement = swap_map.get(concept)
        if replacement is None:
            new_concept_to_images[concept] = concept_to_images[concept]
            new_concept_metadata[concept] = concept_metadata[concept]
            continue
        new_concept_to_images[replacement] = []
        new_concept_metadata[replacement] = build_expanded_metadata(manifest, replacement, catalog[replacement])

    manifest["concept_to_images"] = new_concept_to_images
    manifest["concept_metadata"] = new_concept_metadata

    retired_at = time.strftime("%Y-%m-%d %H:%M:%S")
    for row in provenance_rows:
        concept = row.get("concept", "")
        replacement = swap_map.get(concept)
        if replacement is None:
            continue
        if row.get("review_status") == "accepted_auto":
            row["review_status"] = "retired_concept_swap"
            row["rejection_reason"] = ""
        note = row.get("notes", "")
        suffix = f"Retired from active manifest at {retired_at} and replaced by expanded reserve '{replacement}'."
        row["notes"] = f"{note} {suffix}".strip()

    atomic_write_json(manifest_path, manifest)
    if provenance_rows:
        atomic_write_csv(provenance_path, provenance_rows)
    regenerate_curator_sheets(manifest_path, inventory_path, tracker_path)

    print(f"Manifest: {manifest_path}")
    print(f"Provenance: {provenance_path}")
    print(f"Tracker: {tracker_path}")
    print(f"Inventory: {inventory_path}")
    print(f"Applied expanded swaps: {swaps}")


if __name__ == "__main__":
    main()
