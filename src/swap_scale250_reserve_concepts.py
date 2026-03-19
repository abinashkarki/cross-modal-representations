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
DEFAULT_ROSTER_PATH = os.path.join(REPO_ROOT, "data", "concept_roster_250_scaffold.json")
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
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_swap_", suffix=".json", dir=os.path.dirname(path))
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
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_swap_", suffix=".csv", dir=os.path.dirname(path))
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


def build_reserve_metadata(manifest: Dict[str, Any], stratum_id: str, concept: str, row: Dict[str, Any]) -> Dict[str, Any]:
    target_per_source = manifest["source_balance_policy"]["target_per_source"]
    return {
        "source": "mixed_balanced",
        "stratum": stratum_id,
        "semantic_type": row.get("semantic_type", "entity"),
        "selection_status": "reserve_swap",
        "source_feasibility": row.get("source_feasibility", "unknown"),
        "storage_slug": slugify(concept),
        "num_images": 0,
        "description": concept,
        "source_mix_target": dict(target_per_source),
        "source_mix_actual": {source: 0 for source in target_per_source},
        "clip_scores": {},
        "image_sources": {},
    }


def find_reserve(roster: Dict[str, Any], concept: str) -> Tuple[str, Dict[str, Any]]:
    for stratum in roster.get("strata", []):
        for row in stratum.get("reserve_candidates", []):
            if row["concept"] == concept:
                return stratum["id"], row
    raise KeyError(f"Reserve concept not found in roster: {concept}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap blocked active concepts for reserve concepts in the fresh scale250 manifest.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--roster-path", default=DEFAULT_ROSTER_PATH)
    parser.add_argument("--tracker-path", default=DEFAULT_TRACKER_PATH)
    parser.add_argument("--inventory-path", default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--swap", nargs="+", required=True, help="Pairs in old:new form")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest_path)
    roster_path = os.path.abspath(args.roster_path)
    tracker_path = os.path.abspath(args.tracker_path)
    inventory_path = os.path.abspath(args.inventory_path)

    manifest = load_json(manifest_path)
    roster = load_json(roster_path)

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
    swap_notes = {}
    for old_concept, new_concept in swaps:
        if old_concept not in concept_to_images:
            raise KeyError(f"Active concept not found: {old_concept}")
        if new_concept in concept_to_images:
            raise ValueError(f"Reserve concept already present in manifest: {new_concept}")
        old_md = concept_metadata[old_concept]
        stratum_id, reserve_row = find_reserve(roster, new_concept)
        if stratum_id != old_md["stratum"]:
            raise ValueError(f"Stratum mismatch for swap {old_concept} -> {new_concept}: {old_md['stratum']} vs {stratum_id}")
        if old_md.get("num_images", 0) > 0:
            swap_notes[old_concept] = f"Retired on {time.strftime('%Y-%m-%d %H:%M:%S')} and replaced by reserve concept '{new_concept}'."

    new_concept_to_images: Dict[str, List[str]] = {}
    new_concept_metadata: Dict[str, Any] = {}
    for concept in old_concepts:
        replacement = swap_map.get(concept)
        if replacement is None:
            new_concept_to_images[concept] = concept_to_images[concept]
            new_concept_metadata[concept] = concept_metadata[concept]
            continue

        stratum_id, reserve_row = find_reserve(roster, replacement)
        new_concept_to_images[replacement] = []
        new_concept_metadata[replacement] = build_reserve_metadata(manifest, stratum_id, replacement, reserve_row)

    manifest["concept_to_images"] = new_concept_to_images
    manifest["concept_metadata"] = new_concept_metadata

    for row in provenance_rows:
        concept = row.get("concept", "")
        replacement = swap_map.get(concept)
        if replacement is None:
            continue
        if row.get("review_status") == "accepted_auto":
            row["review_status"] = "retired_concept_swap"
            row["rejection_reason"] = ""
        note = row.get("notes", "")
        suffix = f"Retired from active manifest and replaced by '{replacement}'."
        row["notes"] = f"{note} {suffix}".strip()

    atomic_write_json(manifest_path, manifest)
    if provenance_rows:
        atomic_write_csv(provenance_path, provenance_rows)
    regenerate_curator_sheets(manifest_path, inventory_path, tracker_path)

    print(f"Manifest: {manifest_path}")
    print(f"Provenance: {provenance_path}")
    print(f"Tracker: {tracker_path}")
    print(f"Inventory: {inventory_path}")
    print(f"Applied swaps: {swaps}")


if __name__ == "__main__":
    main()
