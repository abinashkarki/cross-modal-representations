import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_ROSTER_PATH = os.path.join(REPO_ROOT, "data", "concept_roster_250_scaffold.json")
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
DEFAULT_IMAGE_ROOT = os.path.join(REPO_ROOT, "data", "images_250_fresh")
DEFAULT_CANDIDATE_ROOT = os.path.join(REPO_ROOT, "data", "scale250_fresh_candidates")
DEFAULT_PROVENANCE_PATH = os.path.join(REPO_ROOT, "data", "scale250_fresh_provenance.csv")
DEFAULT_INVENTORY_PATH = os.path.join(REPO_ROOT, "data", "scale250_fresh_curation_inventory.csv")
DEFAULT_TRACKER_PATH = os.path.join(REPO_ROOT, "data", "scale250_fresh_concept_tracker.csv")


PROVENANCE_FIELDS = [
    "concept",
    "stratum",
    "semantic_type",
    "selection_status",
    "source_feasibility",
    "storage_slug",
    "source",
    "source_slot",
    "candidate_filename",
    "accepted_filename",
    "candidate_relpath",
    "accepted_relpath",
    "candidate_record_id",
    "candidate_source_url",
    "acquisition_method",
    "acquisition_query",
    "source_class_label",
    "source_class_id",
    "proxy_used",
    "clip_score",
    "review_status",
    "rejection_reason",
    "diversity_notes",
    "license_or_terms",
    "curator_initials",
    "reviewed_at",
    "notes",
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_fresh_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def atomic_write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_fresh_", suffix=".csv", dir=os.path.dirname(path))
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


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def ensure_missing(path: str, force: bool) -> None:
    if os.path.exists(path) and not force:
        raise FileExistsError(f"Refusing to overwrite existing path without --force: {path}")


def build_manifest(
    roster: Dict[str, Any],
    roster_source_rel: str,
    image_root_rel: str,
    candidate_root_rel: str,
    provenance_rel: str,
) -> Dict[str, Any]:
    strata = roster.get("strata", [])
    if not strata:
        raise ValueError("Roster missing strata.")

    balance = roster.get("source_balance_policy", {})
    target_per_source = {source: int(count) for source, count in balance.get("target_per_source", {}).items()}
    if not target_per_source:
        raise ValueError("Roster missing source_balance_policy.target_per_source.")

    concept_to_images: Dict[str, List[str]] = {}
    concept_metadata: Dict[str, Any] = {}
    strata_target_counts: Dict[str, int] = {}
    seen = set()

    for stratum in strata:
        stratum_id = stratum["id"]
        core = stratum.get("core_candidates", [])
        target_count = int(stratum.get("target_count", len(core)))
        if len(core) != target_count:
            raise ValueError(
                f"Stratum '{stratum_id}' target_count={target_count} but core_candidates={len(core)}."
            )
        strata_target_counts[stratum_id] = target_count

        for row in core:
            concept = row["concept"]
            if concept in seen:
                raise ValueError(f"Duplicate concept in roster: {concept}")
            seen.add(concept)
            storage_slug = slugify(concept)
            concept_to_images[concept] = []
            concept_metadata[concept] = {
                "source": "mixed_balanced",
                "stratum": stratum_id,
                "semantic_type": row.get("semantic_type", "entity"),
                "selection_status": "planned_core",
                "source_feasibility": row.get("source_feasibility", "unknown"),
                "storage_slug": storage_slug,
                "num_images": 0,
                "description": concept,
                "source_mix_target": dict(target_per_source),
                "source_mix_actual": {source: 0 for source in target_per_source},
                "clip_scores": {},
                "image_sources": {},
            }

    return {
        "manifest_version": "3.1.0",
        "description": (
            "Fresh shadow-build manifest for the 250-concept balanced benchmark. "
            "This manifest is for clean re-sourcing from scratch and must remain separate from existing curated assets "
            "until exact quotas, provenance, and quality gates all pass."
        ),
        "manifest_status": "fresh_shadow_initialized",
        "images_per_concept_target": int(sum(target_per_source.values())),
        "image_size": [224, 224],
        "sources": {
            "imagenet": "ILSVRC/imagenet-1k validation split via HuggingFace datasets",
            "openimages": "Open Images V7 via FiftyOne",
            "unsplash": "Unsplash API",
        },
        "source_balance_policy": {
            "mode": "within_concept_balanced",
            "required_sources": balance.get("required_sources", list(target_per_source.keys())),
            "target_per_source": dict(target_per_source),
            "minimum_per_source": dict(target_per_source),
            "allow_source_substitution": False,
            "drop_concept_if_unbalanced": True,
        },
        "selection_protocol": {
            "design": "stratified_a_priori",
            "primary_set": "base250",
            "roster_source": roster_source_rel,
            "reserve_pool_per_stratum": max(len(stratum.get("reserve_candidates", [])) for stratum in strata),
            "include_compounds_in_primary_set": False,
            "strata_target_counts": strata_target_counts,
            "replacement_rule": "replace blocked concepts only from reserve list in same stratum",
        },
        "analysis_contract": {
            "primary_source_holdout_mode": "per_image_source",
            "primary_source_holdout_field": "concept_metadata.*.image_sources",
            "concept_level_source_field_status": "deprecated_for_balanced_runs",
            "prompt_protocol": "extract baseline3; analyze prompt range or preregistered ensemble",
        },
        "shadow_build": {
            "keeps_existing_dataset_untouched": True,
            "active_image_root": image_root_rel,
            "candidate_root": candidate_root_rel,
            "provenance_ledger": provenance_rel,
            "acceptance_policy": "download candidate pools first; accept only reviewed 5/5/5 cells",
        },
        "concept_to_images": concept_to_images,
        "concept_metadata": concept_metadata,
    }


def generate_curator_sheets(manifest_path: str, inventory_path: str, tracker_path: str) -> None:
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
    parser = argparse.ArgumentParser(
        description="Initialize a fresh, non-destructive 250-concept shadow build with provenance tracking."
    )
    parser.add_argument("--roster-path", default=DEFAULT_ROSTER_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--candidate-root", default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--provenance-path", default=DEFAULT_PROVENANCE_PATH)
    parser.add_argument("--inventory-output", default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--tracker-output", default=DEFAULT_TRACKER_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    roster_path = resolve_path(args.roster_path)
    manifest_path = resolve_path(args.manifest_path)
    image_root = resolve_path(args.image_root)
    candidate_root = resolve_path(args.candidate_root)
    provenance_path = resolve_path(args.provenance_path)
    inventory_output = resolve_path(args.inventory_output)
    tracker_output = resolve_path(args.tracker_output)

    for path in [manifest_path, provenance_path, inventory_output, tracker_output]:
        ensure_missing(path, args.force)

    roster = load_json(roster_path)
    roster_source_rel = os.path.relpath(roster_path, REPO_ROOT)
    image_root_rel = os.path.relpath(image_root, REPO_ROOT)
    candidate_root_rel = os.path.relpath(candidate_root, REPO_ROOT)
    provenance_rel = os.path.relpath(provenance_path, REPO_ROOT)
    manifest = build_manifest(
        roster,
        roster_source_rel,
        image_root_rel,
        candidate_root_rel,
        provenance_rel,
    )

    os.makedirs(image_root, exist_ok=True)
    for source in ["imagenet", "openimages", "unsplash"]:
        os.makedirs(os.path.join(candidate_root, source), exist_ok=True)

    atomic_write_json(manifest_path, manifest)
    atomic_write_csv(provenance_path, PROVENANCE_FIELDS, [])
    generate_curator_sheets(manifest_path, inventory_output, tracker_output)

    print(f"Roster: {roster_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Fresh image root: {image_root}")
    print(f"Candidate root: {candidate_root}")
    print(f"Provenance ledger: {provenance_path}")
    print(f"Inventory: {inventory_output}")
    print(f"Tracker: {tracker_output}")
    print(f"Concept count: {len(manifest['concept_to_images'])}")
    print(f"Images per concept target: {manifest['images_per_concept_target']}")
    print("Existing image datasets were left untouched.")


if __name__ == "__main__":
    main()
