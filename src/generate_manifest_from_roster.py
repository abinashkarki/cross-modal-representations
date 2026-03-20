import argparse
import json
import os
import re
from typing import Any, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_ROSTER_PATH = os.path.join(REPO_ROOT, "data", "concept_roster_250_scaffold.json")
DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_skeleton.json")


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


def placeholder_paths(concept: str, image_root: str, count: int) -> List[str]:
    slug = slugify(concept)
    return [os.path.join(image_root, slug, f"{slug}_{idx:03d}.jpg") for idx in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a manifest skeleton from a concept roster.")
    parser.add_argument("--roster-path", default=DEFAULT_ROSTER_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--image-root", default="data/images_250_fresh")
    parser.add_argument(
        "--populate-placeholder-paths",
        type=str,
        default="false",
        help="Populate concept_to_images with expected placeholder file paths (true|false).",
    )
    args = parser.parse_args()

    populate_placeholders = str(args.populate_placeholder_paths).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    roster = load_json(os.path.abspath(args.roster_path))
    strata = roster.get("strata", [])
    if not strata:
        raise ValueError("Roster missing strata.")

    balance = roster.get("source_balance_policy", {})
    target_per_source = balance.get("target_per_source", {})
    total_target = int(sum(int(v) for v in target_per_source.values()))

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
            slug = slugify(concept)
            concept_to_images[concept] = (
                placeholder_paths(concept, args.image_root, total_target) if populate_placeholders else []
            )
            concept_metadata[concept] = {
                "source": "mixed_balanced",
                "stratum": stratum_id,
                "semantic_type": row.get("semantic_type", "entity"),
                "selection_status": "planned_core",
                "source_feasibility": row.get("source_feasibility", "unknown"),
                "storage_slug": slug,
                "num_images": len(concept_to_images[concept]),
                "description": concept,
                "source_mix_target": dict(target_per_source),
                "source_mix_actual": {source: 0 for source in target_per_source},
                "clip_scores": {},
                "image_sources": {},
            }

    manifest = {
        "manifest_version": "3.0.0",
        "description": (
            "Manifest skeleton generated from the 250-concept roster scaffold. "
            "Populate concept_to_images, clip_scores, and image_sources before running extraction."
        ),
        "manifest_status": (
            "skeleton_with_placeholder_paths" if populate_placeholders else "skeleton_empty_image_lists"
        ),
        "images_per_concept_target": total_target,
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
            "roster_source": os.path.relpath(os.path.abspath(args.roster_path), REPO_ROOT),
            "reserve_pool_per_stratum": max(
                len(stratum.get("reserve_candidates", [])) for stratum in strata
            ),
            "include_compounds_in_primary_set": False,
            "strata_target_counts": strata_target_counts,
        },
        "analysis_contract": {
            "primary_source_holdout_mode": "per_image_source",
            "primary_source_holdout_field": "concept_metadata.*.image_sources",
            "concept_level_source_field_status": "deprecated_for_balanced_runs",
            "prompt_protocol": "extract baseline3; analyze prompt range or preregistered ensemble",
        },
        "concept_to_images": concept_to_images,
        "concept_metadata": concept_metadata,
    }

    atomic_write_json(os.path.abspath(args.output_path), manifest)
    print(f"Wrote manifest skeleton -> {os.path.abspath(args.output_path)}")
    print(f"Core concept count: {len(concept_to_images)}")
    print(f"Images per concept target: {total_target}")


if __name__ == "__main__":
    main()
