import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_ROSTER_PATH = os.path.join(REPO_ROOT, "data", "concept_roster_250_scaffold.json")
DEFAULT_FRESH_MANIFEST = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
DEFAULT_OUTPUT_MANIFEST = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh_pilot30.json")
DEFAULT_SELECTION_CSV = os.path.join(REPO_ROOT, "data", "scale250_fresh_pilot30_selection.csv")
DEFAULT_INVENTORY_CSV = os.path.join(REPO_ROOT, "data", "scale250_fresh_pilot30_curation_inventory.csv")
DEFAULT_TRACKER_CSV = os.path.join(REPO_ROOT, "data", "scale250_fresh_pilot30_concept_tracker.csv")
DEFAULT_SUMMARY_JSON = os.path.join(REPO_ROOT, "results", "summaries", "SCALE250_FRESH_PILOT30_2026-03-14.json")
DEFAULT_SUMMARY_MD = os.path.join(REPO_ROOT, "results", "summaries", "SCALE250_FRESH_PILOT30_2026-03-14.md")


SELECTION_FIELDS = [
    "stratum",
    "concept",
    "semantic_type",
    "source_feasibility",
    "pilot_role",
    "selection_reason",
    "storage_slug",
    "reserve_candidates",
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_pilot_", suffix=".json", dir=os.path.dirname(path))
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
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_pilot_", suffix=".csv", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_pilot_", suffix=".md", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def medium_target_for_stratum(core_rows: List[Dict[str, Any]]) -> int:
    medium_count = sum(1 for row in core_rows if row.get("source_feasibility") == "medium")
    if medium_count == 0:
        return 0
    if medium_count / max(1, len(core_rows)) >= 0.4:
        return min(2, medium_count)
    return 1


def select_pilot_concepts(roster: Dict[str, Any], fresh_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    concept_metadata = fresh_manifest.get("concept_metadata", {})
    selected_rows: List[Dict[str, Any]] = []

    for stratum in roster.get("strata", []):
        stratum_id = stratum["id"]
        core_rows = list(stratum.get("core_candidates", []))
        reserve_rows = list(stratum.get("reserve_candidates", []))
        medium_rows = [row for row in core_rows if row.get("source_feasibility") == "medium"]
        high_rows = [row for row in core_rows if row.get("source_feasibility") != "medium"]
        target_medium = medium_target_for_stratum(core_rows)

        picked: List[Dict[str, Any]] = []
        for row in medium_rows[:target_medium]:
            picked.append(
                {
                    "stratum": stratum_id,
                    "concept": row["concept"],
                    "semantic_type": row.get("semantic_type", "entity"),
                    "source_feasibility": row.get("source_feasibility", "unknown"),
                    "pilot_role": "stress_medium",
                    "selection_reason": "Pilot stresses medium-feasibility retrieval risk in this stratum.",
                    "storage_slug": concept_metadata[row["concept"]]["storage_slug"],
                    "reserve_candidates": "; ".join(candidate["concept"] for candidate in reserve_rows[:2]),
                }
            )

        high_slots = 3 - len(picked)
        for row in high_rows[:high_slots]:
            picked.append(
                {
                    "stratum": stratum_id,
                    "concept": row["concept"],
                    "semantic_type": row.get("semantic_type", "entity"),
                    "source_feasibility": row.get("source_feasibility", "unknown"),
                    "pilot_role": "anchor_high",
                    "selection_reason": "Pilot keeps a high-feasibility anchor in the same stratum.",
                    "storage_slug": concept_metadata[row["concept"]]["storage_slug"],
                    "reserve_candidates": "; ".join(candidate["concept"] for candidate in reserve_rows[:2]),
                }
            )

        if len(picked) < 3:
            remaining_medium = [row for row in medium_rows if row["concept"] not in {item["concept"] for item in picked}]
            for row in remaining_medium[: 3 - len(picked)]:
                picked.append(
                    {
                        "stratum": stratum_id,
                        "concept": row["concept"],
                        "semantic_type": row.get("semantic_type", "entity"),
                        "source_feasibility": row.get("source_feasibility", "unknown"),
                        "pilot_role": "stress_medium",
                        "selection_reason": "Pilot filled remaining stratum slots with the next medium-feasibility concept.",
                        "storage_slug": concept_metadata[row["concept"]]["storage_slug"],
                        "reserve_candidates": "; ".join(candidate["concept"] for candidate in reserve_rows[:2]),
                    }
                )

        if len(picked) != 3:
            raise ValueError(f"Expected 3 pilot concepts for stratum '{stratum_id}', got {len(picked)}")
        selected_rows.extend(picked)

    if len(selected_rows) != 30:
        raise ValueError(f"Expected 30 total pilot concepts, got {len(selected_rows)}")
    return selected_rows


def subset_manifest(fresh_manifest: Dict[str, Any], selected_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected_concepts = [row["concept"] for row in selected_rows]
    concept_to_images = fresh_manifest.get("concept_to_images", {})
    concept_metadata = fresh_manifest.get("concept_metadata", {})
    output = dict(fresh_manifest)
    output["manifest_status"] = "fresh_pilot30_initialized"
    output["description"] = (
        "Thirty-concept feasibility pilot derived from the fresh 250-concept shadow build. "
        "This manifest is for retrieval and curation validation before bulk sourcing."
    )
    output["pilot_scope"] = {
        "concept_count": len(selected_concepts),
        "source_cells": len(selected_concepts) * 3,
        "target_images": len(selected_concepts) * int(fresh_manifest["images_per_concept_target"]),
        "selection_strategy": "3 per stratum with medium-feasibility stress where available",
        "selected_concepts": selected_concepts,
    }
    output["concept_to_images"] = {concept: list(concept_to_images.get(concept, [])) for concept in selected_concepts}
    output["concept_metadata"] = {concept: dict(concept_metadata.get(concept, {})) for concept in selected_concepts}
    return output


def create_pilot_dirs(manifest: Dict[str, Any], pilot_manifest: Dict[str, Any]) -> None:
    shadow = manifest.get("shadow_build", {})
    image_root_rel = shadow.get("active_image_root")
    candidate_root_rel = shadow.get("candidate_root")
    if not image_root_rel or not candidate_root_rel:
        raise ValueError("Fresh manifest missing shadow_build roots.")

    image_root = os.path.join(REPO_ROOT, image_root_rel)
    candidate_root = os.path.join(REPO_ROOT, candidate_root_rel)
    for concept, metadata in pilot_manifest.get("concept_metadata", {}).items():
        storage_slug = metadata["storage_slug"]
        os.makedirs(os.path.join(image_root, storage_slug), exist_ok=True)
        for source in ["imagenet", "openimages", "unsplash"]:
            os.makedirs(os.path.join(candidate_root, source, storage_slug), exist_ok=True)


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


def direction_from_mix(selected_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    medium_count = sum(1 for row in selected_rows if row["source_feasibility"] == "medium")
    medium_share = medium_count / max(1, len(selected_rows))
    double_medium_strata = [
        stratum for stratum, count in Counter(
            row["stratum"] for row in selected_rows if row["source_feasibility"] == "medium"
        ).items()
        if count >= 2
    ]
    if medium_share >= 0.35 or double_medium_strata:
        return {
            "direction_now": "Run a blocker-first pilot before any bulk sourcing.",
            "why": (
                "The pilot is deliberately medium-heavy, so the main goal is to test source feasibility and reserve-swap pressure "
                "before scaling the rest of the roster."
            ),
            "go_rule": "If at least 85% of pilot source cells fill cleanly and no more than 3 pilot concepts are blocked, expand to the remaining high-feasibility concepts first.",
            "adjust_rule": "If 4-8 concepts block or failures cluster in 1-2 strata, activate same-stratum reserves there and rerun only those strata before scaling.",
            "stop_rule": "If more than 8 concepts block or one source fails systematically across strata, fix retrieval/QC rules before bulk collection.",
        }
    if medium_share >= 0.2:
        return {
            "direction_now": "Run the pilot, then bulk-source high-feasibility concepts if early yield is clean.",
            "why": "The pilot contains enough medium-feasibility stress to catch problems without overloading the first batch.",
            "go_rule": "If at least 90% of pilot source cells fill cleanly, move straight to high-feasibility bulk sourcing.",
            "adjust_rule": "If a few strata misbehave, isolate them and continue with the rest.",
            "stop_rule": "If failures are broad rather than stratum-specific, revise the acquisition protocol first.",
        }
    return {
        "direction_now": "Pilot is light-risk; use it mainly as a process smoke test.",
        "why": "The pilot mix is mostly high-feasibility and should validate throughput more than concept risk.",
        "go_rule": "If the pilot runs cleanly, bulk-source immediately.",
        "adjust_rule": "If a few concepts fail, swap reserves and continue.",
        "stop_rule": "Only pause if a whole source pipeline is unstable.",
    }


def build_summary_payload(selected_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    feasibility_counts = Counter(row["source_feasibility"] for row in selected_rows)
    stratum_counts: Dict[str, Dict[str, Any]] = {}
    for row in selected_rows:
        summary = stratum_counts.setdefault(
            row["stratum"],
            {"concepts": [], "high": 0, "medium": 0},
        )
        summary["concepts"].append(
            {
                "concept": row["concept"],
                "source_feasibility": row["source_feasibility"],
                "pilot_role": row["pilot_role"],
            }
        )
        summary[row["source_feasibility"]] += 1

    direction = direction_from_mix(selected_rows)
    return {
        "pilot_concept_count": len(selected_rows),
        "pilot_source_cells": len(selected_rows) * 3,
        "pilot_target_images": len(selected_rows) * 15,
        "feasibility_counts": dict(feasibility_counts),
        "medium_share": round(feasibility_counts.get("medium", 0) / max(1, len(selected_rows)), 3),
        "stratum_summary": stratum_counts,
        "direction": direction,
    }


def render_summary_md(payload: Dict[str, Any], selection_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Scale250 Fresh Pilot30 (2026-03-14)",
        "",
        "This pilot selects 3 concepts per stratum from the fresh 250-concept shadow build.",
        "",
        "## Pilot Mix",
        "",
        f"- Pilot concepts: `{payload['pilot_concept_count']}`",
        f"- Source cells: `{payload['pilot_source_cells']}`",
        f"- Target images: `{payload['pilot_target_images']}`",
        f"- High-feasibility concepts: `{payload['feasibility_counts'].get('high', 0)}`",
        f"- Medium-feasibility concepts: `{payload['feasibility_counts'].get('medium', 0)}`",
        f"- Medium share: `{payload['medium_share']:.3f}`",
        "",
        "## By Stratum",
        "",
    ]
    for stratum, summary in payload["stratum_summary"].items():
        concept_bits = ", ".join(
            f"{row['concept']} ({row['source_feasibility']}, {row['pilot_role']})"
            for row in summary["concepts"]
        )
        lines.append(
            f"- `{stratum}`: {concept_bits}"
        )

    direction = payload["direction"]
    lines.extend(
        [
            "",
            "## Direction",
            "",
            f"- Current direction: {direction['direction_now']}",
            f"- Why: {direction['why']}",
            f"- Go rule: {direction['go_rule']}",
            f"- Adjust rule: {direction['adjust_rule']}",
            f"- Stop rule: {direction['stop_rule']}",
            "",
            "## Selected Concepts",
            "",
            "| Stratum | Concept | Feasibility | Role | Reserve hints |",
            "|---|---|---|---|---|",
        ]
    )
    for row in selection_rows:
        lines.append(
            f"| {row['stratum']} | {row['concept']} | {row['source_feasibility']} | {row['pilot_role']} | {row['reserve_candidates']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the 30-concept scale250 feasibility pilot.")
    parser.add_argument("--roster-path", default=DEFAULT_ROSTER_PATH)
    parser.add_argument("--fresh-manifest", default=DEFAULT_FRESH_MANIFEST)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--selection-csv", default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--inventory-output", default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--tracker-output", default=DEFAULT_TRACKER_CSV)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", default=DEFAULT_SUMMARY_MD)
    args = parser.parse_args()

    roster = load_json(os.path.abspath(args.roster_path))
    fresh_manifest = load_json(os.path.abspath(args.fresh_manifest))
    selected_rows = select_pilot_concepts(roster, fresh_manifest)
    pilot_manifest = subset_manifest(fresh_manifest, selected_rows)

    output_manifest = os.path.abspath(args.output_manifest)
    selection_csv = os.path.abspath(args.selection_csv)
    inventory_output = os.path.abspath(args.inventory_output)
    tracker_output = os.path.abspath(args.tracker_output)
    summary_json = os.path.abspath(args.summary_json)
    summary_md = os.path.abspath(args.summary_md)

    create_pilot_dirs(fresh_manifest, pilot_manifest)
    atomic_write_json(output_manifest, pilot_manifest)
    atomic_write_csv(selection_csv, SELECTION_FIELDS, selected_rows)
    generate_curator_sheets(output_manifest, inventory_output, tracker_output)

    payload = build_summary_payload(selected_rows)
    atomic_write_json(summary_json, payload)
    atomic_write_text(summary_md, render_summary_md(payload, selected_rows))

    print(f"Pilot manifest: {output_manifest}")
    print(f"Pilot selection CSV: {selection_csv}")
    print(f"Pilot inventory: {inventory_output}")
    print(f"Pilot tracker: {tracker_output}")
    print(f"Pilot summary JSON: {summary_json}")
    print(f"Pilot summary MD: {summary_md}")
    print(f"Pilot concepts: {payload['pilot_concept_count']}")
    print(f"High feasibility: {payload['feasibility_counts'].get('high', 0)}")
    print(f"Medium feasibility: {payload['feasibility_counts'].get('medium', 0)}")
    print(f"Direction: {payload['direction']['direction_now']}")


if __name__ == "__main__":
    main()
