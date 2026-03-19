import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh_pilot30.json")
DEFAULT_OUTPUT_JSON = os.path.join(REPO_ROOT, "results", "summaries", "SCALE250_FRESH_PILOT30_SOURCE_STATUS_2026-03-14.json")
DEFAULT_OUTPUT_MD = os.path.join(REPO_ROOT, "results", "summaries", "SCALE250_FRESH_PILOT30_SOURCE_STATUS_2026-03-14.md")


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
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_report_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_report_", suffix=".md", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def decide_direction(source_fill_rate: float, blocked_concepts: int, source_rates: Dict[str, float]) -> Dict[str, str]:
    systemic_sources = [source for source, rate in source_rates.items() if rate < 0.5]
    if blocked_concepts > 8 or systemic_sources:
        return {
            "direction": "stop_and_fix_retrieval",
            "message": "One source is failing systematically or too many concepts remain blocked. Fix retrieval/QC before bulk sourcing.",
        }
    if blocked_concepts > 3 or source_fill_rate < 0.85:
        return {
            "direction": "adjust_and_rerun_problem_strata",
            "message": "Pilot found concentrated blockers. Swap same-stratum reserves or improve source-specific retrieval before scaling.",
        }
    return {
        "direction": "expand_high_feasibility_bulk",
        "message": "Pilot fill rate is good enough to expand into the remaining high-feasibility concepts first.",
    }


def build_report(manifest: Dict[str, Any], provenance_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    concept_to_images = manifest.get("concept_to_images", {})
    concept_metadata = manifest.get("concept_metadata", {})
    required_sources = list(manifest.get("source_balance_policy", {}).get("required_sources", []))
    target_per_source = manifest.get("source_balance_policy", {}).get("target_per_source", {})

    concept_rows = []
    filled_cells = 0
    total_cells = len(concept_to_images) * len(required_sources)
    source_cell_fill = Counter()
    blocked_concepts = 0

    for concept, images in concept_to_images.items():
        md = concept_metadata.get(concept, {})
        image_sources = md.get("image_sources", {})
        per_source = Counter()
        for relpath in images:
            image_name = os.path.basename(relpath)
            source = image_sources.get(image_name, "")
            if source:
                per_source[source] += 1

        concept_complete = True
        for source in required_sources:
            if per_source.get(source, 0) >= int(target_per_source.get(source, 0)):
                filled_cells += 1
                source_cell_fill[source] += 1
            else:
                concept_complete = False
        if not concept_complete:
            blocked_concepts += 1

        concept_rows.append(
            {
                "concept": concept,
                "stratum": md.get("stratum", ""),
                "source_feasibility": md.get("source_feasibility", ""),
                "per_source": {source: per_source.get(source, 0) for source in required_sources},
                "complete": concept_complete,
            }
        )

    candidate_counts = Counter()
    accepted_counts = Counter()
    rejection_counts = Counter()
    for row in provenance_rows:
        source = row.get("source", "")
        status = row.get("review_status", "")
        key = f"{row.get('concept','')}::{source}"
        candidate_counts[key] += 1
        if status == "accepted_auto":
            accepted_counts[key] += 1
        else:
            rejection_counts[row.get("rejection_reason", "unknown")] += 1

    source_rates = {
        source: source_cell_fill[source] / max(1, len(concept_to_images))
        for source in required_sources
    }
    source_fill_rate = filled_cells / max(1, total_cells)
    decision = decide_direction(source_fill_rate, blocked_concepts, source_rates)

    worst_concepts = [
        row for row in concept_rows
        if not row["complete"]
    ]
    worst_concepts.sort(
        key=lambda row: (
            sum(row["per_source"].values()),
            row["source_feasibility"] != "medium",
            row["concept"],
        )
    )

    return {
        "concept_count": len(concept_to_images),
        "required_sources": required_sources,
        "target_per_source": target_per_source,
        "filled_cells": filled_cells,
        "total_cells": total_cells,
        "source_fill_rate": round(source_fill_rate, 3),
        "source_cell_fill": dict(source_cell_fill),
        "source_rates": {key: round(value, 3) for key, value in source_rates.items()},
        "blocked_concepts": blocked_concepts,
        "decision": decision,
        "rejection_counts": dict(rejection_counts),
        "worst_concepts": worst_concepts[:15],
        "candidate_counts": dict(candidate_counts),
        "accepted_counts": dict(accepted_counts),
    }


def render_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Scale250 Fresh Sourcing Status (2026-03-14)",
        "",
        f"- Concepts: `{report['concept_count']}`",
        f"- Source cells filled: `{report['filled_cells']}` / `{report['total_cells']}`",
        f"- Source fill rate: `{report['source_fill_rate']:.3f}`",
        f"- Blocked concepts: `{report['blocked_concepts']}`",
        f"- Decision: `{report['decision']['direction']}`",
        f"- Direction: {report['decision']['message']}",
        "",
        "## Source Rates",
        "",
    ]
    for source, rate in report["source_rates"].items():
        lines.append(f"- `{source}`: `{rate:.3f}`")

    lines.extend(
        [
            "",
            "## Rejections",
            "",
        ]
    )
    if report["rejection_counts"]:
        for reason, count in sorted(report["rejection_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Worst Concepts",
            "",
            "| Concept | Stratum | Feasibility | ImageNet | OpenImages | Unsplash | Complete |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["worst_concepts"]:
        lines.append(
            f"| {row['concept']} | {row['stratum']} | {row['source_feasibility']} | "
            f"{row['per_source'].get('imagenet', 0)} | {row['per_source'].get('openimages', 0)} | {row['per_source'].get('unsplash', 0)} | "
            f"{'yes' if row['complete'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a sourcing status report from manifest and provenance.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    manifest = load_json(os.path.abspath(args.manifest_path))
    provenance_rel = manifest.get("shadow_build", {}).get("provenance_ledger", "data/scale250_fresh_provenance.csv")
    provenance_rows = load_csv(os.path.join(REPO_ROOT, provenance_rel))
    report = build_report(manifest, provenance_rows)

    atomic_write_json(os.path.abspath(args.output_json), report)
    atomic_write_text(os.path.abspath(args.output_md), render_md(report))

    print(f"Report JSON: {os.path.abspath(args.output_json)}")
    print(f"Report MD: {os.path.abspath(args.output_md)}")
    print(f"Decision: {report['decision']['direction']}")
    print(f"Source fill rate: {report['source_fill_rate']}")
    print(f"Blocked concepts: {report['blocked_concepts']}")


if __name__ == "__main__":
    main()
