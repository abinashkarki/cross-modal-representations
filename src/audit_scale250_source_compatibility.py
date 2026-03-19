import importlib.util
import json
import os
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_MODULE_PATH = os.path.join(SCRIPT_DIR, "source_scale250_manifest.py")
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
DEFAULT_PILOT_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh_pilot30.json")
DEFAULT_ROSTER_PATH = os.path.join(REPO_ROOT, "data", "concept_roster_250_scaffold.json")
DEFAULT_OUTPUT_JSON = os.path.join(REPO_ROOT, "results", "summaries", "SCALE250_SOURCE_COMPATIBILITY_2026-03-14.json")
DEFAULT_OUTPUT_MD = os.path.join(REPO_ROOT, "results", "summaries", "SCALE250_SOURCE_COMPATIBILITY_2026-03-14.md")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_compat_", suffix=".json", dir=os.path.dirname(path))
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
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_compat_", suffix=".md", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_source_module():
    spec = importlib.util.spec_from_file_location("source_scale250_manifest", SOURCE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def classify_concepts(
    source_mod,
    manifest: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Counter]:
    imagenet_labels, imagenet_synsets = source_mod.load_imagenet_metadata()
    openimages_classes = source_mod.load_openimages_classes()
    openimages_label_catalog = source_mod.load_openimages_label_catalog()

    rows: List[Dict[str, Any]] = []
    counts = Counter()

    for concept, md in manifest.get("concept_metadata", {}).items():
        has_imagenet = bool(source_mod.select_imagenet_label_ids(concept, imagenet_labels, imagenet_synsets))
        has_oi_detection = bool(source_mod.select_openimages_classes(concept, openimages_classes))
        has_oi_labels = bool(source_mod.select_openimages_image_labels(concept, openimages_label_catalog))
        has_openimages = has_oi_detection or has_oi_labels

        if has_imagenet and has_openimages:
            status = "bulk_ready"
        elif has_imagenet and not has_openimages:
            status = "needs_openimages_fix"
        elif not has_imagenet and has_openimages:
            status = "needs_imagenet_swap_or_alias"
        else:
            status = "needs_swap"

        rows.append(
            {
                "concept": concept,
                "stratum": md.get("stratum", ""),
                "source_feasibility": md.get("source_feasibility", ""),
                "has_imagenet": has_imagenet,
                "has_openimages_detection": has_oi_detection,
                "has_openimages_image_labels": has_oi_labels,
                "status": status,
            }
        )
        counts[status] += 1

    rows.sort(key=lambda row: (row["status"], row["stratum"], row["concept"]))
    return rows, counts


def suggest_replacements(
    roster: Dict[str, Any],
    pilot_manifest: Dict[str, Any],
    compat_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    compat_map = {row["concept"]: row for row in compat_rows}
    pilot_concepts = set(pilot_manifest.get("concept_to_images", {}))
    blocked = []
    for concept, md in pilot_manifest.get("concept_metadata", {}).items():
        actual = md.get("source_mix_actual", {})
        target = md.get("source_mix_target", {})
        if not all(actual.get(source, 0) >= target.get(source, 0) for source in target):
            blocked.append((concept, md.get("stratum", ""), md.get("source_feasibility", "")))

    suggestions: List[Dict[str, Any]] = []
    strata = {entry["id"]: entry for entry in roster.get("strata", [])}
    for concept, stratum_id, feasibility in blocked:
        stratum = strata.get(stratum_id, {})
        candidates = []
        for group_name in ("core_candidates", "reserve_candidates"):
            for item in stratum.get(group_name, []):
                replacement = item["concept"]
                if replacement in pilot_concepts or replacement == concept:
                    continue
                compat = compat_map.get(replacement)
                if compat is None or compat["status"] != "bulk_ready":
                    continue
                candidates.append(
                    {
                        "concept": replacement,
                        "group": group_name,
                        "source_feasibility": item.get("source_feasibility", ""),
                    }
                )
        suggestions.append(
            {
                "blocked_concept": concept,
                "stratum": stratum_id,
                "source_feasibility": feasibility,
                "suggested_replacements": candidates[:5],
            }
        )

    return suggestions


def render_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Scale250 Source Compatibility Audit (2026-03-14)",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in report["status_counts"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Pilot Remediation",
            "",
        ]
    )
    for item in report["pilot_replacements"]:
        lines.append(
            f"- `{item['blocked_concept']}` ({item['stratum']}, {item['source_feasibility']}): "
            + (
                ", ".join(
                    f"{candidate['concept']} [{candidate['group']}]"
                    for candidate in item["suggested_replacements"]
                )
                if item["suggested_replacements"]
                else "no same-stratum bulk-ready replacements found"
            )
        )

    status_buckets = defaultdict(list)
    for row in report["concept_rows"]:
        status_buckets[row["status"]].append(row["concept"])

    lines.extend(
        [
            "",
            "## Bulk Ready Sample",
            "",
            "- " + ", ".join(status_buckets["bulk_ready"][:25]),
            "",
            "## Needs ImageNet Swap Or Alias Sample",
            "",
            "- " + ", ".join(status_buckets["needs_imagenet_swap_or_alias"][:25]),
            "",
            "## Needs OpenImages Fix Sample",
            "",
            "- " + ", ".join(status_buckets["needs_openimages_fix"][:25]),
            "",
            "## Needs Swap Sample",
            "",
            "- " + ", ".join(status_buckets["needs_swap"][:25]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    source_mod = load_source_module()
    manifest = load_json(DEFAULT_MANIFEST_PATH)
    pilot_manifest = load_json(DEFAULT_PILOT_PATH)
    roster = load_json(DEFAULT_ROSTER_PATH)

    concept_rows, counts = classify_concepts(source_mod, manifest)
    replacements = suggest_replacements(roster, pilot_manifest, concept_rows)

    report = {
        "status_counts": dict(counts),
        "concept_rows": concept_rows,
        "pilot_replacements": replacements,
    }

    atomic_write_json(DEFAULT_OUTPUT_JSON, report)
    atomic_write_text(DEFAULT_OUTPUT_MD, render_md(report))

    print(f"Report JSON: {DEFAULT_OUTPUT_JSON}")
    print(f"Report MD: {DEFAULT_OUTPUT_MD}")
    print(f"Status counts: {dict(counts)}")


if __name__ == "__main__":
    main()
