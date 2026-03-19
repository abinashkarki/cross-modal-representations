import importlib.util
import json
import os
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_MODULE_PATH = os.path.join(SCRIPT_DIR, "source_scale250_manifest.py")
DEFAULT_ROSTER_PATH = os.path.join(REPO_ROOT, "data", "concept_roster_250_scaffold.json")
DEFAULT_OUTPUT_JSON = os.path.join(REPO_ROOT, "results", "summaries", "SCALE250_RESERVE_FEASIBILITY_2026-03-14.json")
DEFAULT_OUTPUT_MD = os.path.join(REPO_ROOT, "results", "summaries", "SCALE250_RESERVE_FEASIBILITY_2026-03-14.md")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_reserve_", suffix=".json", dir=os.path.dirname(path))
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
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_reserve_", suffix=".md", dir=os.path.dirname(path))
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


def render_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Scale250 Reserve Feasibility Audit (2026-03-14)",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in report["status_counts"].items():
        lines.append(f"- `{key}`: `{value}`")

    by_status = defaultdict(list)
    for row in report["reserve_rows"]:
        by_status[row["status"]].append(row)

    for status in ["bulk_ready", "needs_openimages_fix", "needs_imagenet_swap_or_alias", "needs_swap"]:
        lines.extend(
            [
                "",
                f"## {status}",
                "",
            ]
        )
        rows = by_status.get(status, [])
        if not rows:
            lines.append("- None")
            continue
        for row in rows:
            preview = ", ".join(row["imagenet_label_preview"]) if row["imagenet_label_preview"] else "no defensible ImageNet labels"
            lines.append(
                f"- `{row['concept']}` ({row['stratum']}, {row['source_feasibility']}): "
                f"ImageNet preview: {preview}; "
                f"OpenImages detection={row['has_openimages_detection']}, labels={row['has_openimages_image_labels']}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    source_mod = load_source_module()
    roster = load_json(DEFAULT_ROSTER_PATH)
    imagenet_labels, imagenet_synsets = source_mod.load_imagenet_metadata()
    openimages_classes = source_mod.load_openimages_classes()
    openimages_label_catalog = source_mod.load_openimages_label_catalog()

    reserve_rows: List[Dict[str, Any]] = []
    counts = Counter()

    for stratum in roster.get("strata", []):
        for item in stratum.get("reserve_candidates", []):
            concept = item["concept"]
            label_ids = source_mod.select_imagenet_label_ids(concept, imagenet_labels, imagenet_synsets)
            has_imagenet = bool(label_ids)
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

            row = {
                "concept": concept,
                "stratum": stratum["id"],
                "source_feasibility": item.get("source_feasibility", ""),
                "has_imagenet": has_imagenet,
                "has_openimages_detection": has_oi_detection,
                "has_openimages_image_labels": has_oi_labels,
                "status": status,
                "imagenet_label_preview": [imagenet_labels[idx] for idx in label_ids[:6]],
            }
            reserve_rows.append(row)
            counts[status] += 1

    reserve_rows.sort(key=lambda row: (row["status"], row["stratum"], row["concept"]))
    report = {
        "status_counts": dict(counts),
        "reserve_rows": reserve_rows,
    }

    atomic_write_json(DEFAULT_OUTPUT_JSON, report)
    atomic_write_text(DEFAULT_OUTPUT_MD, render_md(report))

    print(f"Report JSON: {DEFAULT_OUTPUT_JSON}")
    print(f"Report MD: {DEFAULT_OUTPUT_MD}")
    print(f"Status counts: {dict(counts)}")


if __name__ == "__main__":
    main()
