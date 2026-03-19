import argparse
import csv
import importlib.util
import json
import os
import tempfile
import time
from collections import Counter
from typing import Any, Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_MODULE_PATH = os.path.join(SCRIPT_DIR, "source_scale250_manifest.py")
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")


def load_source_module():
    spec = importlib.util.spec_from_file_location("source_scale250_manifest", SOURCE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_rollback_", suffix=".json", dir=os.path.dirname(path))
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
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_rollback_", suffix=".csv", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollback invalid ImageNet accepts from the fresh build.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_mod = load_source_module()
    manifest_path = os.path.abspath(args.manifest_path)
    manifest = load_json(manifest_path)
    provenance_rel = manifest.get("shadow_build", {}).get("provenance_ledger", "data/scale250_fresh_provenance.csv")
    provenance_path = os.path.join(REPO_ROOT, provenance_rel)
    provenance_rows = load_csv(provenance_path)

    imagenet_synsets = source_mod.load_imagenet_synsets()
    concept_to_images = manifest.setdefault("concept_to_images", {})
    concept_metadata = manifest.setdefault("concept_metadata", {})
    rollback_counts = Counter()

    for row in provenance_rows:
        if row.get("source") != "imagenet" or row.get("review_status") != "accepted_auto":
            continue
        concept = row.get("concept", "")
        class_id_text = row.get("source_class_id", "")
        if not class_id_text:
            continue
        label_synset = imagenet_synsets[int(class_id_text)]
        if source_mod.imagenet_label_matches_concept(concept, row.get("source_class_label", ""), label_synset):
            continue

        accepted_relpath = row.get("accepted_relpath", "")
        accepted_filename = row.get("accepted_filename", "")
        accepted_abs = os.path.join(REPO_ROOT, accepted_relpath) if accepted_relpath else ""

        if accepted_relpath in concept_to_images.get(concept, []):
            concept_to_images[concept] = [path for path in concept_to_images[concept] if path != accepted_relpath]

        md = concept_metadata.get(concept, {})
        if accepted_filename:
            md.get("clip_scores", {}).pop(accepted_filename, None)
            md.get("image_sources", {}).pop(accepted_filename, None)

        if accepted_abs and os.path.exists(accepted_abs) and not args.dry_run:
            os.remove(accepted_abs)

        row["review_status"] = "rolled_back_invalid_semantics"
        row["rejection_reason"] = "imagenet_semantic_mismatch"
        row["notes"] = f"Rolled back on {time.strftime('%Y-%m-%d %H:%M:%S')} after WordNet semantic audit."
        rollback_counts[concept] += 1

    for concept, md in concept_metadata.items():
        concept_to_images[concept] = sorted(concept_to_images.get(concept, []), key=os.path.basename)
        md["num_images"] = len(concept_to_images.get(concept, []))
        md["source_mix_actual"] = {
            source: source_mod.existing_source_count(md, source)
            for source in md.get("source_mix_target", {})
        }

    if not args.dry_run:
        atomic_write_json(manifest_path, manifest)
        atomic_write_csv(provenance_path, provenance_rows)

    total = sum(rollback_counts.values())
    print(f"Manifest: {manifest_path}")
    print(f"Provenance: {provenance_path}")
    print(f"Rolled back accepted ImageNet rows: {total}")
    print(f"Affected concepts: {dict(sorted(rollback_counts.items()))}")


if __name__ == "__main__":
    main()
