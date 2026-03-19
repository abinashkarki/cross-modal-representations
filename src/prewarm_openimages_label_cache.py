import argparse
import importlib.util
import json
import os
from typing import List


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


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def concept_list_from_manifest(manifest_path: str, only_empty: bool) -> List[str]:
    manifest = load_manifest(manifest_path)
    concepts = []
    for concept, md in manifest.get("concept_metadata", {}).items():
        if only_empty and md.get("num_images", 0) != 0:
            continue
        concepts.append(concept)
    return concepts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prewarm OpenImages image-label cache for selected concepts.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--only-empty", action="store_true")
    parser.add_argument("--concepts", nargs="*")
    args = parser.parse_args()

    source_mod = load_source_module()
    label_catalog = source_mod.load_openimages_label_catalog()

    concepts = args.concepts or concept_list_from_manifest(os.path.abspath(args.manifest_path), args.only_empty)
    label_ids = set()
    concept_to_labels = {}
    for concept in concepts:
        selected = source_mod.select_openimages_image_labels(concept, label_catalog)
        if not selected:
            continue
        concept_to_labels[concept] = [name for _, name in selected]
        for class_id, _ in selected:
            label_ids.add(class_id)

    cache = source_mod.ensure_openimages_label_hit_cache(sorted(label_ids))
    labels_cache = cache.get("labels", {})
    print(f"Concepts with image-label mappings: {len(concept_to_labels)}")
    print(f"Label ids cached: {len(label_ids)}")
    for concept in concepts:
        names = concept_to_labels.get(concept, [])
        hit_count = 0
        for class_id, _ in source_mod.select_openimages_image_labels(concept, label_catalog):
            hit_count += sum(len(ids) for ids in labels_cache.get(class_id, {}).values())
        print(f"{concept}: labels={names} cached_hits={hit_count}")


if __name__ == "__main__":
    main()
