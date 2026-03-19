import json
import os
import tempfile
from typing import Any, Dict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FULL_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
PILOT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh_pilot30.json")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_merge_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main() -> None:
    full_manifest = load_json(FULL_MANIFEST_PATH)
    pilot_manifest = load_json(PILOT_MANIFEST_PATH)

    full_concept_to_images = full_manifest.setdefault("concept_to_images", {})
    full_metadata = full_manifest.setdefault("concept_metadata", {})

    merged = 0
    completed = 0
    partial = 0

    for concept, pilot_images in pilot_manifest.get("concept_to_images", {}).items():
        if not pilot_images:
            continue

        pilot_md = pilot_manifest["concept_metadata"][concept]
        full_concept_to_images[concept] = list(pilot_images)
        full_metadata[concept].update(
            {
                "num_images": pilot_md.get("num_images", 0),
                "clip_scores": dict(pilot_md.get("clip_scores", {})),
                "image_sources": dict(pilot_md.get("image_sources", {})),
                "source_mix_actual": dict(pilot_md.get("source_mix_actual", {})),
            }
        )

        target = pilot_md.get("source_mix_target", {})
        actual = pilot_md.get("source_mix_actual", {})
        is_complete = all(actual.get(source, 0) >= target.get(source, 0) for source in target)
        if is_complete:
            completed += 1
        else:
            partial += 1
        merged += 1

    atomic_write_json(FULL_MANIFEST_PATH, full_manifest)

    print(f"Merged concepts: {merged}")
    print(f"Completed concepts: {completed}")
    print(f"Partial concepts: {partial}")
    print(f"Updated full manifest: {FULL_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
