import argparse
import json
import os
import tempfile
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_skeleton.json")
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_SIZE = (224, 224)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_clip_backfill_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def clip_score(model: CLIPModel, processor: CLIPProcessor, image: Image.Image, text: str) -> float:
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return float(outputs.logits_per_image.item() / 100.0)


def prompt_for_concept(concept: str, metadata: Dict[str, Any]) -> str:
    description = str(metadata.get("description") or concept).strip()
    return f"a photo of {description}"


def score_manifest_images(
    manifest: Dict[str, Any],
    concepts: List[str] | None,
    overwrite: bool,
    model_id: str,
) -> List[Dict[str, Any]]:
    concept_to_images = manifest.setdefault("concept_to_images", {})
    concept_metadata = manifest.setdefault("concept_metadata", {})

    concept_names = list(concept_to_images.keys())
    if concepts:
        selected = set(concepts)
        missing = sorted(selected - set(concept_names))
        if missing:
            raise ValueError(f"Concepts not found in manifest: {missing}")
        concept_names = [concept for concept in concept_names if concept in selected]

    print(f"Loading CLIP model ({model_id})...")
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    updates: List[Dict[str, Any]] = []
    for concept in concept_names:
        md = concept_metadata.setdefault(concept, {})
        clip_scores = md.setdefault("clip_scores", {})
        prompt = prompt_for_concept(concept, md)
        scored = 0
        skipped = 0

        for rel_path in concept_to_images.get(concept, []):
            image_name = os.path.basename(rel_path)
            if image_name in clip_scores and not overwrite:
                skipped += 1
                continue

            abs_path = os.path.join(REPO_ROOT, rel_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Missing image file for scoring: {abs_path}")

            image = Image.open(abs_path).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
            score = clip_score(model, processor, image, prompt)
            clip_scores[image_name] = round(float(score), 4)
            scored += 1

        updates.append(
            {
                "concept": concept,
                "prompt": prompt,
                "scored": scored,
                "skipped": skipped,
                "total_images": len(concept_to_images.get(concept, [])),
            }
        )
    return updates


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing CLIP scores in a manifest.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--concepts", nargs="*", default=None)
    parser.add_argument("--overwrite", type=str, default="false")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest_path)
    manifest = load_json(manifest_path)
    updates = score_manifest_images(
        manifest=manifest,
        concepts=args.concepts,
        overwrite=parse_bool(args.overwrite),
        model_id=args.model_id,
    )

    if args.write:
        atomic_write_json(manifest_path, manifest)

    print(f"Manifest: {manifest_path}")
    print(f"Write changes: {args.write}")
    for row in updates:
        print(
            f"  - {row['concept']}: prompt='{row['prompt']}' "
            f"scored={row['scored']} skipped={row['skipped']} total_images={row['total_images']}"
        )


if __name__ == "__main__":
    main()
