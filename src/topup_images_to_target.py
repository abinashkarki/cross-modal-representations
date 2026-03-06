"""
Top up multi-concept image dataset to a target count per concept.

Strategy:
- Keep existing images.
- Download only missing images per concept from Unsplash public search.
- Filter candidates with CLIP image/text similarity.
- Update manifest concept_to_images and concept_metadata.clip_scores/image_sources.

Usage:
  python topup_images_to_target.py --target 30
  python topup_images_to_target.py --target 30 --clip-threshold 0.22
  python topup_images_to_target.py --target 30 --dry-run
"""

import argparse
import json
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
MANIFEST_PATH = os.path.join(EXPERIMENT_DIR, "data", "data_manifest_multi.json")
IMAGE_SIZE = (224, 224)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _atomic_write_json(path: str, payload: Dict) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_manifest_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _existing_images_for_concept(concept_dir: str) -> List[str]:
    if not os.path.isdir(concept_dir):
        return []
    files = [f for f in os.listdir(concept_dir) if f.lower().endswith(IMAGE_EXTS)]
    return sorted(files)


def _next_index(existing_files: List[str], concept: str) -> int:
    pattern = re.compile(rf"^{re.escape(concept)}_(\d+)\.jpg$", re.IGNORECASE)
    max_idx = -1
    for name in existing_files:
        m = pattern.match(name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    if max_idx >= 0:
        return max_idx + 1
    return len(existing_files)


def _prompt_for_concept(concept: str) -> str:
    custom = {
        "fire": "a photo of fire with flames",
        "water": "a photo of water",
        "forest": "a photo of a forest with many trees",
        "ocean": "a photo of the ocean",
        "road": "a photo of a road",
        "city": "a photo of a city with buildings",
        "sun": "a photo of the sun in the sky",
        "forest fire": "a photo of a forest fire with trees burning",
        "space city": "a photo of a futuristic city in space",
        "water city": "a photo of a city waterfront with visible water",
        "city forest": "a photo of trees in a city",
        "mountain road": "a photo of a road going through mountains",
        "ocean bridge": "a photo of a bridge over the ocean or sea",
        "city bridge": "a photo of a bridge with a city skyline",
        "mountain forest": "a photo of a forested mountain",
    }
    return custom.get(concept, f"a photo of {concept}")


def _query_candidates(concept: str) -> List[str]:
    shared = [
        concept,
        f"{concept} photo",
        f"{concept} landscape",
    ]
    custom = {
        "cat": ["cat domestic cat", "cat portrait"],
        "dog": ["dog pet canine", "dog portrait"],
        "bird": ["bird wildlife avian", "bird flying"],
        "fish": ["fish underwater", "tropical fish"],
        "elephant": ["elephant wildlife safari", "elephant herd"],
        "car": ["car automobile vehicle", "car on road"],
        "bridge": ["bridge architecture", "bridge river"],
        "airplane": ["airplane aircraft aviation", "airplane in sky"],
        "mountain": ["mountain landscape peak", "mountain range"],
        "building": ["building architecture", "skyscraper building"],
        "fire": ["fire flames burning", "bonfire flames"],
        "water": ["water river lake", "waterfall stream"],
        "forest": ["forest trees woodland", "forest path"],
        "ocean": ["ocean waves sea", "ocean seascape"],
        "road": ["road highway asphalt", "road landscape"],
        "city": ["city skyline downtown", "cityscape urban"],
        "space": ["outer space galaxy nebula", "space stars"],
        "sun": ["sun in sky", "sunlight sky"],
        "moon": ["moon lunar night", "moon in sky"],
        "storm": ["storm thunderstorm lightning", "storm clouds"],
        "forest fire": ["forest fire wildfire", "wildfire burning trees"],
        "space city": ["futuristic city skyline night neon", "sci fi city"],
        "water city": ["city waterfront river", "city by water"],
        "city forest": ["urban park trees city", "city forest greenery"],
        "mountain road": ["mountain road winding", "road through mountains"],
        "ocean bridge": ["bridge over ocean sea", "coastal bridge"],
        "city bridge": ["bridge city skyline", "urban bridge river"],
        "mountain forest": ["forested mountain landscape", "mountain covered in trees"],
    }
    return custom.get(concept, []) + shared


def _unsplash_search(
    query: str,
    page: int,
    per_page: int,
    access_key: Optional[str],
) -> List[Tuple[str, str]]:
    if access_key:
        params = urllib.parse.urlencode(
            {"query": query, "per_page": per_page, "page": page, "orientation": "squarish"}
        )
        url = f"https://api.unsplash.com/search/photos?{params}"
        headers = {
            "Authorization": f"Client-ID {access_key}",
            "Accept-Version": "v1",
            "User-Agent": "platonic-representation-topup/1.0",
            "Accept": "application/json",
        }
    else:
        params = urllib.parse.urlencode({"query": query, "per_page": per_page, "page": page})
        url = f"https://unsplash.com/napi/search/photos?{params}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        remaining = exc.headers.get("X-Ratelimit-Remaining", "unknown")
        reset = exc.headers.get("X-Ratelimit-Reset", "unknown")
        raise RuntimeError(
            f"HTTP {exc.code} for query='{query}' page={page} "
            f"(rate_remaining={remaining}, rate_reset={reset})"
        ) from exc

    out = []
    for row in payload.get("results", []):
        img_id = row.get("id")
        img_url = row.get("urls", {}).get("regular")
        if img_id and img_url:
            out.append((img_id, img_url))
    return out


def _wikimedia_search(
    query: str,
    limit: int,
    continue_token: Optional[Dict[str, str]],
) -> Tuple[List[Tuple[str, str]], Optional[Dict[str, str]]]:
    params: Dict[str, str] = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": "1280",
    }
    if continue_token:
        params.update(continue_token)
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "platonic-representation-topup/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows: List[Tuple[str, str]] = []
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        imageinfo = page.get("imageinfo") or []
        if not imageinfo:
            continue
        info = imageinfo[0]
        mime = str(info.get("mime", "")).lower()
        if "svg" in mime:
            continue
        img_url = info.get("thumburl") or info.get("url")
        if not img_url:
            continue
        img_id = str(page.get("pageid", img_url))
        rows.append((img_id, img_url))
    return rows, payload.get("continue")


def _download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    return img.resize(IMAGE_SIZE, Image.LANCZOS)


def _clip_score(model: CLIPModel, processor: CLIPProcessor, image: Image.Image, text: str) -> float:
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return float(outputs.logits_per_image.item() / 100.0)


@dataclass
class TopupResult:
    concept: str
    before: int
    added: int
    after: int


def topup_manifest(
    manifest_path: str,
    target: int,
    clip_threshold: float,
    per_page: int,
    max_pages: int,
    unsplash_access_key: Optional[str],
    search_source: str,
    selected_concepts: Optional[List[str]],
    dry_run: bool,
    sleep_s: float,
    allow_partial: bool,
) -> Tuple[List[TopupResult], List[Tuple[str, int]], bool]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    concept_to_images: Dict[str, List[str]] = manifest.get("concept_to_images", {})
    concept_metadata: Dict[str, Dict] = manifest.setdefault("concept_metadata", {})
    if not concept_to_images:
        raise RuntimeError("Manifest missing concept_to_images")

    model = None
    processor = None
    original_target = int(manifest.get("images_per_concept_target", target))

    if not dry_run:
        print("Loading CLIP model (openai/clip-vit-base-patch32)...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

    results: List[TopupResult] = []
    concepts = list(concept_to_images.keys())
    if selected_concepts:
        selected_set = set(selected_concepts)
        concepts = [c for c in concepts if c in selected_set]
        missing = sorted(selected_set - set(concepts))
        if missing:
            raise RuntimeError(f"Selected concepts not found in manifest: {missing}")

    for concept in concepts:
        rel_paths = concept_to_images.get(concept, [])
        md = concept_metadata.setdefault(concept, {})
        clip_scores = md.setdefault("clip_scores", {})
        image_sources = md.setdefault("image_sources", {})

        concept_dir = os.path.join(EXPERIMENT_DIR, "data", "images_multi", concept)
        os.makedirs(concept_dir, exist_ok=True)
        existing_files = _existing_images_for_concept(concept_dir)
        before = len(existing_files)
        need = max(0, target - before)
        if need == 0:
            rels = [os.path.join("data", "images_multi", concept, fn) for fn in existing_files]
            concept_to_images[concept] = rels
            if not dry_run:
                prompt = _prompt_for_concept(concept)
                for filename in existing_files:
                    if filename in clip_scores:
                        continue
                    abs_path = os.path.join(concept_dir, filename)
                    try:
                        img = Image.open(abs_path).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
                        score = _clip_score(model, processor, img, prompt)
                        clip_scores[filename] = round(float(score), 4)
                    except Exception as exc:
                        print(f"  score backfill failed {concept}/{filename}: {exc}")
            md["num_images"] = before
            results.append(TopupResult(concept=concept, before=before, added=0, after=before))
            continue

        print(f"[{concept}] need {need} (have {before}, target {target})")
        if dry_run:
            results.append(TopupResult(concept=concept, before=before, added=0, after=before))
            continue

        seen_ids = set()
        next_idx = _next_index(existing_files, concept)
        accepted = 0
        prompt = _prompt_for_concept(concept)
        queries = _query_candidates(concept)

        for query in queries:
            if accepted >= need:
                break
            continue_token = None
            for page in range(1, max_pages + 1):
                if accepted >= need:
                    break
                try:
                    if search_source == "wikimedia":
                        rows, continue_token = _wikimedia_search(
                            query=query,
                            limit=per_page,
                            continue_token=continue_token,
                        )
                    else:
                        rows = _unsplash_search(
                            query=query,
                            page=page,
                            per_page=per_page,
                            access_key=unsplash_access_key,
                        )
                except Exception as exc:
                    print(f"  search failed query='{query}' page={page}: {exc}")
                    continue
                if not rows:
                    continue
                for img_id, img_url in rows:
                    if accepted >= need:
                        break
                    if img_id in seen_ids:
                        continue
                    seen_ids.add(img_id)
                    try:
                        img = _download_image(img_url)
                        score = _clip_score(model, processor, img, prompt)
                    except Exception as exc:
                        print(f"  skip candidate ({img_id}): {exc}")
                        continue
                    if score < clip_threshold:
                        continue
                    filename = f"{concept}_{next_idx:03d}.jpg"
                    out_path = os.path.join(concept_dir, filename)
                    img.save(out_path, "JPEG", quality=95)
                    clip_scores[filename] = round(float(score), 4)
                    if search_source == "wikimedia":
                        image_sources[filename] = "wikimedia_commons"
                    else:
                        image_sources[filename] = "unsplash_api" if unsplash_access_key else "unsplash_public"
                    next_idx += 1
                    accepted += 1
                    print(f"  accept {accepted}/{need}: {filename} score={score:.3f}")
                    if sleep_s > 0:
                        time.sleep(sleep_s)
            if accepted < need:
                print(f"  partial after query='{query}': {accepted}/{need}")

        existing_files = _existing_images_for_concept(concept_dir)
        after = len(existing_files)
        rels = [os.path.join("data", "images_multi", concept, fn) for fn in existing_files]
        concept_to_images[concept] = rels
        md["num_images"] = after
        if not md.get("source"):
            md["source"] = "unsplash"

        results.append(TopupResult(concept=concept, before=before, added=max(0, after - before), after=after))

    failures = [(row.concept, row.after) for row in results if row.after < target]
    wrote_manifest = False
    if not dry_run and (allow_partial or not failures):
        if not failures:
            manifest["images_per_concept_target"] = target
        else:
            manifest["images_per_concept_target"] = original_target
        _atomic_write_json(manifest_path, manifest)
        wrote_manifest = True

    return results, failures, wrote_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Top up images per concept with CLIP-gated Unsplash samples.")
    parser.add_argument("--manifest", type=str, default=MANIFEST_PATH)
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--clip-threshold", type=float, default=0.20)
    parser.add_argument("--per-page", type=int, default=30)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument(
        "--search-source",
        type=str,
        choices=["unsplash", "wikimedia"],
        default="unsplash",
        help="Image search source for top-up.",
    )
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="Optional subset of concepts to top up.",
    )
    parser.add_argument(
        "--unsplash-access-key",
        type=str,
        default=None,
        help="Unsplash API access key (or set UNSPLASH_ACCESS_KEY env var).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write manifest updates even when some concepts remain below target.",
    )
    args = parser.parse_args()

    access_key = args.unsplash_access_key or os.environ.get("UNSPLASH_ACCESS_KEY")
    if not access_key and not args.dry_run:
        print("WARNING: no Unsplash API key provided. Falling back to public endpoint (may be blocked).")

    results, failures, wrote_manifest = topup_manifest(
        manifest_path=os.path.abspath(args.manifest),
        target=args.target,
        clip_threshold=args.clip_threshold,
        per_page=args.per_page,
        max_pages=args.max_pages,
        unsplash_access_key=access_key,
        search_source=args.search_source,
        selected_concepts=args.concepts,
        dry_run=args.dry_run,
        sleep_s=args.sleep_s,
        allow_partial=args.allow_partial,
    )

    print("\nTop-up summary:")
    for row in results:
        print(f"  {row.concept:<18} before={row.before:>2} added={row.added:>2} after={row.after:>2}")
    if wrote_manifest:
        print(f"\nManifest updated: {os.path.abspath(args.manifest)}")
    else:
        print("\nManifest not updated (incomplete run and --allow-partial not set).")
    if failures:
        print("\nINCOMPLETE concepts:")
        for concept, count in failures:
            print(f"  - {concept}: {count}/{args.target}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
