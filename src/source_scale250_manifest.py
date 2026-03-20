import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import boto3
import botocore
import nltk
import requests
import torch
from nltk.corpus import wordnet as wn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
IMAGE_SIZE = (224, 224)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
OPENIMAGES_CLASS_DESCRIPTIONS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
OPENIMAGES_IMAGE_LABEL_URLS = {
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv",
    "train": "https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels.csv",
}
OPENIMAGES_IMAGE_BUCKET = "open-images-dataset"
OPENIMAGES_IMAGE_LABEL_CACHE_PATH = os.path.join(REPO_ROOT, "data", "openimages_image_label_hits.json")

DEFAULT_THRESHOLDS = {
    "imagenet": 0.17,
    "openimages": 0.18,
    "unsplash": 0.20,
}

IMAGENET_CONCEPT_TERM_OVERRIDES = {
    "apartment building": ["apartment_building"],
    "bookshelf": ["bookcase"],
    "car": ["automobile", "motorcar"],
    "dresser": ["chest_of_drawers"],
    "handbag": ["purse", "pocketbook"],
    "hot dog": ["hotdog"],
    "sink": ["washbasin"],
    "tram": ["tramcar", "streetcar"],
    "truck": ["motortruck"],
}

IMAGENET_CONCEPT_SYNSET_OVERRIDES = {
    "chair": ["chair.n.01"],
    "fox": ["fox.n.01"],
    "glove": ["glove.n.02"],
    "house": ["house.n.01"],
    "jacket": ["jacket.n.01"],
    "mountain": ["mountain.n.01"],
    "table": ["table.n.02", "table.n.03"],
    "turtle": ["turtle.n.02"],
    "tree": ["tree.n.01"],
    "truck": ["truck.n.01"],
    "whale": ["whale.n.02"],
    "wolf": ["wolf.n.01"],
}

IMAGENET_LABEL_DENYLIST = {
    "house": {"cinema", "home theater", "monastery", "palace", "boathouse", "birdhouse", "greenhouse"},
    "jacket": {"book jacket"},
    "mountain": {"volcano"},
    "wolf": {"coyote"},
}

IMAGENET_LABEL_ALLOWLIST = {
    "beach": {"seashore"},
    "bowl": {"mixing bowl", "soup bowl"},
    "glove": {"mitten"},
    "handbag": {"purse"},
    "house": {"mobile home", "yurt", "monastery", "palace"},
    "lake": {"lakeside"},
    "mug": {"coffee mug"},
    "pagoda": {"stupa"},
    "saw": {"chain saw"},
    "scooter": {"motor scooter"},
    "shorts": {"swimming trunks"},
    "skirt": {"hoopskirt", "miniskirt", "overskirt"},
    "sink": {"washbasin"},
    "temple": {"stupa"},
    "van": {"minivan", "moving van", "police van"},
}

OPENIMAGES_PROXY_MAP = {
    "agaric": ["Mushroom"],
    "apartment building": ["Building", "Office building"],
    "bolete": ["Mushroom"],
    "bamboo": ["Plant", "Tree"],
    "cabinet": ["Cabinetry", "Chest of drawers", "Drawer"],
    "canyon": [],
    "drill": ["Drill (Tool)", "Handheld power drill", "Pneumatic drill"],
    "dresser": ["Chest of drawers", "Cabinetry", "Drawer"],
    "fern": ["Plant", "Houseplant"],
    "forklift": ["Forklift truck"],
    "gyromitra": ["Mushroom"],
    "bridge": ["Bridge (structure)", "Arch bridge", "Suspension bridge", "Truss bridge", "Cable-stayed bridge", "Beam bridge"],
    "maracas": ["Maraca"],
    "mosque": ["Building"],
    "microwave": ["Microwave oven"],
    "mountain": [],
    "pan": ["Frying pan"],
    "pot": ["Saucepan", "Cauldron"],
    "purse": ["Coin purse"],
    "sofa": ["Sofa bed"],
    "spatula": ["Spatula", "Metal spatula"],
    "tram": ["Train", "Bus"],
    "valley": [],
    "washing machine": ["Washing machine"],
    "wolf": ["Dog"],
}

OPENIMAGES_IMAGE_LABEL_MAP = {
    "apartment building": ["Office building", "Building", "Skyscraper"],
    "bamboo": ["Bamboo", "Plant", "Tree"],
    "cabinet": ["Cabinetry", "Chest of drawers", "Drawer"],
    "bridge": ["Bridge (structure)", "Arch bridge", "Suspension bridge", "Truss bridge", "Cable-stayed bridge", "Beam bridge"],
    "canyon": ["Canyon"],
    "drill": ["Drill (Tool)", "Handheld power drill", "Pneumatic drill"],
    "fern": ["Fern", "Plant", "Houseplant"],
    "forest": ["Forest"],
    "forklift": ["Forklift truck"],
    "hammer": ["Hammer"],
    "lake": ["Lake"],
    "maracas": ["Maraca"],
    "microwave": ["Microwave oven"],
    "mountain": ["Mountain"],
    "ocean": ["Ocean"],
    "pan": ["Frying pan"],
    "pot": ["Saucepan", "Cauldron"],
    "purse": ["Coin purse"],
    "river": ["River"],
    "screwdriver": ["Screwdriver"],
    "sofa": ["Sofa bed"],
    "spatula": ["Spatula", "Metal spatula"],
    "valley": ["Valley"],
    "waterfall": ["Waterfall"],
    "washing machine": ["Washing machine"],
    "wrench": ["Wrench"],
}

UNSPLASH_QUERY_MAP = {
    "apartment building": ["apartment building exterior", "apartment block urban", "residential apartment building"],
    "bamboo": ["bamboo plant grove", "bamboo stalks forest", "bamboo leaves"],
    "banjo": ["banjo instrument", "banjo close up", "banjo music instrument"],
    "cabinet": ["cabinet furniture", "wood cabinet", "storage cabinet"],
    "canyon": ["canyon landscape", "canyon cliffs", "canyon vista"],
    "dresser": ["dresser furniture bedroom", "wood dresser drawers", "dresser cabinet"],
    "fern": ["fern plant leaves", "fern fronds forest", "fern foliage"],
    "geyser": ["geyser eruption", "geyser geothermal", "geyser national park"],
    "guitar": ["guitar instrument", "acoustic guitar", "electric guitar"],
    "gyromitra": ["false morel mushroom", "gyromitra mushroom", "brain mushroom forest"],
    "hammer": ["hammer tool", "claw hammer", "hammer on workbench"],
    "harp": ["harp instrument", "concert harp", "harp strings"],
    "hot dog": ["hot dog food", "hot dog bun", "hot dog on plate"],
    "jacket": ["jacket clothing", "winter jacket", "jacket on person"],
    "mosque": ["mosque exterior", "mosque building", "islamic mosque architecture"],
    "mountain": ["mountain landscape", "mountain peak", "mountain range"],
    "sandal": ["sandal footwear", "leather sandal", "sandal shoe"],
    "screwdriver": ["screwdriver tool", "screwdriver on table", "hand tool screwdriver"],
    "shirt": ["shirt clothing", "dress shirt", "shirt on hanger"],
    "spatula": ["spatula kitchen utensil", "metal spatula cooking", "spatula on countertop"],
    "sunglasses": ["sunglasses eyewear", "sunglasses product photo", "sunglasses on face"],
    "table": ["table furniture", "wood dining table", "table interior"],
    "tram": ["tram streetcar", "tram city transport", "tram on tracks"],
    "tree": ["tree nature", "single tree outdoors", "tree trunk canopy"],
    "truck": ["truck vehicle", "cargo truck", "pickup truck"],
    "valley": ["valley landscape", "mountain valley", "green valley"],
    "washing machine": ["washing machine appliance", "washing machine laundry", "front load washing machine"],
    "wolf": ["wolf wildlife", "wolf animal", "wolf in nature"],
}


@dataclass
class Candidate:
    concept: str
    source: str
    image: Image.Image
    record_id: str
    source_url: str
    acquisition_method: str
    acquisition_query: str
    source_class_label: str
    source_class_id: str
    proxy_used: str
    candidate_filename: str
    candidate_relpath: str
    clip_score: float | None = None
    aux_score: float | None = None


_OPENIMAGES_BUCKET = None
_WORDNET_READY = False
_CONCEPT_SYNSET_CACHE: Dict[str, List[Any]] = {}
_IMAGENET_SYNSET_CACHE: List[Any] | None = None


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_source_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_cached_openimages_label_hits(path: str = OPENIMAGES_IMAGE_LABEL_CACHE_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"schema_version": "1.0", "labels": {}}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "labels" not in payload:
        payload = {"schema_version": "1.0", "labels": {}}
    return payload


def save_cached_openimages_label_hits(payload: Dict[str, Any], path: str = OPENIMAGES_IMAGE_LABEL_CACHE_PATH) -> None:
    atomic_write_json(path, payload)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def split_label_terms(label_name: str) -> List[str]:
    return [part.strip() for part in label_name.split(",") if part.strip()]


def ensure_wordnet() -> None:
    global _WORDNET_READY
    if _WORDNET_READY:
        return
    try:
        wn.synsets("dog", pos=wn.NOUN)
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    _WORDNET_READY = True


def concept_synsets(concept: str) -> List[Any]:
    cached = _CONCEPT_SYNSET_CACHE.get(concept)
    if cached is not None:
        return cached

    ensure_wordnet()
    override_names = IMAGENET_CONCEPT_SYNSET_OVERRIDES.get(concept, [])
    if override_names:
        synsets = [wn.synset(name) for name in override_names]
        _CONCEPT_SYNSET_CACHE[concept] = synsets
        return synsets

    terms = [concept.replace(" ", "_"), concept]
    terms.extend(IMAGENET_CONCEPT_TERM_OVERRIDES.get(concept, []))

    synsets: List[Any] = []
    seen = set()
    for term in terms:
        term_synsets = wn.synsets(term, pos=wn.NOUN)
        if not term_synsets:
            continue
        synset = term_synsets[0]
        if synset.name() in seen:
            continue
        seen.add(synset.name())
        synsets.append(synset)

    _CONCEPT_SYNSET_CACHE[concept] = synsets
    return synsets


def load_imagenet_synsets() -> List[Any]:
    global _IMAGENET_SYNSET_CACHE
    if _IMAGENET_SYNSET_CACHE is not None:
        return _IMAGENET_SYNSET_CACHE

    ensure_wordnet()
    response = requests.get(IMAGENET_CLASS_INDEX_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()

    synsets: List[Any] = []
    for idx in range(len(payload)):
        wnid = payload[str(idx)][0]
        synsets.append(wn.synset_from_pos_and_offset("n", int(wnid[1:])))

    _IMAGENET_SYNSET_CACHE = synsets
    return synsets


def synset_is_descendant(label_synset: Any, candidate_hypernyms: Sequence[Any]) -> bool:
    hypernym_names = {node.name() for node in label_synset.closure(lambda current: current.hypernyms())}
    hypernym_names.add(label_synset.name())
    return any(synset.name() in hypernym_names for synset in candidate_hypernyms)


def imagenet_label_exact_match(concept: str, label_name: str) -> bool:
    target = normalize(concept)
    return target in {normalize(term) for term in split_label_terms(label_name)}


def imagenet_label_matches_concept(concept: str, label_name: str, label_synset: Any) -> bool:
    normalized_label_terms = {normalize(term) for term in split_label_terms(label_name)}
    allowed_terms = IMAGENET_LABEL_ALLOWLIST.get(concept, set())
    if normalized_label_terms & {normalize(term) for term in allowed_terms}:
        return True
    if normalized_label_terms & IMAGENET_LABEL_DENYLIST.get(concept, set()):
        return False
    if imagenet_label_exact_match(concept, label_name):
        return True

    concept_hypernyms = concept_synsets(concept)
    if not concept_hypernyms:
        return False
    return synset_is_descendant(label_synset, concept_hypernyms)


def article_for(concept: str) -> str:
    return "an" if concept[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def concept_prompt(concept: str) -> str:
    return f"{article_for(concept)} photo of {concept}"


def average_hash(image: Image.Image, size: int = 8) -> int:
    gray = image.convert("L").resize((size, size), Image.LANCZOS)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for idx, value in enumerate(pixels):
        if value > avg:
            bits |= 1 << idx
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def parse_float_map(text: str) -> Dict[str, float]:
    if not text:
        return dict(DEFAULT_THRESHOLDS)
    result = dict(DEFAULT_THRESHOLDS)
    for part in text.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = float(value.strip())
    return result


def existing_source_count(md: Dict[str, Any], source: str) -> int:
    image_sources = md.get("image_sources", {})
    return sum(1 for value in image_sources.values() if value == source)


def load_provenance_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def atomic_write_provenance(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_prov_", suffix=".csv", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def source_target(md: Dict[str, Any], source: str) -> int:
    return int(md.get("source_mix_target", {}).get(source, 0))


def source_needs_fill(md: Dict[str, Any], source: str) -> bool:
    return existing_source_count(md, source) < source_target(md, source)


def load_clip(device: str) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model.to(device)
    return model, processor


def clip_score(model: CLIPModel, processor: CLIPProcessor, device: str, image: Image.Image, text: str) -> float:
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return float(outputs.logits_per_image.item() / 100.0)


def download_image(url: str) -> Image.Image:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)


def load_image_bytes(raw_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)


def get_openimages_bucket():
    global _OPENIMAGES_BUCKET
    if _OPENIMAGES_BUCKET is None:
        _OPENIMAGES_BUCKET = boto3.resource(
            "s3",
            config=botocore.config.Config(signature_version=botocore.UNSIGNED),
        ).Bucket(OPENIMAGES_IMAGE_BUCKET)
    return _OPENIMAGES_BUCKET


def download_openimages_image(split: str, image_id: str) -> Image.Image:
    obj = get_openimages_bucket().Object(f"{split}/{image_id}.jpg")
    return load_image_bytes(obj.get()["Body"].read())


def save_candidate_image(candidate_root: str, source: str, storage_slug: str, filename: str, image: Image.Image) -> str:
    source_dir = os.path.join(candidate_root, source, storage_slug)
    os.makedirs(source_dir, exist_ok=True)
    abs_path = os.path.join(source_dir, filename)
    image.save(abs_path, "JPEG", quality=95)
    return os.path.relpath(abs_path, REPO_ROOT)


def select_imagenet_label_ids(concept: str, label_names: Sequence[str], label_synsets: Sequence[Any]) -> List[int]:
    selected: List[int] = []
    for idx, (label_name, label_synset) in enumerate(zip(label_names, label_synsets)):
        if imagenet_label_matches_concept(concept, label_name, label_synset):
            selected.append(idx)
    return selected


def load_imagenet_metadata() -> tuple[List[str], List[Any]]:
    label_names = requests.get(IMAGENET_LABELS_URL, timeout=30).text.splitlines()
    label_synsets = load_imagenet_synsets()
    if len(label_names) != len(label_synsets):
        raise RuntimeError(
            f"ImageNet metadata mismatch: {len(label_names)} labels vs {len(label_synsets)} synsets"
        )
    return label_names, label_synsets


def load_imagenet_dataset():
    from datasets import load_dataset

    dataset = load_dataset("mrm8488/ImageNet1K-val", split="train", download_mode="reuse_cache_if_exists")
    return dataset


def build_imagenet_label_index(dataset) -> Dict[int, List[int]]:
    label_index: Dict[int, List[int]] = {}
    for idx, label in enumerate(dataset["label"]):
        label_index.setdefault(int(label), []).append(idx)
    return label_index


def imagenet_candidates(
    concept: str,
    storage_slug: str,
    candidate_root: str,
    label_names: Sequence[str],
    label_synsets: Sequence[Any],
    dataset,
    label_index: Dict[int, List[int]],
    max_candidates: int,
) -> List[Candidate]:
    label_ids = select_imagenet_label_ids(concept, label_names, label_synsets)
    if not label_ids:
        return []

    candidates: List[Candidate] = []
    for label in label_ids:
        for idx in label_index.get(label, []):
            row = dataset[idx]
            image = row["image"].convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
            record_id = f"imagenet_val_{idx}"
            candidate_name = f"{storage_slug}_imagenet_candidate_{len(candidates)+1:03d}.jpg"
            candidate_relpath = save_candidate_image(candidate_root, "imagenet", storage_slug, candidate_name, image)
            candidates.append(
                Candidate(
                    concept=concept,
                    source="imagenet",
                    image=image,
                    record_id=record_id,
                    source_url="mrm8488/ImageNet1K-val",
                    acquisition_method="imagenet_label_match",
                    acquisition_query="; ".join(label_names[label_id] for label_id in label_ids[:8]),
                    source_class_label=label_names[label],
                    source_class_id=str(label),
                    proxy_used="true" if not imagenet_label_exact_match(concept, label_names[label]) else "false",
                    candidate_filename=candidate_name,
                    candidate_relpath=candidate_relpath,
                )
            )
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def load_openimages_classes() -> List[str]:
    from fiftyone.utils import openimages as foi

    return list(foi.get_classes())


def load_openimages_label_catalog() -> Dict[str, tuple[str, str]]:
    response = requests.get(OPENIMAGES_CLASS_DESCRIPTIONS_URL, timeout=30)
    response.raise_for_status()
    catalog: Dict[str, tuple[str, str]] = {}
    for row in csv.reader(io.StringIO(response.text)):
        if len(row) < 2:
            continue
        class_id, name = row[0].strip(), row[1].strip()
        catalog[normalize(name)] = (class_id, name)
    return catalog


def select_openimages_classes(concept: str, classes: Sequence[str]) -> List[str]:
    exact = [name for name in classes if normalize(name) == normalize(concept)]
    if exact:
        return exact
    return [name for name in OPENIMAGES_PROXY_MAP.get(concept, []) if name in classes]


def select_openimages_image_labels(
    concept: str,
    label_catalog: Dict[str, tuple[str, str]],
) -> List[tuple[str, str]]:
    preferred_terms = [concept]
    preferred_terms.extend(OPENIMAGES_IMAGE_LABEL_MAP.get(concept, []))
    preferred_terms.extend(OPENIMAGES_PROXY_MAP.get(concept, []))

    selected: List[tuple[str, str]] = []
    seen_norm = set()
    for term in preferred_terms:
        normalized = normalize(term)
        if not normalized or normalized in seen_norm:
            continue
        seen_norm.add(normalized)
        match = label_catalog.get(normalized)
        if match is not None:
            selected.append(match)
    return selected


def openimages_detection_candidates(
    concept: str,
    storage_slug: str,
    candidate_root: str,
    classes: Sequence[str],
    max_samples: int,
) -> List[Candidate]:
    import fiftyone as fo
    import fiftyone.zoo as foz

    chosen_classes = select_openimages_classes(concept, classes)
    if not chosen_classes:
        return []

    dataset_name = f"oi7_{storage_slug}_{hashlib.sha1(f'{concept}_{time.time()}'.encode()).hexdigest()[:8]}"
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=chosen_classes,
        max_samples=max_samples,
        dataset_name=dataset_name,
    )

    candidates: List[Candidate] = []
    try:
        for sample in dataset:
            detections = getattr(sample, "ground_truth", None)
            if detections is None:
                continue
            matched = [det for det in detections.detections if normalize(det.label) in {normalize(name) for name in chosen_classes}]
            if not matched:
                continue
            max_area = max(float(det.bounding_box[2]) * float(det.bounding_box[3]) for det in matched)
            image = Image.open(sample.filepath).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
            record_id = os.path.splitext(os.path.basename(sample.filepath))[0]
            candidate_name = f"{storage_slug}_openimages_candidate_{len(candidates)+1:03d}.jpg"
            candidate_relpath = save_candidate_image(candidate_root, "openimages", storage_slug, candidate_name, image)
            primary_label = matched[0].label
            candidates.append(
                Candidate(
                    concept=concept,
                    source="openimages",
                    image=image,
                    record_id=record_id,
                    source_url=sample.filepath,
                    acquisition_method="openimages_detection_exact" if normalize(primary_label) == normalize(concept) else "openimages_detection_proxy",
                    acquisition_query="; ".join(chosen_classes),
                    source_class_label=primary_label,
                    source_class_id=primary_label,
                    proxy_used="true" if normalize(primary_label) != normalize(concept) else "false",
                    candidate_filename=candidate_name,
                    candidate_relpath=candidate_relpath,
                    aux_score=max_area,
                )
            )
    finally:
        fo.delete_dataset(dataset.name)

    return candidates


def iter_openimages_image_labels(url: str) -> Iterable[Dict[str, str]]:
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        lines = (line.decode("utf-8") for line in response.iter_lines())
        yield from csv.DictReader(lines)


def ensure_openimages_label_hit_cache(
    label_ids: Sequence[str],
    per_label_limit: int = 256,
    cache_path: str = OPENIMAGES_IMAGE_LABEL_CACHE_PATH,
) -> Dict[str, Any]:
    cache = load_cached_openimages_label_hits(cache_path)
    labels_cache: Dict[str, Dict[str, List[str]]] = cache.setdefault("labels", {})

    missing_ids = [label_id for label_id in label_ids if label_id not in labels_cache]
    if not missing_ids:
        return cache

    pending = set(missing_ids)
    for split, url in OPENIMAGES_IMAGE_LABEL_URLS.items():
        split_done = set()
        for row in iter_openimages_image_labels(url):
            if row.get("Confidence") != "1":
                continue
            class_id = row.get("LabelName", "")
            if class_id not in pending:
                continue
            image_id = row.get("ImageID", "")
            if not image_id:
                continue
            per_label = labels_cache.setdefault(class_id, {})
            split_hits = per_label.setdefault(split, [])
            if len(split_hits) < per_label_limit:
                split_hits.append(image_id)
            if sum(len(ids) for ids in per_label.values()) >= per_label_limit:
                split_done.add(class_id)
            if split_done == pending:
                break
        pending -= split_done
        if not pending:
            break

    for label_id in missing_ids:
        labels_cache.setdefault(label_id, {})

    save_cached_openimages_label_hits(cache, cache_path)
    return cache


def openimages_image_label_candidates(
    concept: str,
    storage_slug: str,
    candidate_root: str,
    label_catalog: Dict[str, tuple[str, str]],
    max_samples: int,
    seen_record_ids: set[str],
) -> List[Candidate]:
    selected_labels = select_openimages_image_labels(concept, label_catalog)
    if not selected_labels:
        return []

    label_id_to_name = {class_id: name for class_id, name in selected_labels}
    wanted_ids = set(label_id_to_name)
    cache = ensure_openimages_label_hit_cache(sorted(wanted_ids))
    labels_cache: Dict[str, Dict[str, List[str]]] = cache.get("labels", {})
    candidates: List[Candidate] = []
    for class_id, class_name in selected_labels:
        split_map = labels_cache.get(class_id, {})
        for split in ("validation", "test", "train"):
            for image_id in split_map.get(split, []):
                record_id = f"{split}/{image_id}"
                if record_id in seen_record_ids:
                    continue
                seen_record_ids.add(record_id)
                try:
                    image = download_openimages_image(split, image_id)
                except Exception:
                    continue

                candidates.append(
                    Candidate(
                        concept=concept,
                        source="openimages",
                        image=image,
                        record_id=record_id,
                        source_url=f"s3://{OPENIMAGES_IMAGE_BUCKET}/{split}/{image_id}.jpg",
                        acquisition_method="openimages_image_label_exact" if normalize(class_name) == normalize(concept) else "openimages_image_label_proxy",
                        acquisition_query="; ".join(name for _, name in selected_labels),
                        source_class_label=class_name,
                        source_class_id=class_id,
                        proxy_used="true" if normalize(class_name) != normalize(concept) else "false",
                        candidate_filename="",
                        candidate_relpath="",
                    )
                )
                if len(candidates) >= max_samples:
                    return candidates

    return candidates


def openimages_candidates(
    concept: str,
    storage_slug: str,
    candidate_root: str,
    classes: Sequence[str],
    label_catalog: Dict[str, tuple[str, str]],
    max_samples: int,
) -> List[Candidate]:
    candidates = openimages_detection_candidates(
        concept,
        storage_slug,
        candidate_root,
        classes,
        max_samples,
    )
    seen_record_ids = {candidate.record_id for candidate in candidates}
    if len(candidates) >= max_samples:
        return candidates

    fallback_candidates = openimages_image_label_candidates(
        concept,
        storage_slug,
        candidate_root,
        label_catalog,
        max_samples - len(candidates),
        seen_record_ids,
    )

    next_index = len(candidates) + 1
    for candidate in fallback_candidates:
        candidate_name = f"{storage_slug}_openimages_candidate_{next_index:03d}.jpg"
        candidate.candidate_filename = candidate_name
        candidate.candidate_relpath = save_candidate_image(candidate_root, "openimages", storage_slug, candidate_name, candidate.image)
        candidates.append(candidate)
        next_index += 1
    return candidates


def unsplash_queries(concept: str) -> List[str]:
    queries = list(UNSPLASH_QUERY_MAP.get(concept, []))
    if not queries:
        queries = [
            concept,
            f"{concept} photo",
            f"{concept} realistic photograph",
        ]
    return queries


def unsplash_candidates(
    concept: str,
    storage_slug: str,
    candidate_root: str,
    per_page: int,
    max_pages: int,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen_ids = set()

    for query in unsplash_queries(concept):
        for page in range(1, max_pages + 1):
            response = requests.get(
                "https://unsplash.com/napi/search/photos",
                params={"query": query, "per_page": per_page, "page": page},
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            rows = response.json().get("results", [])
            for row in rows:
                record_id = row.get("id")
                image_url = row.get("urls", {}).get("regular")
                if not record_id or not image_url or record_id in seen_ids:
                    continue
                seen_ids.add(record_id)
                try:
                    image = download_image(image_url)
                except Exception:
                    continue
                candidate_name = f"{storage_slug}_unsplash_candidate_{len(candidates)+1:03d}.jpg"
                candidate_relpath = save_candidate_image(candidate_root, "unsplash", storage_slug, candidate_name, image)
                candidates.append(
                    Candidate(
                        concept=concept,
                        source="unsplash",
                        image=image,
                        record_id=record_id,
                        source_url=image_url,
                        acquisition_method="unsplash_public_search",
                        acquisition_query=query,
                        source_class_label="",
                        source_class_id="",
                        proxy_used="false",
                        candidate_filename=candidate_name,
                        candidate_relpath=candidate_relpath,
                    )
                )
    return candidates


def build_provenance_row(
    md: Dict[str, Any],
    candidate: Candidate,
    source_slot: int,
    accepted_filename: str,
    accepted_relpath: str,
    review_status: str,
    rejection_reason: str,
) -> Dict[str, Any]:
    return {
        "concept": candidate.concept,
        "stratum": md.get("stratum", ""),
        "semantic_type": md.get("semantic_type", ""),
        "selection_status": md.get("selection_status", ""),
        "source_feasibility": md.get("source_feasibility", ""),
        "storage_slug": md.get("storage_slug", ""),
        "source": candidate.source,
        "source_slot": str(source_slot),
        "candidate_filename": candidate.candidate_filename,
        "accepted_filename": accepted_filename,
        "candidate_relpath": candidate.candidate_relpath,
        "accepted_relpath": accepted_relpath,
        "candidate_record_id": candidate.record_id,
        "candidate_source_url": candidate.source_url,
        "acquisition_method": candidate.acquisition_method,
        "acquisition_query": candidate.acquisition_query,
        "source_class_label": candidate.source_class_label,
        "source_class_id": candidate.source_class_id,
        "proxy_used": candidate.proxy_used,
        "clip_score": "" if candidate.clip_score is None else f"{candidate.clip_score:.4f}",
        "review_status": review_status,
        "rejection_reason": rejection_reason,
        "diversity_notes": "",
        "license_or_terms": "",
        "curator_initials": "",
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "",
    }


def ensure_concept_dir(image_root: str, storage_slug: str) -> str:
    concept_dir = os.path.join(image_root, storage_slug)
    os.makedirs(concept_dir, exist_ok=True)
    return concept_dir


def accepted_hashes_for_concept(concept_paths: Sequence[str]) -> List[int]:
    hashes: List[int] = []
    for relpath in concept_paths:
        abs_path = os.path.join(REPO_ROOT, relpath)
        if not os.path.exists(abs_path):
            continue
        image = Image.open(abs_path).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
        hashes.append(average_hash(image))
    return hashes


def accept_source_candidates(
    concept: str,
    md: Dict[str, Any],
    concept_to_images: Dict[str, List[str]],
    image_root: str,
    thresholds: Dict[str, float],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    candidates: List[Candidate],
) -> tuple[int, List[Dict[str, Any]]]:
    target = int(md.get("source_mix_target", {}).get(candidates[0].source if candidates else "", 0))
    if target <= 0:
        return 0, []

    already = existing_source_count(md, candidates[0].source if candidates else "")
    need = max(0, target - already)
    if need <= 0:
        return 0, []

    accepted_paths = concept_to_images.get(concept, [])
    existing_hashes = accepted_hashes_for_concept(accepted_paths)
    provenance_rows: List[Dict[str, Any]] = []
    accepted = 0
    slot = already + 1
    prompt = concept_prompt(concept)
    source = candidates[0].source if candidates else ""
    threshold = thresholds[source]
    concept_dir = ensure_concept_dir(image_root, md["storage_slug"])

    for candidate in candidates:
        if accepted >= need:
            break
        candidate.clip_score = clip_score(model, processor, device, candidate.image, prompt)
        rejection_reason = ""
        review_status = "accepted_auto"

        if candidate.clip_score < threshold:
            review_status = "rejected_auto"
            rejection_reason = "clip_below_threshold"
        elif candidate.aux_score is not None and source == "openimages" and candidate.aux_score < 0.06:
            review_status = "rejected_auto"
            rejection_reason = "bbox_area_too_small"
        else:
            cand_hash = average_hash(candidate.image)
            if any(hamming_distance(cand_hash, prev) <= 6 for prev in existing_hashes):
                review_status = "rejected_auto"
                rejection_reason = "near_duplicate"
            else:
                accepted_name = f"{md['storage_slug']}_{source}_{slot:02d}.jpg"
                accepted_abs = os.path.join(concept_dir, accepted_name)
                candidate.image.save(accepted_abs, "JPEG", quality=95)
                accepted_rel = os.path.relpath(accepted_abs, REPO_ROOT)
                concept_to_images.setdefault(concept, []).append(accepted_rel)
                md.setdefault("clip_scores", {})[accepted_name] = round(candidate.clip_score, 4)
                md.setdefault("image_sources", {})[accepted_name] = source
                existing_hashes.append(cand_hash)
                provenance_rows.append(
                    build_provenance_row(md, candidate, slot, accepted_name, accepted_rel, review_status, "")
                )
                accepted += 1
                slot += 1
                continue

        if review_status != "accepted_auto":
            provenance_rows.append(
                build_provenance_row(md, candidate, 0, "", "", review_status, rejection_reason)
            )

    md["num_images"] = len(concept_to_images.get(concept, []))
    counts = {
        src: existing_source_count(md, src)
        for src in md.get("source_mix_target", {})
    }
    md["source_mix_actual"] = counts
    concept_to_images[concept] = sorted(concept_to_images.get(concept, []), key=os.path.basename)
    return accepted, provenance_rows


def resolve_roots(manifest: Dict[str, Any]) -> tuple[str, str, str]:
    shadow = manifest.get("shadow_build", {})
    image_root_rel = shadow.get("active_image_root", "data/images_250_fresh")
    candidate_root_rel = shadow.get("candidate_root", "data/scale250_fresh_candidates")
    provenance_rel = shadow.get("provenance_ledger", "data/scale250_fresh_provenance.csv")
    return (
        os.path.join(REPO_ROOT, image_root_rel),
        os.path.join(REPO_ROOT, candidate_root_rel),
        os.path.join(REPO_ROOT, provenance_rel),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Source images for a scale250 fresh manifest.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--concepts", nargs="*", default=None)
    parser.add_argument("--sources", nargs="*", default=["imagenet", "openimages", "unsplash"])
    parser.add_argument("--max-imagenet-candidates", type=int, default=60)
    parser.add_argument("--max-openimages-samples", type=int, default=40)
    parser.add_argument("--unsplash-per-page", type=int, default=20)
    parser.add_argument("--unsplash-pages", type=int, default=2)
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest_path)
    manifest = load_json(manifest_path)
    concept_to_images = manifest.setdefault("concept_to_images", {})
    concept_metadata = manifest.setdefault("concept_metadata", {})
    image_root, candidate_root, provenance_path = resolve_roots(manifest)
    os.makedirs(image_root, exist_ok=True)
    os.makedirs(candidate_root, exist_ok=True)

    thresholds = parse_float_map(args.thresholds)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, processor = load_clip(device)

    selected_concepts = list(concept_to_images.keys())
    if args.concepts:
        allowed = set(args.concepts)
        selected_concepts = [concept for concept in selected_concepts if concept in allowed]

    imagenet_labels = []
    imagenet_synsets: List[Any] = []
    imagenet_dataset = None
    imagenet_label_index: Dict[int, List[int]] = {}
    if "imagenet" in args.sources:
        imagenet_labels, imagenet_synsets = load_imagenet_metadata()
        imagenet_dataset = load_imagenet_dataset()
        imagenet_label_index = build_imagenet_label_index(imagenet_dataset)

    openimages_classes = []
    openimages_label_catalog: Dict[str, tuple[str, str]] = {}
    if "openimages" in args.sources:
        openimages_classes = load_openimages_classes()
        openimages_label_catalog = load_openimages_label_catalog()

    provenance_rows = load_provenance_rows(provenance_path)
    stats = {"accepted": 0, "rejected": 0}

    for concept in selected_concepts:
        md = concept_metadata[concept]
        storage_slug = md["storage_slug"]
        concept_rows: List[Dict[str, Any]] = []

        if "imagenet" in args.sources:
            if not source_needs_fill(md, "imagenet"):
                print(f"[imagenet] {concept}: already complete")
            else:
                candidates = imagenet_candidates(
                    concept,
                    storage_slug,
                    candidate_root,
                    imagenet_labels,
                    imagenet_synsets,
                    imagenet_dataset,
                    imagenet_label_index,
                    args.max_imagenet_candidates,
                )
                accepted, rows = accept_source_candidates(
                    concept,
                    md,
                    concept_to_images,
                    image_root,
                    thresholds,
                    model,
                    processor,
                    device,
                    candidates,
                )
                stats["accepted"] += sum(1 for row in rows if row["review_status"] == "accepted_auto")
                stats["rejected"] += sum(1 for row in rows if row["review_status"] != "accepted_auto")
                concept_rows.extend(rows)
                print(f"[imagenet] {concept}: candidates={len(candidates)} accepted={accepted}")

        if "openimages" in args.sources:
            if not source_needs_fill(md, "openimages"):
                print(f"[openimages] {concept}: already complete")
            else:
                candidates = openimages_candidates(
                    concept,
                    storage_slug,
                    candidate_root,
                    openimages_classes,
                    openimages_label_catalog,
                    args.max_openimages_samples,
                )
                accepted, rows = accept_source_candidates(
                    concept,
                    md,
                    concept_to_images,
                    image_root,
                    thresholds,
                    model,
                    processor,
                    device,
                    candidates,
                )
                stats["accepted"] += sum(1 for row in rows if row["review_status"] == "accepted_auto")
                stats["rejected"] += sum(1 for row in rows if row["review_status"] != "accepted_auto")
                concept_rows.extend(rows)
                print(f"[openimages] {concept}: candidates={len(candidates)} accepted={accepted}")

        if "unsplash" in args.sources:
            if not source_needs_fill(md, "unsplash"):
                print(f"[unsplash] {concept}: already complete")
            else:
                candidates = unsplash_candidates(
                    concept,
                    storage_slug,
                    candidate_root,
                    args.unsplash_per_page,
                    args.unsplash_pages,
                )
                accepted, rows = accept_source_candidates(
                    concept,
                    md,
                    concept_to_images,
                    image_root,
                    thresholds,
                    model,
                    processor,
                    device,
                    candidates,
                )
                stats["accepted"] += sum(1 for row in rows if row["review_status"] == "accepted_auto")
                stats["rejected"] += sum(1 for row in rows if row["review_status"] != "accepted_auto")
                concept_rows.extend(rows)
                print(f"[unsplash] {concept}: candidates={len(candidates)} accepted={accepted}")

        if args.write and concept_rows:
            provenance_rows.extend(concept_rows)
            atomic_write_json(manifest_path, manifest)
            atomic_write_provenance(provenance_path, provenance_rows)

    if args.write:
        atomic_write_json(manifest_path, manifest)
        if provenance_rows:
            atomic_write_provenance(provenance_path, provenance_rows)
        print(f"\nManifest updated: {manifest_path}")
        print(f"Provenance updated: {provenance_path}")

    print(f"Accepted rows this run: {stats['accepted']}")
    print(f"Rejected rows this run: {stats['rejected']}")


if __name__ == "__main__":
    main()
