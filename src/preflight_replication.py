import argparse
import json
import os
import re
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST_PATH = os.path.join(EXPERIMENT_DIR, "data", "data_manifest_multi.json")
DEFAULT_RAW_DIR = os.path.join(EXPERIMENT_DIR, "results", "replication", "raw_data")
DEFAULT_MODELS_FILE = os.path.join(EXPERIMENT_DIR, "docs", "run_all_models.sh")
DEFAULT_CACHE_DIR = os.path.join(EXPERIMENT_DIR, "results", "replication", "cache")
CACHE_SCHEMA_VERSION = "1.1.0"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    raise SystemExit(1)


def parse_bool_arg(value: str) -> bool:
    val = str(value).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    fail(f"Invalid boolean value '{value}'. Use true/false.")
    return False


def parse_models_from_shell_script(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"MODELS=\((.*?)\)", text, re.DOTALL)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def load_manifest(path: str) -> Any:
    if not os.path.exists(path):
        fail(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    concept_to_images = manifest.get("concept_to_images", {})
    concept_metadata = manifest.get("concept_metadata", {})
    if not concept_to_images:
        fail("Manifest missing concept_to_images.")
    return manifest, concept_to_images, concept_metadata


def check_manifest_images(experiment_dir: str, concept_to_images: Dict[str, List[str]], min_images: int) -> None:
    missing_files = []
    low_count = []
    for concept, rel_paths in concept_to_images.items():
        if len(rel_paths) < min_images:
            low_count.append((concept, len(rel_paths)))
        for rel in rel_paths:
            abs_path = os.path.join(experiment_dir, rel)
            if not os.path.exists(abs_path):
                missing_files.append((concept, rel))
    if low_count:
        fail(
            "Concepts below minimum image count: "
            + ", ".join(f"{c}({n})" for c, n in low_count)
        )
    if missing_files:
        sample = ", ".join(f"{c}:{p}" for c, p in missing_files[:5])
        fail(f"Manifest references missing image files (showing up to 5): {sample}")


def check_clip_scores(
    concept_to_images: Dict[str, List[str]],
    concept_metadata: Dict[str, Any],
    require_clip_scores: bool = False,
) -> None:
    missing = []
    partial = []
    for concept, images in concept_to_images.items():
        md = concept_metadata.get(concept, {})
        scores = md.get("clip_scores", {})
        if not scores:
            missing.append(concept)
            continue
        if len(scores) < len(images):
            partial.append((concept, len(scores), len(images)))

    if missing or partial:
        lines = []
        if missing:
            lines.append(f"missing concepts: {len(missing)}")
        if partial:
            lines.append(f"partial concepts: {len(partial)}")
        msg = "CLIP score coverage incomplete (" + ", ".join(lines) + ")."
        if require_clip_scores:
            fail(msg)
        print(f"WARNING: {msg}")


def check_source_metadata(
    concept_to_images: Dict[str, List[str]],
    concept_metadata: Dict[str, Any],
    require_image_source_metadata: bool = True,
) -> None:
    missing_source = []
    invalid_image_source_map = []
    for concept, images in concept_to_images.items():
        md = concept_metadata.get(concept, {})
        source = md.get("source")
        if not source:
            missing_source.append(concept)
        image_sources = md.get("image_sources")
        if image_sources is not None:
            if not isinstance(image_sources, dict):
                invalid_image_source_map.append(f"{concept}: image_sources is not a dict")
                continue
            image_names = {os.path.basename(p) for p in images}
            unknown_keys = sorted(set(image_sources.keys()) - image_names)
            if unknown_keys:
                invalid_image_source_map.append(
                    f"{concept}: unknown image_sources keys {unknown_keys[:5]}"
                )

    if missing_source:
        msg = (
            "Concept metadata missing required source field for concepts: "
            + ", ".join(sorted(missing_source))
        )
        if require_image_source_metadata:
            fail(msg)
        print(f"WARNING: {msg}")
    if invalid_image_source_map:
        fail("Invalid image_sources mappings:\n" + "\n".join(invalid_image_source_map))


def read_raw_jsons(raw_dir: str) -> Dict[str, Any]:
    if not os.path.exists(raw_dir):
        return {}
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".json"))
    result: Dict[str, Any] = {}
    for fn in files:
        path = os.path.join(raw_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                result[fn] = json.load(f)
        except Exception as e:  # pragma: no cover - defensive for local file corruption
            fail(f"Could not read {path}: {e}")
    return result


def check_raw_outputs(raw_data: Dict[str, Any], expected_models: List[str], expected_concepts: List[str]) -> None:
    if not raw_data:
        fail("No raw model JSON files found.")

    found_models = set()
    concept_ref = expected_concepts

    for filename, data in raw_data.items():
        concepts = data.get("concepts", [])
        models = data.get("models", {})
        if not concepts:
            fail(f"{filename}: missing concepts list.")
        if set(concepts) != set(concept_ref):
            extra = sorted(set(concepts) - set(concept_ref))
            missing = sorted(set(concept_ref) - set(concepts))
            fail(
                f"{filename}: concept set mismatch vs manifest. "
                f"extra={extra}, missing={missing}"
            )
        if len(concepts) != len(concept_ref):
            fail(f"{filename}: concept count mismatch ({len(concepts)} vs {len(concept_ref)}).")
        if len(models) != 1:
            fail(f"{filename}: expected exactly one model entry, got {len(models)}.")
        found_models.update(models.keys())

    if expected_models:
        missing_models = sorted(set(expected_models) - found_models)
        if missing_models:
            fail(f"Missing raw outputs for configured models: {missing_models}")


def load_cache_manifest(cache_manifest_path: str, model_name: str) -> Dict[str, Any]:
    if not os.path.exists(cache_manifest_path):
        fail(
            f"Missing cache manifest for {model_name}: {cache_manifest_path}. "
            "Rerun extraction with --cache-image-embeddings true."
        )
    with open(cache_manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    required_fields = [
        "schema_version",
        "model_name",
        "model_config_fingerprint",
        "manifest_fingerprint",
        "layer_keys",
        "layer_profile_id",
        "requested_layers_spec",
        "concept_to_images",
        "dtype",
        "embedding_dim",
        "created_at",
    ]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        fail(f"Cache manifest missing fields for {model_name}: {missing}")
    if payload["schema_version"] != CACHE_SCHEMA_VERSION:
        fail(
            f"Cache schema mismatch for {model_name}: "
            f"{payload['schema_version']} != {CACHE_SCHEMA_VERSION}"
        )
    if payload["model_name"] != model_name:
        fail(
            f"Cache manifest model mismatch for {model_name}: "
            f"got {payload['model_name']}"
        )
    if payload.get("dtype") != "float32":
        fail(f"Cache dtype must be float32 for {model_name}, got {payload.get('dtype')}")
    if not isinstance(payload.get("embedding_dim"), int) or payload["embedding_dim"] <= 0:
        fail(f"Invalid cache embedding_dim for {model_name}: {payload.get('embedding_dim')}")
    if not isinstance(payload.get("layer_keys"), list) or not payload["layer_keys"]:
        fail(f"Invalid cache layer_keys for {model_name}: {payload.get('layer_keys')}")
    if not isinstance(payload.get("layer_profile_id"), str) or not payload["layer_profile_id"]:
        fail(
            f"Invalid cache layer_profile_id for {model_name}: "
            f"{payload.get('layer_profile_id')}"
        )
    if (
        not isinstance(payload.get("requested_layers_spec"), str)
        or not payload["requested_layers_spec"].strip()
    ):
        fail(
            f"Invalid cache requested_layers_spec for {model_name}: "
            f"{payload.get('requested_layers_spec')}"
        )
    return payload


def check_cache_outputs(
    raw_data: Dict[str, Any],
    cache_dir: str,
    expected_concepts: List[str],
) -> None:
    missing = []
    for _, data in raw_data.items():
        models = data.get("models", {})
        for model_name, model_info in models.items():
            model_type = model_info.get("config", {}).get("type")
            if model_type == "causal":
                # Language models still produce cache manifests, but no per-image shards are required.
                manifest_path = os.path.join(cache_dir, model_name, "cache_manifest.json")
                if not os.path.exists(manifest_path):
                    missing.append(f"{model_name}: missing cache manifest {manifest_path}")
                continue
            model_cache_dir = os.path.join(cache_dir, model_name)
            payload = load_cache_manifest(
                os.path.join(model_cache_dir, "cache_manifest.json"),
                model_name,
            )
            layer_keys = payload["layer_keys"]
            concept_to_images = payload.get("concept_to_images", {})
            for concept in expected_concepts:
                image_rows = concept_to_images.get(concept)
                if image_rows is None:
                    fail(
                        f"Cache manifest for {model_name} missing concept '{concept}' in concept_to_images."
                    )
                for layer_name in layer_keys:
                    shard = os.path.join(model_cache_dir, layer_name, f"{concept}.npy")
                    if not os.path.exists(shard):
                        missing.append(f"{model_name}: missing shard {shard}")
    if missing:
        preview = "\n".join(missing[:15])
        fail(
            "Cache integrity check failed. Missing cache files:\n"
            f"{preview}\n"
            "Remediation: rerun extraction for missing models with "
            "--force-cache-rebuild."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for replication experiment.")
    parser.add_argument("--phase", choices=["pre", "post"], required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--models-file", default=DEFAULT_MODELS_FILE)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--min-images-per-concept", type=int, default=10)
    parser.add_argument("--require-clip-scores", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument(
        "--require-image-source-metadata",
        type=str,
        default="true",
        help="Require concept_metadata.source for all concepts (true|false). Default: true.",
    )
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    raw_dir = os.path.abspath(args.raw_dir)
    models_file = os.path.abspath(args.models_file)
    cache_dir = os.path.abspath(args.cache_dir)
    require_source_metadata = parse_bool_arg(args.require_image_source_metadata)

    manifest, concept_to_images, concept_metadata = load_manifest(manifest_path)
    expected_concepts = list(concept_to_images.keys())

    print("Running manifest checks...")
    check_manifest_images(EXPERIMENT_DIR, concept_to_images, args.min_images_per_concept)
    check_clip_scores(
        concept_to_images,
        concept_metadata,
        require_clip_scores=args.require_clip_scores,
    )
    check_source_metadata(
        concept_to_images,
        concept_metadata,
        require_image_source_metadata=require_source_metadata,
    )
    print(
        f"Manifest checks passed: concepts={len(expected_concepts)}, "
        f"target_images={manifest.get('images_per_concept_target', 'unknown')}"
    )

    expected_models = parse_models_from_shell_script(models_file)
    if args.phase == "pre":
        if expected_models:
            print(f"Preflight model target count: {len(expected_models)}")
        print("Preflight checks passed.")
        return

    print("Running post-run raw output checks...")
    raw_data = read_raw_jsons(raw_dir)
    check_raw_outputs(raw_data, expected_models, expected_concepts)
    if args.require_cache:
        print("Running cache integrity checks...")
        check_cache_outputs(raw_data, cache_dir, expected_concepts)
    print(
        "Post-run checks passed: "
        f"raw_files={len(raw_data)}, models={len(expected_models) or 'n/a'}, "
        f"require_cache={args.require_cache}"
    )


if __name__ == "__main__":
    main()
