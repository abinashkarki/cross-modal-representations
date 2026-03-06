import os
import json
import argparse
import platform
import subprocess
import sys
import traceback
import logging
import hashlib
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image
import torch
import numpy as np
import transformers
from transformers import (
    AutoProcessor, 
    AutoModel, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoImageProcessor
)
from sklearn.metrics.pairwise import cosine_similarity

mx = None
mlx_load = None
HAS_MLX = False
MLX_IMPORT_ERROR = None

# Optimization env vars
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Setup Device (M1 Mac GPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
MANIFEST_PATH = os.path.join(EXPERIMENT_DIR, "data", "data_manifest_multi.json")
LOCAL_MODELS_DIR = os.path.join(REPO_ROOT, "models")

PROMPT_TEMPLATE = "The concept of {concept}"
DEFAULT_LAYER_SPEC = "-1"
DEFAULT_CACHE_DIR = os.path.join(EXPERIMENT_DIR, "results", "replication", "cache")
CACHE_SCHEMA_VERSION = "1.1.0"
DEFAULT_TEXT_TEMPLATE_SET = "baseline3"
ALIGNED5_FRACTIONS = [0.00, 0.25, 0.50, 0.75, 1.00]
LOCAL_MLX_MODEL_OVERRIDES = {
    "Qwen3-1.7B-MLX-8bit": os.path.join(LOCAL_MODELS_DIR, "qwen3-1.7B-mlx-8bit"),
    "Qwen3-4B-MLX-8bit": os.path.join(LOCAL_MODELS_DIR, "qwen3-4B-mlx-8bit"),
}

TEXT_TEMPLATE_SETS = {
    "baseline1": {
        "t0": "The concept of {concept}",
    },
    "baseline3": {
        "t0": "The concept of {concept}",
        "t1": "An example of {concept}",
        "t2": "The meaning of {concept}",
    },
}


class ConfigurationError(RuntimeError):
    """Raised when user configuration or CLI options are invalid."""


class DataIntegrityError(RuntimeError):
    """Raised when cache, manifest, or output invariants are violated."""


class ExtractionError(RuntimeError):
    """Raised when model extraction fails in a recoverable way."""


class AnalysisError(RuntimeError):
    """Raised when analysis preconditions are unmet."""

# --- Configurations ---

LANGUAGE_MODELS = {
    "Qwen2.5-1.5B-Instruct-4bit": {
        "id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "type": "causal",
        "param_size_b": 1.54,
        "backend": "mlx",
        "quantization": "4bit",
    },
    "Qwen2.5-1.5B-Instruct-8bit": {
        "id": "mlx-community/Qwen2.5-1.5B-Instruct-8bit",
        "type": "causal",
        "param_size_b": 1.54,
        "backend": "mlx",
        "quantization": "8bit",
    },
    "Falcon3-1B-Instruct-4bit": {
        "id": "mlx-community/Falcon3-1B-Instruct-4bit",
        "type": "causal",
        "param_size_b": 1.0,
        "backend": "mlx",
        "quantization": "4bit",
    },
    "Falcon3-1B-Instruct-8bit": {
        "id": "mlx-community/Falcon3-1B-Instruct-8bit",
        "type": "causal",
        "param_size_b": 1.0,
        "backend": "mlx",
        "quantization": "8bit",
    },
    "Granite-3.3-2B-Instruct-4bit": {
        "id": "mlx-community/granite-3.3-2b-instruct-4bit",
        "type": "causal",
        "param_size_b": 2.0,
        "backend": "mlx",
        "quantization": "4bit",
    },
    "Granite-3.3-2B-Instruct-8bit": {
        "id": "mlx-community/granite-3.3-2b-instruct-8bit",
        "type": "causal",
        "param_size_b": 2.0,
        "backend": "mlx",
        "quantization": "8bit",
    },
    "Qwen3-0.6B-MLX-4bit": {
        "id": "Qwen/Qwen3-0.6B-MLX-4bit",
        "type": "causal",
        "embedding_dim": 1024,
        "param_size_b": 0.6,
        "backend": "mlx",
        "quantization": "4bit",
    },
    "Qwen3-0.6B-MLX-8bit": {
        "id": "Qwen/Qwen3-0.6B-MLX-8bit",
        "type": "causal",
        "embedding_dim": 1024,
        "param_size_b": 0.6,
        "backend": "mlx",
        "quantization": "8bit",
    },
    "Qwen3-1.7B-MLX-4bit": {
        "id": "Qwen/Qwen3-1.7B-MLX-4bit",
        "type": "causal",
        "embedding_dim": 2048,
        "param_size_b": 1.7,
        "backend": "mlx",
        "quantization": "4bit",
    },
    "Qwen3-1.7B-MLX-8bit": {
        "id": "Qwen/Qwen3-1.7B-MLX-8bit",
        "type": "causal",
        "embedding_dim": 2048,
        "param_size_b": 1.7,
        "backend": "mlx",
        "quantization": "8bit",
    },
    "Qwen3-4B-MLX-4bit": {
        "id": "Qwen/Qwen3-4B-MLX-4bit",
        "type": "causal",
        "embedding_dim": 2560,
        "param_size_b": 4.0,
        "backend": "mlx",
        "quantization": "4bit",
    },
    "Qwen3-4B-MLX-8bit": {
        "id": "Qwen/Qwen3-4B-MLX-8bit",
        "type": "causal",
        "embedding_dim": 2560,
        "param_size_b": 4.0,
        "backend": "mlx",
        "quantization": "8bit",
    },
    "LFM2-2.6B-Exp-4bit": {
        "id": "mlx-community/LFM2-2.6B-Exp-4bit",
        "type": "causal",
        "embedding_dim": 2048,
        "param_size_b": 2.6,
        "backend": "mlx",
        "quantization": "4bit",
    },
    "LFM2-2.6B-Exp-8bit": {
        "id": "mlx-community/LFM2-2.6B-Exp-8bit",
        "type": "causal",
        "embedding_dim": 2048,
        "param_size_b": 2.6,
        "backend": "mlx",
        "quantization": "8bit",
    },
    "SmolLM3-3B-4bit": {
        "id": "mlx-community/SmolLM3-3B-4bit",
        "type": "causal",
        "embedding_dim": 2048,
        "param_size_b": 3.0,
        "backend": "mlx",
        "quantization": "4bit",
    },
    "SmolLM3-3B-8bit": {
        "id": "mlx-community/SmolLM3-3B-8bit",
        "type": "causal",
        "embedding_dim": 2048,
        "param_size_b": 3.0,
        "backend": "mlx",
        "quantization": "8bit",
    }
}

VISION_MODELS_SSL = {
    "DINOv2-base": {
        "id": "facebook/dinov2-base",
        "type": "vision",
        "method": "cls",
        "embedding_dim": 768,
    },
    "DINOv2-small": {
        "id": "facebook/dinov2-small",
        "type": "vision",
        "method": "cls",
        "embedding_dim": 384,
    },
    "DINOv3-base": {
        "id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "type": "vision",
        "method": "cls",
        "embedding_dim": 768,
    },
    "DINOv3-large": {
        "id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "type": "vision",
        "method": "cls",
        "embedding_dim": 1024,
    },
    "ViT-MAE-base": {
        "id": "facebook/vit-mae-base",
        "type": "vision",
        "method": "cls",
        "embedding_dim": 768,
    },
    "BEiT-base": {
        "id": "microsoft/beit-base-patch16-224",
        "type": "vision",
        "method": "cls",
        "embedding_dim": 768,
    },
    "data2vec-vision": {
        "id": "facebook/data2vec-vision-base",
        "type": "vision",
        "method": "mean",
        "embedding_dim": 768,
    },
    "Hiera-base": {
        "id": "facebook/hiera-base-224-in1k-hf",
        "type": "vision",
        "method": "pooler",
        "embedding_dim": 768,
    },
    "ConvNeXt-v2": {
        "id": "facebook/convnextv2-base-1k-224",
        "type": "vision",
        "method": "pooler",
        "embedding_dim": 1024,
    },
    "I-JEPA": {
        "id": "facebook/ijepa_vith14_1k",
        "type": "vision",
        "method": "mean",
        "embedding_dim": 1280,
    },
    "AIMv2": {
        "id": "apple/aimv2-large-patch14-224",
        "type": "vision",
        "method": "mean",
        "trust_remote_code": True,
        "embedding_dim": 1024,
    },
    "DINOv3-ConvNeXt-tiny": {
        "id": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "type": "vision",
        "method": "pooler",
    },
    "ViT-MSN-base": {
        "id": "facebook/vit-msn-base",
        "type": "vision",
        "method": "cls",
        "embedding_dim": 768,
    },
}

VISION_MODELS_VLM = {
    "CLIP-ViT-B32": {
        "id": "openai/clip-vit-base-patch32",
        "type": "vision_language",
        "embedding_dim": 512,
    },
    "MetaCLIP-B32-400m": {
        "id": "facebook/metaclip-b32-400m",
        "type": "vision_language",
        "embedding_dim": 512,
    },
    "SigLIP": {
        "id": "google/siglip-base-patch16-224",
        "type": "vision_language",
        "embedding_dim": 768,
    },
    "SigLIP2": {
        "id": "google/siglip2-base-patch16-224",
        "type": "vision_language",
        "embedding_dim": 768,
    },
}

# Concepts and Image Files
base_concepts = [
    "fire", "water", "forest", "city", "space",
    "cat", "dog", "bird", "fish", "elephant",
    "mountain", "ocean", "sun", "moon", "storm",
    "car", "bridge", "road", "building", "airplane"
]

compound_concepts_v1 = [
    ("forest", "fire", "forest fire"),
    ("space", "city", "space city"),
    ("water", "city", "water city"),
    ("city", "forest", "city forest"),
]

compound_concepts_v2 = [
    ("mountain", "road", "mountain road"),
    ("ocean", "bridge", "ocean bridge"),
    ("city", "bridge", "city bridge"),
    ("mountain", "forest", "mountain forest"),
]

compound_concepts = compound_concepts_v1 + compound_concepts_v2

all_concepts = base_concepts + [c[2] for c in compound_concepts]


def _as_sorted_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def deterministic_fingerprint(obj: Any) -> str:
    """Return a deterministic SHA256 hash for a JSON-serializable object."""
    payload = _as_sorted_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_bool_arg(value: str) -> bool:
    val = str(value).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigurationError(
        f"Invalid boolean value '{value}'. Use one of: true/false, 1/0, yes/no."
    )


def raise_model_load_error(
    *,
    model_id: str,
    component: str,
    local_files_only: bool,
    original_error: Exception,
) -> None:
    """Raise an actionable extraction error for model loading failures."""
    if local_files_only:
        raise ExtractionError(
            f"Failed to load {component} for '{model_id}' with --local-files-only=true. "
            "Required files are not fully present in local cache. "
            "Pre-download the model artifacts or rerun with --local-files-only=false. "
            f"Original error: {original_error}"
        ) from original_error
    raise ExtractionError(
        f"Failed to load {component} for '{model_id}'. Original error: {original_error}"
    ) from original_error


def build_manifest_fingerprint(manifest: Dict[str, Any]) -> str:
    concept_to_images = manifest.get("concept_to_images", {})
    concept_metadata = manifest.get("concept_metadata", {})
    source_payload = {
        concept: {
            "source": concept_metadata.get(concept, {}).get("source"),
            "image_sources": concept_metadata.get(concept, {}).get("image_sources", {}),
            "images": concept_to_images.get(concept, []),
        }
        for concept in sorted(concept_to_images.keys())
    }
    return deterministic_fingerprint(
        {
            "manifest_version": manifest.get("manifest_version"),
            "images_per_concept_target": manifest.get("images_per_concept_target"),
            "concept_payload": source_payload,
        }
    )


def build_model_config_fingerprint(model_name: str, config: Dict[str, Any]) -> str:
    return deterministic_fingerprint({"model_name": model_name, "config": config})


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=os.path.dirname(path), delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=True)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def atomic_write_npy(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=os.path.dirname(path), delete=False, suffix=".tmp"
    ) as tmp:
        np.save(tmp, array)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def get_text_templates(template_set: str) -> Dict[str, str]:
    if template_set not in TEXT_TEMPLATE_SETS:
        raise ConfigurationError(
            f"Unknown --text-template-set '{template_set}'. "
            f"Available: {', '.join(sorted(TEXT_TEMPLATE_SETS.keys()))}"
        )
    templates = TEXT_TEMPLATE_SETS[template_set]
    if "t0" not in templates:
        raise ConfigurationError(f"Template set '{template_set}' must contain baseline key 't0'.")
    return templates


def apply_local_mlx_override(
    model_name: str,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Use local MLX model path overrides when available and complete."""
    local_path = LOCAL_MLX_MODEL_OVERRIDES.get(model_name)
    if not local_path:
        return config

    required_files = [
        "model.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required_files if not os.path.isfile(os.path.join(local_path, name))]
    if missing:
        logger.warning(
            "Local model override unavailable for %s (missing: %s); using '%s'.",
            model_name,
            ", ".join(missing),
            config["id"],
        )
        return config

    if config.get("id") != local_path:
        updated = dict(config)
        updated["id"] = local_path
        logger.info("Using local model override for %s: %s", model_name, local_path)
        return updated
    return config


def normalize_layer_spec_text(layer_spec: str) -> str:
    raw = str(layer_spec).strip()
    if not raw:
        return DEFAULT_LAYER_SPEC
    return ",".join(part.strip() for part in raw.split(","))


def layer_profile_id_from_spec(layer_spec: str) -> str:
    normalized = normalize_layer_spec_text(layer_spec).lower()
    if normalized in {"-1", "last", "selected"}:
        return "baseline_last"
    if normalized == "aligned5":
        return "aligned5"
    if normalized == "all":
        return "all_layers"
    safe = "".join(ch if ch.isalnum() else "_" for ch in normalized).strip("_")
    return f"custom_{safe}" if safe else "custom_spec"


def ensure_mlx_available() -> Tuple[Any, Any]:
    """Lazily import MLX stack only when an MLX model is requested."""
    global mx, mlx_load, HAS_MLX, MLX_IMPORT_ERROR
    if HAS_MLX and mx is not None and mlx_load is not None:
        return mx, mlx_load
    if MLX_IMPORT_ERROR is not None:
        raise ExtractionError(f"MLX backend unavailable: {MLX_IMPORT_ERROR}")
    try:
        import mlx.core as _mx
        from mlx_lm import load as _mlx_load
    except Exception as exc:
        MLX_IMPORT_ERROR = exc
        raise ExtractionError(f"MLX backend unavailable: {exc}") from exc
    mx = _mx
    mlx_load = _mlx_load
    HAS_MLX = True
    return mx, mlx_load


def default_cache_manifest(
    model_name: str,
    model_config: Dict[str, Any],
    manifest_fingerprint: str,
    concept_to_images: Dict[str, List[str]],
    layer_keys: List[str],
    embedding_dim: int,
    layer_profile_id: str,
    requested_layers_spec: str,
    text_templates: Optional[Dict[str, str]] = None,
    layer_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": model_name,
        "model_config_fingerprint": build_model_config_fingerprint(model_name, model_config),
        "manifest_fingerprint": manifest_fingerprint,
        "layer_keys": list(layer_keys),
        "layer_profile_id": layer_profile_id,
        "requested_layers_spec": requested_layers_spec,
        "concept_to_images": concept_to_images,
        "dtype": "float32",
        "embedding_dim": int(embedding_dim),
        "created_at": datetime.now().isoformat(),
        "layer_metadata": layer_metadata or {},
    }
    if text_templates is not None:
        payload["text_templates"] = text_templates
    return payload


def validate_cache_manifest_payload(
    payload: Dict[str, Any],
    expected_model_name: str,
    expected_model_config: Dict[str, Any],
    expected_manifest_fingerprint: str,
    expected_layer_profile_id: str,
    expected_requested_layers_spec: str,
) -> None:
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
        raise DataIntegrityError(
            f"Cache manifest missing fields for {expected_model_name}: {missing}"
        )
    if payload["schema_version"] != CACHE_SCHEMA_VERSION:
        raise DataIntegrityError(
            f"Unsupported cache schema for {expected_model_name}: "
            f"{payload['schema_version']} != {CACHE_SCHEMA_VERSION}"
        )
    if payload["model_name"] != expected_model_name:
        raise DataIntegrityError(
            f"Cache model mismatch: expected {expected_model_name}, got {payload['model_name']}"
        )
    expected_model_fp = build_model_config_fingerprint(expected_model_name, expected_model_config)
    if payload["model_config_fingerprint"] != expected_model_fp:
        raise DataIntegrityError(
            f"Cache model fingerprint mismatch for {expected_model_name}. "
            "Use --force-cache-rebuild to refresh stale cache."
        )
    if payload["manifest_fingerprint"] != expected_manifest_fingerprint:
        raise DataIntegrityError(
            f"Cache manifest fingerprint mismatch for {expected_model_name}. "
            "Manifest changed; rerun with --force-cache-rebuild."
        )
    if payload.get("layer_profile_id") != expected_layer_profile_id:
        raise DataIntegrityError(
            f"Cache layer profile mismatch for {expected_model_name}: "
            f"{payload.get('layer_profile_id')} != {expected_layer_profile_id}. "
            "Rerun with --force-cache-rebuild."
        )
    normalized_payload_spec = normalize_layer_spec_text(payload.get("requested_layers_spec", ""))
    normalized_expected_spec = normalize_layer_spec_text(expected_requested_layers_spec)
    if normalized_payload_spec != normalized_expected_spec:
        raise DataIntegrityError(
            f"Cache requested layer spec mismatch for {expected_model_name}: "
            f"{normalized_payload_spec} != {normalized_expected_spec}. "
            "Rerun with --force-cache-rebuild."
        )
    layer_keys = payload.get("layer_keys", [])
    if not isinstance(layer_keys, list) or not layer_keys:
        raise DataIntegrityError(f"Cache manifest layer_keys invalid for {expected_model_name}.")
    if payload.get("dtype") != "float32":
        raise DataIntegrityError(
            f"Cache dtype must be float32 for {expected_model_name}, got {payload.get('dtype')}."
        )
    if not isinstance(payload.get("embedding_dim"), int) or payload["embedding_dim"] <= 0:
        raise DataIntegrityError(
            f"Cache embedding_dim invalid for {expected_model_name}: {payload.get('embedding_dim')}"
        )


def load_cache_manifest(
    cache_manifest_path: str,
    model_name: str,
    model_config: Dict[str, Any],
    manifest_fingerprint: str,
    layer_profile_id: str,
    requested_layers_spec: str,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_manifest_path):
        return None
    with open(cache_manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    validate_cache_manifest_payload(
        payload,
        model_name,
        model_config,
        manifest_fingerprint,
        layer_profile_id,
        requested_layers_spec,
    )
    return payload


def ensure_cache_array(
    array: np.ndarray,
    expected_rows: int,
    expected_dim: Optional[int],
    cache_path: str,
) -> np.ndarray:
    if array.ndim != 2:
        raise DataIntegrityError(f"Cache array must be rank-2 at {cache_path}, got shape={array.shape}")
    if array.shape[0] != expected_rows:
        raise DataIntegrityError(
            f"Cache row mismatch at {cache_path}: expected {expected_rows}, got {array.shape[0]}"
        )
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise DataIntegrityError(
            f"Cache dim mismatch at {cache_path}: expected {expected_dim}, got {array.shape[1]}"
        )
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return array


def cache_file_path(cache_model_dir: str, layer_name: str, concept: str) -> str:
    return os.path.join(cache_model_dir, layer_name, f"{concept}.npy")


def load_cached_per_image_by_layer(
    cache_model_dir: str,
    layer_keys: List[str],
    concept: str,
    expected_rows: int,
    expected_dim: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    per_image: Dict[str, np.ndarray] = {}
    for layer_name in layer_keys:
        path = cache_file_path(cache_model_dir, layer_name, concept)
        if not os.path.exists(path):
            return None
        try:
            array = np.load(path)
        except Exception as exc:  # pragma: no cover - defensive for corrupt files
            raise DataIntegrityError(f"Failed to read cache file {path}: {exc}") from exc
        per_image[layer_name] = ensure_cache_array(array, expected_rows, expected_dim, path)
    return per_image


def save_cached_per_image_by_layer(
    cache_model_dir: str,
    concept: str,
    per_image_by_layer: Dict[str, np.ndarray],
) -> None:
    for layer_name, array in per_image_by_layer.items():
        path = cache_file_path(cache_model_dir, layer_name, concept)
        atomic_write_npy(path, np.asarray(array, dtype=np.float32))


def load_data_manifest():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Data manifest not found at {MANIFEST_PATH}. "
            "Ensure data/data_manifest_multi.json and data/images_multi are present."
        )

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    concept_to_images = manifest.get("concept_to_images", {})
    if not concept_to_images:
        raise ValueError("Manifest is missing 'concept_to_images' mapping.")

    return manifest, concept_to_images


def validate_manifest_images(required_concepts, concept_to_images):
    missing_concepts = [c for c in required_concepts if c not in concept_to_images]
    if missing_concepts:
        raise ValueError(
            "Manifest is missing concepts: "
            + ", ".join(sorted(missing_concepts))
        )

    empty_concepts = [c for c in required_concepts if not concept_to_images[c]]
    if empty_concepts:
        raise ValueError(
            "Concepts with no images: "
            + ", ".join(sorted(empty_concepts))
        )

    missing_files = []
    for concept in required_concepts:
        for rel_path in concept_to_images[concept]:
            abs_path = os.path.join(EXPERIMENT_DIR, rel_path)
            if not os.path.exists(abs_path):
                missing_files.append(f"{concept}: {rel_path}")

    if missing_files:
        details = "\n".join(missing_files)
        raise FileNotFoundError(
            "Missing image files referenced by manifest:\n" + details
        )


def get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "unknown"


def get_environment_metadata():
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "transformers_version": transformers.__version__,
        "git_commit": get_git_commit(),
    }


def parse_layer_spec(layer_spec: str) -> Dict[str, Any]:
    """Parse a layer selector like '-1', '0,4,8', 'all', or 'aligned5'."""
    raw = (layer_spec or DEFAULT_LAYER_SPEC).strip().lower()
    if raw in {"all", "*"}:
        return {"mode": "all", "indices": []}
    if raw == "aligned5":
        return {"mode": "aligned5", "fractions": ALIGNED5_FRACTIONS}

    indices: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            indices.append(int(token))
        except ValueError as exc:
            raise ValueError(
                f"Invalid layer token '{token}' in --layers='{layer_spec}'. "
                "Use comma-separated integers or 'all'."
            ) from exc

    if not indices:
        raise ValueError(
            f"Invalid --layers='{layer_spec}'. Use comma-separated integers or 'all'."
        )
    return {"mode": "explicit", "indices": indices}


def resolve_requested_layers(num_layers: int, layer_spec: Dict[str, Any]) -> List[int]:
    """Resolve requested layer indices into canonical 0-based layer ids."""
    if num_layers <= 0:
        raise ValueError(f"Model reports no hidden layers (num_layers={num_layers}).")

    if layer_spec["mode"] == "all":
        return list(range(num_layers))
    if layer_spec["mode"] == "aligned5":
        if num_layers == 1:
            return [0]
        resolved = []
        for fraction in layer_spec["fractions"]:
            idx = int(round(fraction * (num_layers - 1)))
            idx = max(0, min(num_layers - 1, idx))
            if idx not in resolved:
                resolved.append(idx)
        return resolved

    resolved: List[int] = []
    for raw_idx in layer_spec["indices"]:
        idx = num_layers + raw_idx if raw_idx < 0 else raw_idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(
                f"Requested layer {raw_idx} resolves to {idx}, "
                f"outside valid range [0, {num_layers - 1}]."
            )
        if idx not in resolved:
            resolved.append(idx)
    return resolved


def layer_key(layer_idx: int) -> str:
    return f"layer_{layer_idx}"


def infer_hidden_state_layout(hidden_states: Tuple[torch.Tensor, ...], model: Any) -> Tuple[int, int]:
    """Infer (offset, num_layers) where offset skips token-embedding hidden state when present."""
    total_states = len(hidden_states)
    config = getattr(model, "config", None)
    config_layers = getattr(config, "num_hidden_layers", None)

    if isinstance(config_layers, int):
        if total_states == config_layers + 1:
            return 1, config_layers
        if total_states == config_layers:
            return 0, config_layers

    # Fallback: most transformer outputs include an embedding state at index 0.
    offset = 1 if total_states > 1 else 0
    num_layers = max(total_states - offset, 1)
    return offset, num_layers


def is_last_layer_only_request(layer_spec: Dict[str, Any]) -> bool:
    return (
        layer_spec["mode"] == "explicit"
        and len(layer_spec["indices"]) == 1
        and layer_spec["indices"][0] == -1
    )

# --- Embedding Extraction Functions ---

def get_causal_lm_embeddings_by_layer(
    word: str,
    tokenizer: Any,
    model: Any,
    layer_spec: Dict[str, Any],
    prompt_template: str = PROMPT_TEMPLATE,
    force_cpu: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Extract per-layer last-token embeddings from causal language models."""
    prompt = prompt_template.format(concept=word)
    target_device = "cpu" if force_cpu else device
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ExtractionError(
            f"Model '{type(model).__name__}' did not return hidden_states for prompt '{prompt}'."
        )

    offset, num_layers = infer_hidden_state_layout(hidden_states, model)
    selected_layers = resolve_requested_layers(num_layers, layer_spec)

    embeddings_by_layer: Dict[str, np.ndarray] = {}
    for idx in selected_layers:
        hs_index = idx + offset
        vector = hidden_states[hs_index][:, -1, :].cpu().float().numpy().flatten()
        embeddings_by_layer[layer_key(idx)] = vector

    layer_info = {
        "supports_layer_selection": True,
        "num_hidden_states": len(hidden_states),
        "offset": offset,
        "num_layers": num_layers,
        "selected_layer_indices": selected_layers,
        "selected_layer_keys": [layer_key(i) for i in selected_layers],
        "default_layer_key": layer_key(selected_layers[-1]),
    }
    return embeddings_by_layer, layer_info


def get_mlx_causal_lm_embedding(
    word: str,
    tokenizer: Any,
    model: Any,
    prompt_template: str = PROMPT_TEMPLATE,
) -> np.ndarray:
    prompt = prompt_template.format(concept=word)
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise RuntimeError(f"Tokenizer returned no tokens for prompt: {prompt!r}")

    input_ids = mx.array([token_ids])

    # In mlx-lm, model(...) returns logits; model.model(...) returns final hidden states.
    hidden_states = model.model(input_ids, cache=None)
    embedding = hidden_states[:, -1, :]
    return np.array(embedding.tolist()[0], dtype=np.float32).flatten()


def get_vision_ssl_embeddings_by_layer(
    image_path: str,
    processor: Any,
    model: Any,
    layer_spec: Dict[str, Any],
    method: str = "cls",
    target_device: Any = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Extract per-layer embeddings for one image from vision SSL models."""
    if target_device is None:
        target_device = device
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(target_device)

    if method == "pooler":
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output"):
                embedding = outputs.pooler_output.cpu().numpy().flatten()
            else:
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        return {
            "layer_last": embedding
        }, {
            "supports_layer_selection": False,
            "reason": "pooler extraction path does not expose comparable intermediate layers",
            "selected_layer_keys": ["layer_last"],
            "default_layer_key": "layer_last",
        }

    with torch.no_grad():
        try:
            outputs = model(**inputs, output_hidden_states=True)
        except TypeError:
            outputs = model(**inputs)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        offset, num_layers = infer_hidden_state_layout(hidden_states, model)
        selected_layers = resolve_requested_layers(num_layers, layer_spec)

        embeddings_by_layer: Dict[str, np.ndarray] = {}
        for idx in selected_layers:
            hs = hidden_states[idx + offset].cpu()
            if method == "cls":
                emb = hs[:, 0, :]
            else:
                emb = hs.mean(dim=1)
            embeddings_by_layer[layer_key(idx)] = emb.numpy().flatten()

        return embeddings_by_layer, {
            "supports_layer_selection": True,
            "num_hidden_states": len(hidden_states),
            "offset": offset,
            "num_layers": num_layers,
            "selected_layer_indices": selected_layers,
            "selected_layer_keys": [layer_key(i) for i in selected_layers],
            "default_layer_key": layer_key(selected_layers[-1]),
        }

    with torch.no_grad():
        last_hidden_state = outputs.last_hidden_state.cpu()
        if method == "cls":
            embedding = last_hidden_state[:, 0, :].numpy().flatten()
        else:
            embedding = last_hidden_state.mean(dim=1).numpy().flatten()
    return {
        "layer_last": embedding
    }, {
        "supports_layer_selection": False,
        "reason": "model output did not include hidden_states",
        "selected_layer_keys": ["layer_last"],
        "default_layer_key": "layer_last",
    }


def get_vision_ssl_embedding_multi_by_layer(
    image_paths: List[str],
    processor: Any,
    model: Any,
    layer_spec: Dict[str, Any],
    method: str = "cls",
    target_device: Any = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, np.ndarray]]:
    """Extract and average per-layer embeddings across images for one concept."""
    layer_stacks: Dict[str, List[np.ndarray]] = {}
    layer_info: Dict[str, Any] = {}

    for path in image_paths:
        emb_by_layer, current_info = get_vision_ssl_embeddings_by_layer(
            path, processor, model, layer_spec, method, target_device
        )
        if not layer_stacks:
            layer_stacks = {k: [v] for k, v in emb_by_layer.items()}
            layer_info = current_info
        else:
            if set(emb_by_layer.keys()) != set(layer_stacks.keys()):
                raise RuntimeError(
                    "Inconsistent layer keys across images. "
                    f"Expected={sorted(layer_stacks.keys())} got={sorted(emb_by_layer.keys())}"
                )
            for key, value in emb_by_layer.items():
                layer_stacks[key].append(value)

    per_image_arrays = {
        key: np.stack(values, axis=0).astype(np.float32)
        for key, values in layer_stacks.items()
    }
    averaged = {
        key: arr.mean(axis=0).astype(np.float32)
        for key, arr in per_image_arrays.items()
    }
    return averaged, layer_info, per_image_arrays


def get_siglip_embedding(image_path: str, processor: Any, model: Any, target_device: Any = None) -> np.ndarray:
    """Extract embedding from a single image using SigLIP / SigLIP2 / CLIP-style heads."""
    if target_device is None:
        target_device = device
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(target_device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    # get_image_features may return a tensor or an object with .image_embeds
    if isinstance(outputs, torch.Tensor):
        return outputs.cpu().numpy().flatten()
    if hasattr(outputs, "image_embeds"):
        return outputs.image_embeds.cpu().numpy().flatten()
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output.cpu().numpy().flatten()
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    raise RuntimeError(f"Unexpected output type from get_image_features: {type(outputs)}")


def get_siglip_embedding_multi(
    image_paths: List[str],
    processor: Any,
    model: Any,
    target_device: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and average SigLIP/SigLIP2 embeddings across multiple images."""
    all_emb: List[np.ndarray] = []
    for path in image_paths:
        emb = get_siglip_embedding(path, processor, model, target_device)
        all_emb.append(np.asarray(emb, dtype=np.float32))
    per_image = np.stack(all_emb, axis=0).astype(np.float32)
    return per_image.mean(axis=0).astype(np.float32), per_image

# --- Analysis Functions ---

def calculate_similarity(emb1, emb2):
    return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0])

def generate_similarity_matrix(embeddings_dict, concepts):
    n = len(concepts)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            matrix[i, j] = calculate_similarity(embeddings_dict[c1], embeddings_dict[c2])
    return matrix

# --- Main Execution ---

def setup_model_logger(model_name, log_dir):
    """Create a per-model log file that captures all stdout/stderr for this run."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{model_name}_{ts}.log")

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — verbose, full tracebacks
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)

    # Console handler — same output so terminal still shows everything
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger, log_path


def run_replication_for_model(
    model_name: str,
    force: bool = False,
    layers_arg: str = DEFAULT_LAYER_SPEC,
    cache_image_embeddings: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_cache_rebuild: bool = False,
    text_template_set: str = DEFAULT_TEXT_TEMPLATE_SET,
    local_files_only: bool = False,
    output_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
):
    resolved_log_dir = os.path.abspath(
        log_dir
        or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "results", "replication", "logs",
        )
    )
    logger, log_path = setup_model_logger(model_name, resolved_log_dir)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"PLATONIC REPRESENTATION EXTRACTION: {model_name}")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Requested layers: {layers_arg}")
    logger.info(f"Cache image embeddings: {cache_image_embeddings}")
    logger.info(f"Cache dir: {os.path.abspath(cache_dir)}")
    logger.info(f"Force cache rebuild: {force_cache_rebuild}")
    logger.info(f"Text template set: {text_template_set}")
    logger.info(f"Local files only: {local_files_only}")

    # Setup output directory
    resolved_output_dir = os.path.abspath(
        output_dir or os.path.join(EXPERIMENT_DIR, "results", "replication", "raw_data")
    )
    os.makedirs(resolved_output_dir, exist_ok=True)

    output_file = os.path.join(resolved_output_dir, f"{model_name}.json")
    if os.path.exists(output_file) and not force:
        logger.info(f"Results for {model_name} already exist at {output_file}. Skipping.")
        logger.info("Use --force to overwrite.")
        return

    manifest, concept_to_images = load_data_manifest()
    try:
        layer_spec = parse_layer_spec(layers_arg)
        templates = get_text_templates(text_template_set)
    except (ValueError, ConfigurationError) as e:
        logger.error(f"Configuration error: {e}")
        return
    manifest_fingerprint = build_manifest_fingerprint(manifest)
    requested_layers_spec = normalize_layer_spec_text(layers_arg)
    layer_profile_id = layer_profile_id_from_spec(layers_arg)
    cache_root = os.path.abspath(cache_dir)
    cache_model_dir = os.path.join(cache_root, model_name)
    cache_manifest_path = os.path.join(cache_model_dir, "cache_manifest.json")
    concept_image_map = {concept: concept_to_images[concept] for concept in all_concepts}

    model_result = {
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "environment": get_environment_metadata(),
            "data_manifest_version": manifest.get("manifest_version", "unknown"),
            "text_prompt_template": templates["t0"],
            "text_embedding_method": "last_token",
            "vision_embedding_method": "multi_image_average",
            "requested_layers": layers_arg,
            "images_per_concept_target": manifest.get("images_per_concept_target", "unknown"),
            "manifest_fingerprint": manifest_fingerprint,
            "cache_image_embeddings": cache_image_embeddings,
            "cache_dir": cache_root,
            "force_cache_rebuild": force_cache_rebuild,
            "text_template_set": text_template_set,
            "layer_profile_id": layer_profile_id,
            "requested_layers_spec": requested_layers_spec,
        },
        "concepts": all_concepts,
        "models": {}
    }

    try:
        if model_name in LANGUAGE_MODELS:
            config = LANGUAGE_MODELS[model_name]
            backend = config.get("backend", "hf")
            if backend == "mlx":
                config = apply_local_mlx_override(model_name, config, logger)
            if cache_image_embeddings and not force_cache_rebuild:
                _ = load_cache_manifest(
                    cache_manifest_path,
                    model_name,
                    config,
                    manifest_fingerprint,
                    layer_profile_id,
                    requested_layers_spec,
                )
            template_embeddings_by_layer: Dict[str, Dict[str, Dict[str, List[float]]]] = {
                template_key: {} for template_key in templates
            }
            layer_info: Dict[str, Any] = {}

            if backend == "hf":
                logger.info(f"\nProcessing Language Model {model_name} (HF/CPU for stability)...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        config["id"], local_files_only=local_files_only
                    )
                except Exception as exc:
                    raise_model_load_error(
                        model_id=config["id"],
                        component="tokenizer",
                        local_files_only=local_files_only,
                        original_error=exc,
                    )
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        config["id"], local_files_only=local_files_only
                    ).to("cpu").eval()
                except Exception as exc:
                    raise_model_load_error(
                        model_id=config["id"],
                        component="causal model weights",
                        local_files_only=local_files_only,
                        original_error=exc,
                    )

                for concept in all_concepts:
                    for template_key, template in templates.items():
                        logger.info(
                            f"     {concept}: template={template_key} language extraction"
                        )
                        current_embs, current_info = get_causal_lm_embeddings_by_layer(
                            concept,
                            tokenizer,
                            model,
                            layer_spec,
                            prompt_template=template,
                            force_cpu=True,
                        )
                        if not layer_info:
                            layer_info = current_info
                            for tk in templates:
                                template_embeddings_by_layer[tk] = {
                                    key: {} for key in layer_info["selected_layer_keys"]
                                }
                            logger.info(
                                "     extracting layers: "
                                + ", ".join(layer_info["selected_layer_keys"])
                            )
                        if set(current_embs.keys()) != set(layer_info["selected_layer_keys"]):
                            raise DataIntegrityError(
                                "Inconsistent language layer keys across templates/concepts: "
                                f"expected={layer_info['selected_layer_keys']} "
                                f"got={sorted(current_embs.keys())}"
                            )
                        for layer_name, emb in current_embs.items():
                            template_embeddings_by_layer[template_key][layer_name][concept] = (
                                np.asarray(emb, dtype=np.float32).tolist()
                            )
            elif backend == "mlx":
                logger.info(f"\nProcessing Language Model {model_name} (MLX backend)...")
                _, mlx_loader = ensure_mlx_available()
                if not is_last_layer_only_request(layer_spec):
                    logger.warning(
                        "     MLX backend currently exposes final hidden state only. "
                        "Falling back to last-layer extraction."
                    )
                model, tokenizer = mlx_loader(config["id"])
                layer_info = {
                    "supports_layer_selection": False,
                    "reason": "mlx backend currently returns final hidden state only",
                    "selected_layer_keys": ["layer_last"],
                    "default_layer_key": "layer_last",
                }
                for tk in templates:
                    template_embeddings_by_layer[tk] = {"layer_last": {}}
                for concept in all_concepts:
                    for template_key, template in templates.items():
                        logger.info(
                            f"     {concept}: template={template_key} MLX extraction"
                        )
                        emb = get_mlx_causal_lm_embedding(
                            concept, tokenizer, model, prompt_template=template
                        )
                        template_embeddings_by_layer[template_key]["layer_last"][concept] = (
                            np.asarray(emb, dtype=np.float32).tolist()
                        )
            else:
                raise ConfigurationError(
                    f"Unsupported language backend '{backend}' for model {model_name}"
                )

            baseline_by_layer = template_embeddings_by_layer["t0"]
            default_layer_key = layer_info.get("default_layer_key")
            if default_layer_key not in baseline_by_layer:
                default_layer_key = sorted(baseline_by_layer.keys())[-1]

            model_result["models"][model_name] = {
                "config": config,
                "embeddings": baseline_by_layer[default_layer_key],
                "embeddings_by_layer": baseline_by_layer,
                "layer_metadata": {
                    "requested_layers": layers_arg,
                    **layer_info,
                },
                "text_template_metadata": {
                    "template_set": text_template_set,
                    "baseline_template_key": "t0",
                    "templates": templates,
                },
                "text_template_embeddings_by_layer": template_embeddings_by_layer,
            }

            if cache_image_embeddings:
                os.makedirs(cache_model_dir, exist_ok=True)
                sample_vec = next(iter(baseline_by_layer[default_layer_key].values()))
                cache_payload = default_cache_manifest(
                    model_name=model_name,
                    model_config=config,
                    manifest_fingerprint=manifest_fingerprint,
                    concept_to_images=concept_image_map,
                    layer_keys=list(baseline_by_layer.keys()),
                    embedding_dim=len(sample_vec),
                    layer_profile_id=layer_profile_id,
                    requested_layers_spec=requested_layers_spec,
                    text_templates=templates,
                    layer_metadata={
                        "requested_layers": layers_arg,
                        **layer_info,
                    },
                )
                atomic_write_json(cache_manifest_path, cache_payload)

        elif model_name in VISION_MODELS_SSL:
            config = VISION_MODELS_SSL[model_name]
            validate_manifest_images(all_concepts, concept_to_images)
            trust_rc = config.get("trust_remote_code", False)

            cache_manifest = None
            if cache_image_embeddings and not force_cache_rebuild:
                cache_manifest = load_cache_manifest(
                    cache_manifest_path,
                    model_name,
                    config,
                    manifest_fingerprint,
                    layer_profile_id,
                    requested_layers_spec,
                )
            expected_layer_keys: Optional[List[str]] = (
                list(cache_manifest["layer_keys"]) if cache_manifest else None
            )
            expected_embedding_dim: Optional[int] = (
                int(cache_manifest["embedding_dim"]) if cache_manifest else None
            )
            layer_info: Dict[str, Any] = (
                dict(cache_manifest.get("layer_metadata", {})) if cache_manifest else {}
            )
            embeddings_by_layer: Dict[str, Dict[str, List[float]]] = {}

            processor = None
            model = None
            target_device = device

            def ensure_ssl_model_loaded() -> Tuple[Any, Any, Any]:
                nonlocal processor, model, target_device
                if processor is not None and model is not None:
                    return processor, model, target_device
                logger.info(f"Loading processor for {model_name}...")
                try:
                    processor = AutoImageProcessor.from_pretrained(
                        config["id"],
                        trust_remote_code=trust_rc,
                        local_files_only=local_files_only,
                    )
                except Exception as exc:
                    raise_model_load_error(
                        model_id=config["id"],
                        component="image processor",
                        local_files_only=local_files_only,
                        original_error=exc,
                    )
                logger.info(f"Loading model {model_name} onto {target_device}...")
                try:
                    if trust_rc:
                        import transformers.modeling_utils as _mu
                        _orig = _mu.PreTrainedModel
                        if not hasattr(_orig, "all_tied_weights_keys"):
                            _orig.all_tied_weights_keys = property(
                                lambda self: getattr(self, "_tied_weights_keys", None) or {}
                            )
                    model = AutoModel.from_pretrained(
                        config["id"],
                        trust_remote_code=trust_rc,
                        local_files_only=local_files_only,
                    ).to(target_device).eval()
                    dummy = processor(images=Image.new("RGB", (224, 224)), return_tensors="pt").to(
                        target_device
                    )
                    with torch.no_grad():
                        model(**dummy)
                except Exception as exc:
                    logger.warning(
                        f"  MPS probe failed for {model_name} ({exc}), falling back to CPU..."
                    )
                    logger.warning(traceback.format_exc())
                    target_device = torch.device("cpu")
                    try:
                        model = AutoModel.from_pretrained(
                            config["id"],
                            trust_remote_code=trust_rc,
                            local_files_only=local_files_only,
                        ).to(target_device).eval()
                    except Exception as cpu_exc:
                        raise_model_load_error(
                            model_id=config["id"],
                            component="vision model weights (CPU fallback)",
                            local_files_only=local_files_only,
                            original_error=cpu_exc,
                        )
                logger.info(f"\nProcessing Vision SSL Model {model_name} ({target_device})...")
                return processor, model, target_device

            for concept in all_concepts:
                image_paths = [os.path.join(EXPERIMENT_DIR, p) for p in concept_to_images[concept]]
                per_image_by_layer = None
                if cache_image_embeddings and not force_cache_rebuild and expected_layer_keys:
                    per_image_by_layer = load_cached_per_image_by_layer(
                        cache_model_dir=cache_model_dir,
                        layer_keys=expected_layer_keys,
                        concept=concept,
                        expected_rows=len(image_paths),
                        expected_dim=expected_embedding_dim,
                    )
                    if per_image_by_layer is not None:
                        logger.info(
                            f"     {concept}: loaded cached embeddings ({len(image_paths)} images)"
                        )

                if per_image_by_layer is None:
                    processor, model, target_device = ensure_ssl_model_loaded()
                    logger.info(f"     {concept}: extracting and caching {len(image_paths)} images")
                    _, current_info, per_image_by_layer = get_vision_ssl_embedding_multi_by_layer(
                        image_paths, processor, model, layer_spec, config["method"], target_device
                    )
                    current_keys = list(current_info["selected_layer_keys"])
                    if expected_layer_keys is None:
                        expected_layer_keys = current_keys
                    elif current_keys != expected_layer_keys:
                        raise DataIntegrityError(
                            f"Layer key mismatch for {model_name}/{concept}: "
                            f"expected={expected_layer_keys}, got={current_keys}"
                        )
                    if not layer_info:
                        layer_info = current_info
                    if cache_image_embeddings:
                        os.makedirs(cache_model_dir, exist_ok=True)
                        save_cached_per_image_by_layer(cache_model_dir, concept, per_image_by_layer)

                if expected_layer_keys is None:
                    expected_layer_keys = list(per_image_by_layer.keys())
                if not embeddings_by_layer:
                    for layer_name in expected_layer_keys:
                        embeddings_by_layer[layer_name] = {}
                if not layer_info:
                    layer_info = {
                        "supports_layer_selection": len(expected_layer_keys) > 1,
                        "selected_layer_keys": expected_layer_keys,
                        "default_layer_key": expected_layer_keys[-1],
                        "reason": "layer metadata unavailable; inferred from cache",
                    }
                for layer_name in expected_layer_keys:
                    arr = ensure_cache_array(
                        np.asarray(per_image_by_layer[layer_name], dtype=np.float32),
                        expected_rows=len(image_paths),
                        expected_dim=expected_embedding_dim,
                        cache_path=cache_file_path(cache_model_dir, layer_name, concept),
                    )
                    if expected_embedding_dim is None:
                        expected_embedding_dim = int(arr.shape[1])
                    averaged = arr.mean(axis=0).astype(np.float32)
                    embeddings_by_layer[layer_name][concept] = averaged.tolist()

            if not layer_info:
                raise DataIntegrityError(
                    f"Could not infer layer metadata for {model_name}. "
                    "Delete cache or rerun with --force-cache-rebuild."
                )
            default_layer_key = layer_info.get("default_layer_key", expected_layer_keys[-1])
            if default_layer_key not in embeddings_by_layer:
                default_layer_key = sorted(embeddings_by_layer.keys())[-1]
            model_result["models"][model_name] = {
                "config": config,
                "embeddings": embeddings_by_layer[default_layer_key],
                "embeddings_by_layer": embeddings_by_layer,
                "layer_metadata": {
                    "requested_layers": layers_arg,
                    **layer_info,
                },
            }
            if cache_image_embeddings:
                sample_vec = next(iter(embeddings_by_layer[default_layer_key].values()))
                cache_payload = default_cache_manifest(
                    model_name=model_name,
                    model_config=config,
                    manifest_fingerprint=manifest_fingerprint,
                    concept_to_images=concept_image_map,
                    layer_keys=list(embeddings_by_layer.keys()),
                    embedding_dim=len(sample_vec),
                    layer_profile_id=layer_profile_id,
                    requested_layers_spec=requested_layers_spec,
                    layer_metadata={
                        "requested_layers": layers_arg,
                        **layer_info,
                    },
                )
                atomic_write_json(cache_manifest_path, cache_payload)

        elif model_name in VISION_MODELS_VLM:
            config = VISION_MODELS_VLM[model_name]
            validate_manifest_images(all_concepts, concept_to_images)
            if not is_last_layer_only_request(layer_spec):
                logger.warning(
                    "Vision-language image-feature extraction currently returns final image features only. "
                    "Requested multi-layer selection will fall back to the final layer."
                )

            cache_manifest = None
            if cache_image_embeddings and not force_cache_rebuild:
                cache_manifest = load_cache_manifest(
                    cache_manifest_path,
                    model_name,
                    config,
                    manifest_fingerprint,
                    layer_profile_id,
                    requested_layers_spec,
                )
            expected_layer_keys = list(cache_manifest["layer_keys"]) if cache_manifest else ["layer_last"]
            if expected_layer_keys != ["layer_last"]:
                raise DataIntegrityError(
                    f"VLM cache must use ['layer_last'], found {expected_layer_keys}"
                )
            expected_embedding_dim: Optional[int] = (
                int(cache_manifest["embedding_dim"]) if cache_manifest else None
            )

            processor = None
            model = None
            target_device = device

            def ensure_vlm_loaded() -> Tuple[Any, Any, Any]:
                nonlocal processor, model, target_device
                if processor is not None and model is not None:
                    return processor, model, target_device
                logger.info(f"Loading processor for {model_name}...")
                try:
                    processor = AutoProcessor.from_pretrained(
                        config["id"], local_files_only=local_files_only
                    )
                except Exception as exc:
                    logger.warning(
                        "  AutoProcessor load failed for %s (%s). "
                        "Falling back to AutoImageProcessor for image-feature extraction.",
                        model_name,
                        exc,
                    )
                    try:
                        processor = AutoImageProcessor.from_pretrained(
                            config["id"], local_files_only=local_files_only
                        )
                    except Exception as img_exc:
                        raise_model_load_error(
                            model_id=config["id"],
                            component="processor",
                            local_files_only=local_files_only,
                            original_error=img_exc,
                        )
                logger.info(f"Loading model {model_name} onto {target_device}...")
                try:
                    model = AutoModel.from_pretrained(
                        config["id"], local_files_only=local_files_only
                    ).to(target_device).eval()
                    dummy = processor(
                        images=Image.new("RGB", (224, 224)), return_tensors="pt"
                    ).to(target_device)
                    with torch.no_grad():
                        dummy_out = model.get_image_features(**dummy)
                    if not isinstance(dummy_out, torch.Tensor):
                        _ = getattr(dummy_out, "image_embeds", None) or \
                            getattr(dummy_out, "pooler_output", None) or \
                            getattr(dummy_out, "last_hidden_state", None)
                except Exception as exc:
                    logger.warning(
                        f"  MPS probe failed for {model_name} ({exc}), falling back to CPU..."
                    )
                    logger.warning(traceback.format_exc())
                    target_device = torch.device("cpu")
                    try:
                        model = AutoModel.from_pretrained(
                            config["id"], local_files_only=local_files_only
                        ).to(target_device).eval()
                    except Exception as cpu_exc:
                        raise_model_load_error(
                            model_id=config["id"],
                            component="vision-language model weights (CPU fallback)",
                            local_files_only=local_files_only,
                            original_error=cpu_exc,
                        )
                logger.info(
                    f"\nProcessing Vision-Language Model {model_name} ({target_device})..."
                )
                return processor, model, target_device

            embeddings: Dict[str, List[float]] = {}
            for concept in all_concepts:
                image_paths = [os.path.join(EXPERIMENT_DIR, p) for p in concept_to_images[concept]]
                per_image_by_layer = None
                if cache_image_embeddings and not force_cache_rebuild:
                    per_image_by_layer = load_cached_per_image_by_layer(
                        cache_model_dir=cache_model_dir,
                        layer_keys=expected_layer_keys,
                        concept=concept,
                        expected_rows=len(image_paths),
                        expected_dim=expected_embedding_dim,
                    )
                    if per_image_by_layer is not None:
                        logger.info(
                            f"     {concept}: loaded cached embeddings ({len(image_paths)} images)"
                        )
                if per_image_by_layer is None:
                    processor, model, target_device = ensure_vlm_loaded()
                    logger.info(f"     {concept}: extracting and caching {len(image_paths)} images")
                    _, per_image = get_siglip_embedding_multi(
                        image_paths, processor, model, target_device
                    )
                    per_image_by_layer = {"layer_last": per_image}
                    if cache_image_embeddings:
                        os.makedirs(cache_model_dir, exist_ok=True)
                        save_cached_per_image_by_layer(cache_model_dir, concept, per_image_by_layer)

                arr = ensure_cache_array(
                    np.asarray(per_image_by_layer["layer_last"], dtype=np.float32),
                    expected_rows=len(image_paths),
                    expected_dim=expected_embedding_dim,
                    cache_path=cache_file_path(cache_model_dir, "layer_last", concept),
                )
                if expected_embedding_dim is None:
                    expected_embedding_dim = int(arr.shape[1])
                embeddings[concept] = arr.mean(axis=0).astype(np.float32).tolist()

            layer_info = {
                "requested_layers": layers_arg,
                "supports_layer_selection": False,
                "reason": "vision-language get_image_features path returns final features only",
                "selected_layer_keys": ["layer_last"],
                "default_layer_key": "layer_last",
            }
            model_result["models"][model_name] = {
                "config": config,
                "embeddings": embeddings,
                "embeddings_by_layer": {
                    "layer_last": embeddings,
                },
                "layer_metadata": layer_info,
            }
            if cache_image_embeddings:
                sample_vec = next(iter(embeddings.values()))
                cache_payload = default_cache_manifest(
                    model_name=model_name,
                    model_config=config,
                    manifest_fingerprint=manifest_fingerprint,
                    concept_to_images=concept_image_map,
                    layer_keys=["layer_last"],
                    embedding_dim=len(sample_vec),
                    layer_profile_id=layer_profile_id,
                    requested_layers_spec=requested_layers_spec,
                    layer_metadata=layer_info,
                )
                atomic_write_json(cache_manifest_path, cache_payload)
        else:
            raise ConfigurationError(f"Model '{model_name}' not found in configurations.")

    except (ConfigurationError, DataIntegrityError, ExtractionError, AnalysisError) as exc:
        logger.error(str(exc))
        logger.error(traceback.format_exc())
        raise SystemExit(1)
    except Exception as exc:  # pragma: no cover - safety net for uncaught runtime failures
        logger.error(f"Unexpected failure for {model_name}: {exc}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)

    with open(output_file, "w") as f:
        json.dump(model_result, f, indent=4)
    logger.info(f"\nResults for {model_name} saved to {output_file}")
    logger.info(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract representations for a single model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to extract (e.g., LFM2-2.6B-Exp-8bit)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing model output JSON.")
    parser.add_argument(
        "--layers",
        type=str,
        default=DEFAULT_LAYER_SPEC,
        help=(
            "Layer selection: '-1' (default), 'all', 'aligned5', "
            "or comma-separated indices like '0,4,8,-1'."
        ),
    )
    parser.add_argument(
        "--cache-image-embeddings",
        type=str,
        default="true",
        help="Persist and reuse per-image embeddings cache (true|false). Default: true.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--force-cache-rebuild",
        action="store_true",
        help="Ignore existing cache shards and rebuild per-image cache for this model.",
    )
    parser.add_argument(
        "--text-template-set",
        type=str,
        default=DEFAULT_TEXT_TEMPLATE_SET,
        help=(
            "Language prompt template set. "
            f"Available: {', '.join(sorted(TEXT_TEMPLATE_SETS.keys()))}"
        ),
    )
    parser.add_argument(
        "--local-files-only",
        type=str,
        default="false",
        help="Load HuggingFace models/processors from local cache only (true|false).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for per-model raw JSON outputs.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for per-model logs.",
    )
    args = parser.parse_args()
    run_replication_for_model(
        args.model,
        force=args.force,
        layers_arg=args.layers,
        cache_image_embeddings=parse_bool_arg(args.cache_image_embeddings),
        cache_dir=args.cache_dir,
        force_cache_rebuild=args.force_cache_rebuild,
        text_template_set=args.text_template_set,
        local_files_only=parse_bool_arg(args.local_files_only),
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
