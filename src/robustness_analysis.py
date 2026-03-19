import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import rankdata, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results", "replication")
DATA_FILE = os.path.join(RESULTS_DIR, "replication_results.json")
MANIFEST_PATH = os.path.join(EXPERIMENT_DIR, "data", "data_manifest_250.json")
DEFAULT_CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "robustness")
CACHE_SCHEMA_VERSION = "1.1.0"
ALIGNED5_FRACTIONS = [0.00, 0.25, 0.50, 0.75, 1.00]

COMPOUND_CONCEPTS_V1 = [
    ("forest", "fire", "forest fire"),
    ("space", "city", "space city"),
    ("water", "city", "water city"),
    ("city", "forest", "city forest"),
]
COMPOUND_CONCEPTS_V2 = [
    ("mountain", "road", "mountain road"),
    ("ocean", "bridge", "ocean bridge"),
    ("city", "bridge", "city bridge"),
    ("mountain", "forest", "mountain forest"),
]
KNOWN_COMPOUNDS = {c[2] for c in COMPOUND_CONCEPTS_V1 + COMPOUND_CONCEPTS_V2}


class ConfigurationError(RuntimeError):
    """Raised when CLI arguments or data layout are invalid."""


class DataIntegrityError(RuntimeError):
    """Raised when cache or compiled artifacts fail integrity checks."""


class AnalysisError(RuntimeError):
    """Raised when statistical analysis preconditions are unmet."""


@dataclass
class PairResult:
    model_a: str
    model_b: str
    rho: float


def _safe_spearman(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    rho, _ = spearmanr(vec_a, vec_b)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def _pair_key(model_a: str, model_b: str) -> Tuple[str, str]:
    return (model_a, model_b) if model_a <= model_b else (model_b, model_a)


def _numeric_layer_sort_key(name: str) -> int:
    if name.startswith("layer_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return 10**9
    if name == "layer_last":
        return 10**9 + 1
    return 10**9 + 2


def _layer_label(fraction: float) -> str:
    return f"d{int(round(fraction * 100)):02d}"


def normalize_layer_spec_text(layer_spec: str) -> str:
    raw = str(layer_spec).strip()
    if not raw:
        return "-1"
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


def resolve_embeddings_for_layer(
    model_name: str,
    model_data: Dict[str, Any],
    requested_layer: str,
) -> Tuple[Dict[str, Any], str]:
    layer_meta = model_data.get("layer_metadata", {})
    layered = model_data.get("embeddings_by_layer")

    if requested_layer in {"selected", "default"}:
        return model_data["embeddings"], layer_meta.get("default_layer_key", "selected")
    if not layered:
        return model_data["embeddings"], layer_meta.get("default_layer_key", "selected")
    if requested_layer in {"last", "-1"}:
        default_key = layer_meta.get("default_layer_key")
        if default_key in layered:
            return layered[default_key], default_key
        keys = sorted(layered.keys(), key=_numeric_layer_sort_key)
        return layered[keys[-1]], keys[-1]
    if requested_layer in layered:
        return layered[requested_layer], requested_layer
    return model_data["embeddings"], layer_meta.get("default_layer_key", "selected")


def stack_embeddings(embeddings: Dict[str, Any], concepts: List[str]) -> np.ndarray:
    rows = [
        np.asarray(embeddings[concept], dtype=np.float32).reshape(-1)
        for concept in concepts
    ]
    return np.stack(rows, axis=0)


def stack_per_concept_arrays(
    per_concept: Dict[str, np.ndarray],
    concepts: List[str],
) -> Optional[np.ndarray]:
    arrays = [np.asarray(per_concept[concept], dtype=np.float32) for concept in concepts]
    row_counts = {arr.shape[0] for arr in arrays}
    if len(row_counts) != 1:
        return None
    return np.stack(arrays, axis=0)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


def build_similarity_matrix_from_stacked(stacked_embeddings: np.ndarray) -> np.ndarray:
    normalized = _row_normalize(stacked_embeddings)
    matrix = normalized @ normalized.T
    np.clip(matrix, -1.0, 1.0, out=matrix)
    return matrix.astype(np.float32, copy=False)


def build_similarity_matrix(embeddings: Dict[str, Any], concepts: List[str]) -> np.ndarray:
    return build_similarity_matrix_from_stacked(stack_embeddings(embeddings, concepts))


def build_similarity_flat_from_stacked(
    stacked_embeddings: np.ndarray,
    triu_idx: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    matrix = build_similarity_matrix_from_stacked(stacked_embeddings)
    return matrix[triu_idx]


def upper_triangle_flat(
    matrix: np.ndarray,
    triu_idx: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    idx = triu_idx or np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def build_model_pairs(model_names: List[str]) -> List[Tuple[str, str]]:
    return [
        (model_a, model_b)
        for i, model_a in enumerate(model_names)
        for model_b in model_names[i + 1:]
    ]


def spearman_correlation_matrix(rows: np.ndarray) -> np.ndarray:
    if rows.ndim != 2:
        raise ValueError(f"Expected rank-2 array for row correlations, got {rows.shape}")
    ranked = np.empty(rows.shape, dtype=np.float32)
    for idx, row in enumerate(rows):
        ranked[idx] = rankdata(row, method="average").astype(np.float32)
    ranked -= ranked.mean(axis=1, keepdims=True)
    ranked = _row_normalize(ranked)
    corr = ranked @ ranked.T
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr.astype(np.float32, copy=False)


def default_mantel_workers() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    return min(8, cpu_count - 1)


def load_compiled_data(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Compiled results not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_category(model_info: Dict[str, Any]) -> str:
    mtype = model_info.get("config", {}).get("type")
    if mtype == "causal":
        return "language"
    if mtype in {"vision", "vision_language"}:
        return "image"
    return "other"


def validate_cache_manifest(
    cache_manifest: Dict[str, Any],
    model_name: str,
    expected_layer_profile_id: Optional[str],
    expected_requested_layers_spec: Optional[str],
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
    missing = [field for field in required_fields if field not in cache_manifest]
    if missing:
        raise DataIntegrityError(
            f"Cache manifest missing fields for {model_name}: {missing}"
        )
    if cache_manifest["schema_version"] != CACHE_SCHEMA_VERSION:
        raise DataIntegrityError(
            f"Cache schema mismatch for {model_name}: "
            f"{cache_manifest['schema_version']} != {CACHE_SCHEMA_VERSION}"
        )
    if cache_manifest["model_name"] != model_name:
        raise DataIntegrityError(
            f"Cache manifest model mismatch for {model_name}: {cache_manifest['model_name']}"
        )
    if cache_manifest.get("dtype") != "float32":
        raise DataIntegrityError(
            f"Cache dtype must be float32 for {model_name}, got {cache_manifest.get('dtype')}"
        )
    if not isinstance(cache_manifest.get("embedding_dim"), int):
        raise DataIntegrityError(f"Cache embedding_dim invalid for {model_name}")
    if not isinstance(cache_manifest.get("layer_keys"), list):
        raise DataIntegrityError(f"Cache layer_keys invalid for {model_name}")
    if (
        not isinstance(cache_manifest.get("layer_profile_id"), str)
        or not cache_manifest["layer_profile_id"]
    ):
        raise DataIntegrityError(f"Cache layer_profile_id invalid for {model_name}")
    if (
        not isinstance(cache_manifest.get("requested_layers_spec"), str)
        or not cache_manifest["requested_layers_spec"].strip()
    ):
        raise DataIntegrityError(f"Cache requested_layers_spec invalid for {model_name}")
    if expected_layer_profile_id and cache_manifest.get("layer_profile_id") != expected_layer_profile_id:
        raise DataIntegrityError(
            f"Cache profile mismatch for {model_name}: "
            f"{cache_manifest.get('layer_profile_id')} != {expected_layer_profile_id}"
        )
    if expected_requested_layers_spec:
        got_spec = normalize_layer_spec_text(cache_manifest.get("requested_layers_spec", ""))
        want_spec = normalize_layer_spec_text(expected_requested_layers_spec)
        if got_spec != want_spec:
            raise DataIntegrityError(
                f"Cache requested layer spec mismatch for {model_name}: {got_spec} != {want_spec}"
            )


def load_cache_manifest(
    cache_dir: str,
    model_name: str,
    expected_layer_profile_id: Optional[str],
    expected_requested_layers_spec: Optional[str],
) -> Dict[str, Any]:
    path = os.path.join(cache_dir, model_name, "cache_manifest.json")
    if not os.path.exists(path):
        raise DataIntegrityError(
            f"Missing cache manifest for {model_name}: {path}. "
            "Remediation: rerun extraction with --force-cache-rebuild."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    validate_cache_manifest(
        payload,
        model_name,
        expected_layer_profile_id,
        expected_requested_layers_spec,
    )
    return payload


def load_image_cache_for_model(
    cache_dir: str,
    model_name: str,
    layer_key: str,
    concepts: List[str],
    bootstrap_sample_size: int,
    bootstrap_replacement: bool,
    expected_layer_profile_id: Optional[str],
    expected_requested_layers_spec: Optional[str],
) -> Dict[str, np.ndarray]:
    manifest = load_cache_manifest(
        cache_dir,
        model_name,
        expected_layer_profile_id,
        expected_requested_layers_spec,
    )
    layer_keys = manifest.get("layer_keys", [])
    if layer_key not in layer_keys:
        raise DataIntegrityError(
            f"Cache for {model_name} missing layer '{layer_key}'. "
            f"Available: {layer_keys}"
        )
    expected_dim = int(manifest.get("embedding_dim"))
    per_concept: Dict[str, np.ndarray] = {}
    for concept in concepts:
        shard = os.path.join(cache_dir, model_name, layer_key, f"{concept}.npy")
        if not os.path.exists(shard):
            raise DataIntegrityError(
                f"Missing cache shard for {model_name} concept '{concept}' at {shard}. "
                f"Remediation: python {os.path.join(SCRIPT_DIR, 'main_replication.py')} "
                f"--model \"{model_name}\" --force --force-cache-rebuild"
            )
        arr = np.asarray(np.load(shard), dtype=np.float32)
        if arr.ndim != 2:
            raise DataIntegrityError(
                f"Cache shard {shard} must be rank-2. Got shape={arr.shape}"
            )
        if arr.shape[1] != expected_dim:
            raise DataIntegrityError(
                f"Cache shard dim mismatch for {model_name}/{concept}: "
                f"{arr.shape[1]} != {expected_dim}"
            )
        if (not bootstrap_replacement) and arr.shape[0] < bootstrap_sample_size:
            raise DataIntegrityError(
                f"Not enough images for {model_name}/{concept}: {arr.shape[0]} rows < "
                f"bootstrap sample size {bootstrap_sample_size} without replacement."
            )
        per_concept[concept] = arr
    return per_concept


def mantel_permutation_two_sided(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    triu = np.triu_indices(matrix_a.shape[0], k=1)
    observed = _safe_spearman(matrix_a[triu], matrix_b[triu])
    hits = 0
    for _ in range(permutations):
        perm = rng.permutation(matrix_b.shape[0])
        perm_b = matrix_b[np.ix_(perm, perm)]
        rho = _safe_spearman(matrix_a[triu], perm_b[triu])
        if abs(rho) >= abs(observed):
            hits += 1
    return float((hits + 1) / (permutations + 1))


def _mantel_pair_task(
    task: Tuple[int, str, str, np.ndarray, np.ndarray, int, int]
) -> Tuple[int, str, str, float]:
    idx, model_a, model_b, matrix_a, matrix_b, permutations, seed = task
    pair_rng = np.random.default_rng(seed + idx + 1)
    p = mantel_permutation_two_sided(
        matrix_a,
        matrix_b,
        permutations=permutations,
        rng=pair_rng,
    )
    return idx, model_a, model_b, p


def bh_fdr_correction(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.asarray([p_values[i] for i in order], dtype=np.float64)
    q_sorted = np.zeros(m, dtype=np.float64)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = float(sorted_p[i] * m / rank)
        running_min = min(running_min, q)
        q_sorted[i] = running_min
    q = np.zeros(m, dtype=np.float64)
    for idx, original_idx in enumerate(order):
        q[original_idx] = min(1.0, q_sorted[idx])
    return q.tolist()


def build_pairwise_rhos(
    flats: Dict[str, np.ndarray],
    model_names: List[str],
) -> Dict[Tuple[str, str], float]:
    results: Dict[Tuple[str, str], float] = {}
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1:]:
            results[_pair_key(model_a, model_b)] = _safe_spearman(
                flats[model_a], flats[model_b]
            )
    return results


def aligned_layer_key_for_fraction(layer_keys: List[str], fraction: float) -> str:
    numeric_keys = [k for k in layer_keys if k.startswith("layer_")]
    if not numeric_keys:
        if "layer_last" in layer_keys:
            return "layer_last"
        return sorted(layer_keys, key=_numeric_layer_sort_key)[-1]
    numeric_keys = sorted(numeric_keys, key=_numeric_layer_sort_key)
    if len(numeric_keys) == 1:
        return numeric_keys[0]
    idx = int(round(fraction * (len(numeric_keys) - 1)))
    idx = max(0, min(len(numeric_keys) - 1, idx))
    return numeric_keys[idx]


def validate_robustness_payload(payload: Dict[str, Any]) -> None:
    required = [
        "metadata",
        "rsa_bootstrap_image",
        "rsa_significance",
        "source_holdout",
        "prompt_sensitivity",
        "aligned_layer",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise DataIntegrityError(f"Robustness output missing required top-level keys: {missing}")
    metadata_required = [
        "timestamp",
        "seed",
        "bootstrap_draws",
        "bootstrap_sample_size",
        "bootstrap_replacement",
        "mantel_permutations",
        "layer",
    ]
    metadata_missing = [k for k in metadata_required if k not in payload["metadata"]]
    if metadata_missing:
        raise DataIntegrityError(
            f"Robustness output metadata missing required keys: {metadata_missing}"
        )


def save_rsa_heatmap(
    model_names: List[str],
    pairwise_results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    matrix = np.eye(len(model_names), dtype=np.float32)
    model_idx = {name: i for i, name in enumerate(model_names)}
    for item in pairwise_results:
        i = model_idx[item["model_a"]]
        j = model_idx[item["model_b"]]
        matrix[i, j] = item["rho_mean"]
        matrix[j, i] = item["rho_mean"]
    plt.figure(figsize=(max(10, len(model_names) * 0.7), max(8, len(model_names) * 0.6)))
    sns.heatmap(
        matrix,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        xticklabels=model_names,
        yticklabels=model_names,
        cbar_kws={"label": "RSA rho mean"},
    )
    plt.title("Image-Bootstrap RSA Mean")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_source_holdout_plot(source_holdout: Dict[str, Any], output_path: str) -> None:
    loso = source_holdout.get("leave_one_source_out", {})
    labels = []
    values = []
    for source, payload in loso.items():
        if payload.get("skipped"):
            continue
        deltas = [abs(item["delta"]) for item in payload.get("pairwise", [])]
        labels.append(source)
        values.append(float(np.mean(deltas)) if deltas else 0.0)
    if not labels:
        return
    plt.figure(figsize=(max(8, len(labels) * 1.5), 4))
    plt.bar(labels, values, color="#2E86DE", alpha=0.9)
    plt.ylabel("Mean |delta rho| vs full")
    plt.title("Leave-One-Source-Out Sensitivity")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def main(
    data_file: str,
    manifest_path: str,
    layer: str,
    bootstrap_draws: int,
    bootstrap_sample_size: int,
    bootstrap_replacement: bool,
    mantel_permutations: int,
    min_concepts_for_rsa: int,
    seed: int,
    cache_dir: str,
    output_dir: str,
    expected_layer_profile_id: Optional[str],
    requested_layers_spec: Optional[str],
    mantel_workers: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    run_start = perf_counter()

    compiled = load_compiled_data(data_file)
    manifest = load_manifest(manifest_path)
    concept_to_images = manifest.get("concept_to_images", {})
    concept_metadata = manifest.get("concept_metadata", {})
    if not concept_to_images:
        raise DataIntegrityError("Manifest missing concept_to_images.")
    if not concept_metadata:
        raise DataIntegrityError("Manifest missing concept_metadata.")

    all_concepts = compiled.get("concepts", [])
    raw_models = compiled.get("models", {})
    if not all_concepts or not raw_models:
        raise AnalysisError("Compiled results missing concepts or models.")

    base_concepts = [c for c in all_concepts if c in concept_to_images and c not in KNOWN_COMPOUNDS]
    if len(base_concepts) < min_concepts_for_rsa:
        raise AnalysisError(
            f"Not enough base concepts for RSA: {len(base_concepts)} < {min_concepts_for_rsa}"
        )

    model_names = sorted(raw_models.keys())
    model_pairs = build_model_pairs(model_names)
    model_pair_triu = np.triu_indices(len(model_names), k=1)
    resolved_models: Dict[str, Dict[str, Any]] = {}
    resolved_layers: Dict[str, str] = {}
    categories: Dict[str, str] = {}
    concept_triu = np.triu_indices(len(base_concepts), k=1)
    for model_name in model_names:
        model_data = raw_models[model_name]
        embeddings, resolved_layer = resolve_embeddings_for_layer(model_name, model_data, layer)
        stacked_embeddings = stack_embeddings(embeddings, base_concepts)
        resolved_layers[model_name] = resolved_layer
        categories[model_name] = model_category(model_data)
        resolved_models[model_name] = {
            "raw": model_data,
            "embeddings": embeddings,
            "stacked_embeddings": stacked_embeddings,
        }

    image_models = [m for m in model_names if categories[m] == "image"]
    print(
        "Loaded robustness inputs: "
        f"models={len(model_names)}, image_models={len(image_models)}, "
        f"concepts={len(base_concepts)}, pairs={len(model_pairs)}"
    )

    baseline_matrices: Dict[str, np.ndarray] = {}
    baseline_flats: Dict[str, np.ndarray] = {}
    baseline_start = perf_counter()
    for model_name in model_names:
        stacked_embeddings = resolved_models[model_name]["stacked_embeddings"]
        sim = build_similarity_matrix_from_stacked(stacked_embeddings)
        baseline_matrices[model_name] = sim
        baseline_flats[model_name] = upper_triangle_flat(sim, concept_triu)
    print(f"Built baseline similarity matrices in {perf_counter() - baseline_start:.1f}s")

    if not image_models:
        raise AnalysisError("Robustness analysis requires at least one image model.")
    cache_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    cache_tensors: Dict[str, np.ndarray] = {}
    bootstrap_rows: Optional[int] = None
    bootstrap_vectorized = True
    cache_start = perf_counter()
    for model_name in image_models:
        layer_key = resolved_layers[model_name]
        cache_arrays[model_name] = load_image_cache_for_model(
            cache_dir=cache_dir,
            model_name=model_name,
            layer_key=layer_key,
            concepts=base_concepts,
            bootstrap_sample_size=bootstrap_sample_size,
            bootstrap_replacement=bootstrap_replacement,
            expected_layer_profile_id=expected_layer_profile_id,
            expected_requested_layers_spec=requested_layers_spec,
        )
        cache_tensor = stack_per_concept_arrays(cache_arrays[model_name], base_concepts)
        if cache_tensor is None:
            bootstrap_vectorized = False
            continue
        cache_tensors[model_name] = cache_tensor
        row_count = cache_tensor.shape[1]
        if bootstrap_rows is None:
            bootstrap_rows = row_count
        elif bootstrap_rows != row_count:
            bootstrap_vectorized = False
    print(
        f"Loaded image cache tensors in {perf_counter() - cache_start:.1f}s "
        f"(vectorized_bootstrap={bootstrap_vectorized})"
    )

    rng = np.random.default_rng(seed)
    pair_draws = np.empty((len(model_pairs), bootstrap_draws), dtype=np.float32)
    bootstrap_start = perf_counter()
    if bootstrap_vectorized and bootstrap_rows is not None and len(cache_tensors) == len(image_models):
        concept_index = np.arange(len(base_concepts))[:, None]
        for draw_idx in range(bootstrap_draws):
            if bootstrap_replacement:
                sampled_indices = rng.integers(
                    0,
                    bootstrap_rows,
                    size=(len(base_concepts), bootstrap_sample_size),
                )
            else:
                sampled_indices = np.empty(
                    (len(base_concepts), bootstrap_sample_size),
                    dtype=np.int64,
                )
                for concept_idx in range(len(base_concepts)):
                    sampled_indices[concept_idx] = rng.choice(
                        bootstrap_rows,
                        size=bootstrap_sample_size,
                        replace=False,
                    )

            draw_rows = []
            for model_name in model_names:
                if model_name not in image_models:
                    draw_rows.append(baseline_flats[model_name])
                    continue
                sampled = cache_tensors[model_name][concept_index, sampled_indices]
                centroids = sampled.mean(axis=1)
                draw_rows.append(build_similarity_flat_from_stacked(centroids, concept_triu))

            draw_corr = spearman_correlation_matrix(np.stack(draw_rows, axis=0))
            pair_draws[:, draw_idx] = draw_corr[model_pair_triu]
    else:
        for draw_idx in range(bootstrap_draws):
            sampled_indices = {}
            for concept in base_concepts:
                rows = (
                    cache_arrays[image_models[0]][concept].shape[0]
                    if image_models
                    else bootstrap_sample_size
                )
                if bootstrap_replacement:
                    idx = rng.integers(0, rows, size=bootstrap_sample_size)
                else:
                    idx = rng.choice(rows, size=bootstrap_sample_size, replace=False)
                sampled_indices[concept] = idx

            draw_rows = []
            for model_name in model_names:
                if model_name not in image_models:
                    draw_rows.append(baseline_flats[model_name])
                    continue
                centroids = {}
                for concept in base_concepts:
                    arr = cache_arrays[model_name][concept]
                    centroids[concept] = arr[sampled_indices[concept]].mean(axis=0)
                draw_rows.append(build_similarity_flat_from_stacked(
                    stack_embeddings(centroids, base_concepts),
                    concept_triu,
                ))

            draw_corr = spearman_correlation_matrix(np.stack(draw_rows, axis=0))
            pair_draws[:, draw_idx] = draw_corr[model_pair_triu]
    print(f"Completed bootstrap stage in {perf_counter() - bootstrap_start:.1f}s")

    baseline_corr = spearman_correlation_matrix(
        np.stack([baseline_flats[model_name] for model_name in model_names], axis=0)
    )
    baseline_pair_values = baseline_corr[model_pair_triu]
    baseline_pairwise = {
        _pair_key(model_a, model_b): float(rho)
        for (model_a, model_b), rho in zip(model_pairs, baseline_pair_values)
    }
    rsa_bootstrap_results = []
    for pair_idx, (model_a, model_b) in enumerate(model_pairs):
        key = _pair_key(model_a, model_b)
        baseline_rho = baseline_pairwise[key]
        deterministic = categories[model_a] != "image" and categories[model_b] != "image"
        if deterministic:
            rho_mean = baseline_rho
            ci_low = baseline_rho
            ci_high = baseline_rho
        else:
            draws = pair_draws[pair_idx]
            rho_mean = float(np.mean(draws))
            ci_low, ci_high = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
        rsa_bootstrap_results.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "rho_point_estimate": baseline_rho,
                "rho_mean": rho_mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "deterministic_non_image_pair": deterministic,
            }
        )

    p_values = []
    significance_rows = []
    mantel_start = perf_counter()
    worker_count = mantel_workers if mantel_workers > 0 else default_mantel_workers()
    worker_count = max(1, min(worker_count, len(model_pairs)))
    print(
        "Running Mantel significance stage: "
        f"pairs={len(model_pairs)}, permutations={mantel_permutations}, workers={worker_count}"
    )
    if worker_count == 1:
        mantel_results = []
        for idx, (model_a, model_b) in enumerate(model_pairs):
            p = _mantel_pair_task(
                (
                    idx,
                    model_a,
                    model_b,
                    baseline_matrices[model_a],
                    baseline_matrices[model_b],
                    mantel_permutations,
                    seed,
                )
            )[3]
            mantel_results.append((idx, model_a, model_b, p))
    else:
        tasks = [
            (
                idx,
                model_a,
                model_b,
                baseline_matrices[model_a],
                baseline_matrices[model_b],
                mantel_permutations,
                seed,
            )
            for idx, (model_a, model_b) in enumerate(model_pairs)
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            mantel_results = list(executor.map(_mantel_pair_task, tasks, chunksize=1))
        mantel_results.sort(key=lambda item: item[0])
    for idx, model_a, model_b, p in mantel_results:
        p_values.append(p)
        significance_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "rho": baseline_pairwise[_pair_key(model_a, model_b)],
                "p_mantel_two_sided": p,
            }
        )
    print(f"Completed Mantel significance stage in {perf_counter() - mantel_start:.1f}s")
    q_values = bh_fdr_correction(p_values)
    for row, q in zip(significance_rows, q_values):
        row["q_bh_fdr"] = q

    source_by_concept = {}
    for concept in base_concepts:
        md = concept_metadata.get(concept, {})
        source = md.get("source")
        if not source:
            raise DataIntegrityError(
                f"Concept metadata missing source for '{concept}'."
            )
        source_by_concept[concept] = source
    unique_sources = sorted(set(source_by_concept.values()))
    concept_index_map = {concept: idx for idx, concept in enumerate(base_concepts)}

    def pairwise_for_concepts(subset: List[str]) -> Dict[Tuple[str, str], float]:
        subset_indices = np.asarray([concept_index_map[concept] for concept in subset], dtype=np.int64)
        subset_triu = np.triu_indices(len(subset), k=1)
        flat_rows = []
        for model_name in model_names:
            stacked_embeddings = resolved_models[model_name]["stacked_embeddings"][subset_indices]
            flat_rows.append(build_similarity_flat_from_stacked(stacked_embeddings, subset_triu))
        corr = spearman_correlation_matrix(np.stack(flat_rows, axis=0))
        return {
            _pair_key(model_a, model_b): float(rho)
            for (model_a, model_b), rho in zip(model_pairs, corr[model_pair_triu])
        }

    source_holdout = {
        "min_concepts_for_rsa": min_concepts_for_rsa,
        "sources": unique_sources,
        "leave_one_source_out": {},
        "source_only": {},
    }
    full_pairwise = baseline_pairwise
    for source in unique_sources:
        loso_subset = [c for c in base_concepts if source_by_concept[c] != source]
        if len(loso_subset) < min_concepts_for_rsa:
            source_holdout["leave_one_source_out"][source] = {
                "skipped": True,
                "reason": (
                    f"insufficient concepts ({len(loso_subset)} < {min_concepts_for_rsa})"
                ),
                "concept_count": len(loso_subset),
            }
        else:
            loso_pairwise = pairwise_for_concepts(loso_subset)
            rows = []
            for i, model_a in enumerate(model_names):
                for model_b in model_names[i + 1:]:
                    key = _pair_key(model_a, model_b)
                    subset_rho = loso_pairwise[key]
                    full_rho = full_pairwise[key]
                    rows.append(
                        {
                            "model_a": model_a,
                            "model_b": model_b,
                            "rho_subset": subset_rho,
                            "rho_full": full_rho,
                            "delta": float(subset_rho - full_rho),
                        }
                    )
            source_holdout["leave_one_source_out"][source] = {
                "skipped": False,
                "concept_count": len(loso_subset),
                "pairwise": rows,
            }

        only_subset = [c for c in base_concepts if source_by_concept[c] == source]
        if len(only_subset) < min_concepts_for_rsa:
            source_holdout["source_only"][source] = {
                "skipped": True,
                "reason": (
                    f"insufficient concepts ({len(only_subset)} < {min_concepts_for_rsa})"
                ),
                "concept_count": len(only_subset),
            }
        else:
            only_pairwise = pairwise_for_concepts(only_subset)
            rows = []
            for i, model_a in enumerate(model_names):
                for model_b in model_names[i + 1:]:
                    key = _pair_key(model_a, model_b)
                    subset_rho = only_pairwise[key]
                    full_rho = full_pairwise[key]
                    rows.append(
                        {
                            "model_a": model_a,
                            "model_b": model_b,
                            "rho_subset": subset_rho,
                            "rho_full": full_rho,
                            "delta": float(subset_rho - full_rho),
                        }
                    )
            source_holdout["source_only"][source] = {
                "skipped": False,
                "concept_count": len(only_subset),
                "pairwise": rows,
            }

    prompt_sensitivity = {"models": {}, "skipped": []}
    for model_name in model_names:
        if categories[model_name] != "language":
            continue
        model_data = raw_models[model_name]
        template_meta = model_data.get("text_template_metadata")
        template_layer_embs = model_data.get("text_template_embeddings_by_layer")
        if not template_meta or not template_layer_embs:
            prompt_sensitivity["skipped"].append(
                {
                    "model": model_name,
                    "reason": "missing text_template_metadata or text_template_embeddings_by_layer",
                }
            )
            continue
        templates = template_meta.get("templates", {})
        baseline_key = template_meta.get("baseline_template_key", "t0")
        resolved_layer = resolved_layers[model_name]
        layer_default = model_data.get("layer_metadata", {}).get("default_layer_key")

        template_flats: Dict[str, np.ndarray] = {}
        missing_templates = []
        for template_key in sorted(templates.keys()):
            by_layer = template_layer_embs.get(template_key, {})
            emb = by_layer.get(resolved_layer)
            if emb is None and layer_default in by_layer:
                emb = by_layer[layer_default]
            if emb is None:
                missing_templates.append(template_key)
                continue
            template_flats[template_key] = build_similarity_flat_from_stacked(
                stack_embeddings(emb, base_concepts),
                concept_triu,
            )

        if len(template_flats) < 2:
            prompt_sensitivity["skipped"].append(
                {
                    "model": model_name,
                    "reason": f"insufficient template embeddings for layer {resolved_layer}",
                    "missing_templates": missing_templates,
                }
            )
            continue

        template_pairwise = []
        template_keys_sorted = sorted(template_flats.keys())
        template_corr = spearman_correlation_matrix(
            np.stack([template_flats[key] for key in template_keys_sorted], axis=0)
        )
        template_triu = np.triu_indices(len(template_keys_sorted), k=1)
        for (idx_a, idx_b), rho in zip(zip(*template_triu), template_corr[template_triu]):
            template_pairwise.append(
                {
                    "template_a": template_keys_sorted[idx_a],
                    "template_b": template_keys_sorted[idx_b],
                    "rho": float(rho),
                }
            )

        cross_modal = {}
        for template_key in template_keys_sorted:
            per_image_model = {}
            for image_model in image_models:
                per_image_model[image_model] = _safe_spearman(
                    template_flats[template_key], baseline_flats[image_model]
                )
            vals = np.asarray(list(per_image_model.values()), dtype=np.float32)
            cross_modal[template_key] = {
                "per_model": per_image_model,
                "mean": float(vals.mean()) if vals.size else 0.0,
                "min": float(vals.min()) if vals.size else 0.0,
                "max": float(vals.max()) if vals.size else 0.0,
            }

        baseline_vals = cross_modal.get(baseline_key)
        max_abs_delta = 0.0
        if baseline_vals:
            for template_key, stats in cross_modal.items():
                if template_key == baseline_key:
                    continue
                for image_model, rho in stats["per_model"].items():
                    baseline_rho = baseline_vals["per_model"][image_model]
                    max_abs_delta = max(max_abs_delta, abs(rho - baseline_rho))

        prompt_sensitivity["models"][model_name] = {
            "resolved_layer": resolved_layer,
            "baseline_template_key": baseline_key,
            "template_keys": template_keys_sorted,
            "missing_templates": missing_templates,
            "template_pairwise_rho": template_pairwise,
            "cross_modal_rho_by_template": cross_modal,
            "max_abs_delta_vs_baseline": float(max_abs_delta),
        }

    aligned_layer = {
        "fractions": ALIGNED5_FRACTIONS,
        "model_layer_selection": {},
        "pairwise_by_fraction": {},
    }
    model_layer_candidates: Dict[str, List[str]] = {}
    for model_name in model_names:
        layered = raw_models[model_name].get("embeddings_by_layer") or {}
        if layered:
            model_layer_candidates[model_name] = sorted(layered.keys(), key=_numeric_layer_sort_key)
        else:
            model_layer_candidates[model_name] = ["selected"]
        aligned_layer["model_layer_selection"][model_name] = {}

    for fraction in ALIGNED5_FRACTIONS:
        label = _layer_label(fraction)
        flat_rows = []
        for model_name in model_names:
            candidates = model_layer_candidates[model_name]
            if "selected" in candidates:
                chosen = "selected"
                embs = resolved_models[model_name]["stacked_embeddings"]
            else:
                chosen = aligned_layer_key_for_fraction(candidates, fraction)
                embs = stack_embeddings(
                    raw_models[model_name]["embeddings_by_layer"][chosen],
                    base_concepts,
                )
            aligned_layer["model_layer_selection"][model_name][label] = chosen
            flat_rows.append(build_similarity_flat_from_stacked(embs, concept_triu))
        corr = spearman_correlation_matrix(np.stack(flat_rows, axis=0))
        pairwise = []
        for (model_a, model_b), rho in zip(model_pairs, corr[model_pair_triu]):
            pairwise.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "rho": float(rho),
                }
            )
        aligned_layer["pairwise_by_fraction"][label] = pairwise

    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_sample_size": bootstrap_sample_size,
            "bootstrap_replacement": bootstrap_replacement,
            "mantel_permutations": mantel_permutations,
            "mantel_workers": worker_count,
            "layer": layer,
            "base_concepts_count": len(base_concepts),
            "model_count": len(model_names),
            "image_model_count": len(image_models),
            "ci_method": "image_bootstrap_percentile",
            "p_value_method": "mantel_permutation_two_sided",
            "multiple_comparison_correction": "benjamini_hochberg_fdr",
            "cache_dir": os.path.abspath(cache_dir),
            "data_file": os.path.abspath(data_file),
            "manifest_path": os.path.abspath(manifest_path),
            "expected_layer_profile_id": expected_layer_profile_id,
            "requested_layers_spec": requested_layers_spec,
            "output_dir": os.path.abspath(output_dir),
        },
        "rsa_bootstrap_image": {
            "pairwise_results": rsa_bootstrap_results,
            "deterministic_pair_rule": "language-language pairs have zero-width CI",
        },
        "rsa_significance": {
            "minimum_permutation_p_value": float(1.0 / (mantel_permutations + 1)),
            "pairwise_results": significance_rows,
        },
        "source_holdout": source_holdout,
        "prompt_sensitivity": prompt_sensitivity,
        "aligned_layer": aligned_layer,
    }

    validate_robustness_payload(payload)
    output_json = os.path.join(output_dir, "robustness_stats.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved robustness stats -> {output_json}")

    save_rsa_heatmap(
        model_names=model_names,
        pairwise_results=rsa_bootstrap_results,
        output_path=os.path.join(output_dir, "rsa_bootstrap_matrix_mean.png"),
    )
    save_source_holdout_plot(
        source_holdout=source_holdout,
        output_path=os.path.join(output_dir, "source_holdout_delta_summary.png"),
    )
    print("Saved robustness plots.")
    print(f"Robustness analysis completed in {perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness analysis over compiled replication outputs.")
    parser.add_argument(
        "--data-file",
        type=str,
        default=DATA_FILE,
        help=f"Compiled replication JSON (default: {DATA_FILE}).",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=MANIFEST_PATH,
        help=f"Concept manifest JSON (default: {MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="selected",
        help="Layer key to analyze: selected|default|last|-1|layer_N.",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=300,
        help="Number of bootstrap draws over image subsets.",
    )
    parser.add_argument(
        "--bootstrap-sample-size",
        type=int,
        default=10,
        help="Images sampled per concept per draw.",
    )
    parser.add_argument(
        "--bootstrap-replacement",
        type=str,
        default="true",
        help="Sample images with replacement (true|false). Default: true.",
    )
    parser.add_argument(
        "--mantel-permutations",
        type=int,
        default=3000,
        help="Number of Mantel permutations per model pair.",
    )
    parser.add_argument(
        "--mantel-workers",
        type=int,
        default=0,
        help="Worker processes for Mantel tests. Use 0 for auto.",
    )
    parser.add_argument(
        "--min-concepts-for-rsa",
        type=int,
        default=8,
        help="Minimum concept count required for source split RSA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--expected-layer-profile-id",
        type=str,
        default=None,
        help="Optional expected cache profile id (e.g., baseline_last, aligned5).",
    )
    parser.add_argument(
        "--requested-layers-spec",
        type=str,
        default=None,
        help="Optional expected requested layer spec used to build cache (e.g., -1, aligned5).",
    )
    args = parser.parse_args()
    replacement = str(args.bootstrap_replacement).strip().lower()
    if replacement in {"true", "1", "yes", "y", "on"}:
        bootstrap_replacement = True
    elif replacement in {"false", "0", "no", "n", "off"}:
        bootstrap_replacement = False
    else:
        raise ConfigurationError(
            f"Invalid --bootstrap-replacement '{args.bootstrap_replacement}'. Use true|false."
        )
    main(
        data_file=args.data_file,
        manifest_path=args.manifest_path,
        layer=args.layer,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_sample_size=args.bootstrap_sample_size,
        bootstrap_replacement=bootstrap_replacement,
        mantel_permutations=args.mantel_permutations,
        min_concepts_for_rsa=args.min_concepts_for_rsa,
        seed=args.seed,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        expected_layer_profile_id=args.expected_layer_profile_id,
        requested_layers_spec=args.requested_layers_spec,
        mantel_workers=args.mantel_workers,
    )
