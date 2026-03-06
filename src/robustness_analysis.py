import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results", "replication")
DATA_FILE = os.path.join(RESULTS_DIR, "replication_results.json")
MANIFEST_PATH = os.path.join(EXPERIMENT_DIR, "data", "data_manifest_multi.json")
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


def build_similarity_matrix(embeddings: Dict[str, Any], concepts: List[str]) -> np.ndarray:
    n = len(concepts)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i, c1 in enumerate(concepts):
        e1 = np.asarray(embeddings[c1], dtype=np.float32).reshape(1, -1)
        for j, c2 in enumerate(concepts):
            e2 = np.asarray(embeddings[c2], dtype=np.float32).reshape(1, -1)
            matrix[i, j] = cosine_similarity(e1, e2)[0][0]
    return matrix


def upper_triangle_flat(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


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
) -> None:
    os.makedirs(output_dir, exist_ok=True)

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
    resolved_models: Dict[str, Dict[str, Any]] = {}
    resolved_layers: Dict[str, str] = {}
    categories: Dict[str, str] = {}
    for model_name in model_names:
        model_data = raw_models[model_name]
        embeddings, resolved_layer = resolve_embeddings_for_layer(model_name, model_data, layer)
        resolved_layers[model_name] = resolved_layer
        categories[model_name] = model_category(model_data)
        resolved_models[model_name] = {
            "raw": model_data,
            "embeddings": embeddings,
        }

    baseline_matrices: Dict[str, np.ndarray] = {}
    baseline_flats: Dict[str, np.ndarray] = {}
    for model_name in model_names:
        sim = build_similarity_matrix(resolved_models[model_name]["embeddings"], base_concepts)
        baseline_matrices[model_name] = sim
        baseline_flats[model_name] = upper_triangle_flat(sim)

    image_models = [m for m in model_names if categories[m] == "image"]
    if not image_models:
        raise AnalysisError("Robustness analysis requires at least one image model.")
    cache_arrays: Dict[str, Dict[str, np.ndarray]] = {}
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

    rng = np.random.default_rng(seed)
    pair_draws: Dict[Tuple[str, str], List[float]] = {
        _pair_key(a, b): []
        for i, a in enumerate(model_names) for b in model_names[i + 1:]
    }

    for _ in range(bootstrap_draws):
        sampled_indices = {}
        for concept in base_concepts:
            rows = cache_arrays[image_models[0]][concept].shape[0] if image_models else bootstrap_sample_size
            if bootstrap_replacement:
                idx = rng.integers(0, rows, size=bootstrap_sample_size)
            else:
                idx = rng.choice(rows, size=bootstrap_sample_size, replace=False)
            sampled_indices[concept] = idx

        draw_flats: Dict[str, np.ndarray] = {}
        for model_name in model_names:
            if model_name not in image_models:
                draw_flats[model_name] = baseline_flats[model_name]
                continue
            centroids = {}
            for concept in base_concepts:
                arr = cache_arrays[model_name][concept]
                centroids[concept] = arr[sampled_indices[concept]].mean(axis=0)
            draw_matrix = build_similarity_matrix(centroids, base_concepts)
            draw_flats[model_name] = upper_triangle_flat(draw_matrix)

        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1:]:
                key = _pair_key(model_a, model_b)
                rho = _safe_spearman(draw_flats[model_a], draw_flats[model_b])
                pair_draws[key].append(rho)

    baseline_pairwise = build_pairwise_rhos(baseline_flats, model_names)
    rsa_bootstrap_results = []
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1:]:
            key = _pair_key(model_a, model_b)
            baseline_rho = baseline_pairwise[key]
            deterministic = categories[model_a] != "image" and categories[model_b] != "image"
            if deterministic:
                rho_mean = baseline_rho
                ci_low = baseline_rho
                ci_high = baseline_rho
            else:
                draws = np.asarray(pair_draws[key], dtype=np.float32)
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
    for idx, (model_a, model_b) in enumerate(
        [(a, b) for i, a in enumerate(model_names) for b in model_names[i + 1:]]
    ):
        pair_rng = np.random.default_rng(seed + idx + 1)
        p = mantel_permutation_two_sided(
            baseline_matrices[model_a],
            baseline_matrices[model_b],
            permutations=mantel_permutations,
            rng=pair_rng,
        )
        p_values.append(p)
        significance_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "rho": baseline_pairwise[_pair_key(model_a, model_b)],
                "p_mantel_two_sided": p,
            }
        )
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

    def pairwise_for_concepts(subset: List[str]) -> Dict[Tuple[str, str], float]:
        flats = {}
        for model_name in model_names:
            matrix = build_similarity_matrix(resolved_models[model_name]["embeddings"], subset)
            flats[model_name] = upper_triangle_flat(matrix)
        return build_pairwise_rhos(flats, model_names)

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
            matrix = build_similarity_matrix(emb, base_concepts)
            template_flats[template_key] = upper_triangle_flat(matrix)

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
        for i, template_a in enumerate(template_keys_sorted):
            for template_b in template_keys_sorted[i + 1:]:
                template_pairwise.append(
                    {
                        "template_a": template_a,
                        "template_b": template_b,
                        "rho": _safe_spearman(
                            template_flats[template_a], template_flats[template_b]
                        ),
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
        embs_for_fraction = {}
        for model_name in model_names:
            candidates = model_layer_candidates[model_name]
            if "selected" in candidates:
                chosen = "selected"
                embs = raw_models[model_name]["embeddings"]
            else:
                chosen = aligned_layer_key_for_fraction(candidates, fraction)
                embs = raw_models[model_name]["embeddings_by_layer"][chosen]
            aligned_layer["model_layer_selection"][model_name][label] = chosen
            embs_for_fraction[model_name] = embs

        flats = {}
        for model_name in model_names:
            matrix = build_similarity_matrix(embs_for_fraction[model_name], base_concepts)
            flats[model_name] = upper_triangle_flat(matrix)
        pairwise = []
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1:]:
                pairwise.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "rho": _safe_spearman(flats[model_a], flats[model_b]),
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
    )
