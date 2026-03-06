"""
Scaling analysis: plot key metrics against language model size.

Uses existing replication_results.json — no new extraction needed.
Produces plots in results/replication/scaling/.
"""

import os
import re
import json
import argparse
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_FILE = os.path.join(EXPERIMENT_DIR, "results", "replication", "replication_results.json")
DEFAULT_OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "results", "replication", "scaling")

LEGACY_LANGUAGE_MODEL_SIZES = {
    "Qwen3-0.6B-MLX-4bit": 0.6,
    "Qwen3-0.6B-MLX-8bit": 0.6,
    "Qwen3-1.7B-MLX-4bit": 1.7,
    "Qwen3-1.7B-MLX-8bit": 1.7,
    "Qwen2.5-1.5B-Instruct-4bit": 1.54,
    "Qwen2.5-1.5B-Instruct-8bit": 1.54,
    "Falcon3-1B-Instruct-4bit": 1.0,
    "Falcon3-1B-Instruct-8bit": 1.0,
    "Granite-3.3-2B-Instruct-4bit": 2.0,
    "Granite-3.3-2B-Instruct-8bit": 2.0,
    "LFM2-2.6B-Exp-4bit": 2.6,
    "LFM2-2.6B-Exp-8bit": 2.6,
    "SmolLM3-3B-4bit": 3.0,
    "SmolLM3-3B-8bit": 3.0,
    "Qwen3-4B-MLX-4bit": 4.0,
    "Qwen3-4B-MLX-8bit": 4.0,
}

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

COMPOUND_CONCEPTS = COMPOUND_CONCEPTS_V1 + COMPOUND_CONCEPTS_V2


def load_data(data_file: str):
    with open(data_file, "r") as f:
        return json.load(f)


def build_sim_matrix(embeddings, concepts):
    n = len(concepts)
    mat = np.zeros((n, n))
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            e1 = np.array(embeddings[c1]).reshape(1, -1)
            e2 = np.array(embeddings[c2]).reshape(1, -1)
            mat[i, j] = cosine_similarity(e1, e2)[0][0]
    return mat


def get_upper_triangle(matrix):
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def bootstrap_rsa_ci(flat_a, flat_b, n_bootstrap=5000, ci=0.95):
    """Bootstrap CI for Spearman ρ by resampling concept pairs."""
    n = len(flat_a)
    rng = np.random.default_rng(42)
    rhos = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho, _ = spearmanr(flat_a[idx], flat_b[idx])
        rhos.append(rho)
    alpha = (1 - ci) / 2
    return float(np.quantile(rhos, alpha)), float(np.quantile(rhos, 1 - alpha))


def balance_compositionality(embeddings):
    scores = []
    for comp1, comp2, compound in COMPOUND_CONCEPTS:
        if compound not in embeddings:
            continue
        e1 = np.array(embeddings[comp1]).reshape(1, -1)
        e2 = np.array(embeddings[comp2]).reshape(1, -1)
        ec = np.array(embeddings[compound]).reshape(1, -1)
        s1 = cosine_similarity(ec, e1)[0][0]
        s2 = cosine_similarity(ec, e2)[0][0]
        total = s1 + s2
        balance = s2 / total if total > 0 else 0.5
        scores.append(1 - abs(0.5 - balance) * 2)
    return float(np.mean(scores)) if scores else 0.0


def additive_compositionality(embeddings):
    scores = []
    for comp1, comp2, compound in COMPOUND_CONCEPTS:
        if compound not in embeddings:
            continue
        e1 = np.array(embeddings[comp1]).flatten()
        e2 = np.array(embeddings[comp2]).flatten()
        ec = np.array(embeddings[compound]).flatten()
        e_sum = e1 + e2
        cos = cosine_similarity(ec.reshape(1, -1), e_sum.reshape(1, -1))[0][0]
        scores.append(cos)
    return float(np.mean(scores)) if scores else 0.0


def get_language_model_size(model_name, model_info):
    """Read model size from config when available; fall back to legacy map."""
    cfg = model_info.get("config", {})
    size = cfg.get("param_size_b")
    if isinstance(size, (int, float)):
        return float(size)
    return LEGACY_LANGUAGE_MODEL_SIZES.get(model_name)


def layer_suffix(layer_name: str) -> str:
    if layer_name in {"selected", "default"}:
        return ""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", layer_name)
    return f"__{safe}"


def _numeric_layer_sort_key(name: str) -> int:
    if name.startswith("layer_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return 10**9
    if name == "layer_last":
        return 10**9 + 1
    return 10**9 + 2


def resolve_embeddings_for_layer(
    model_name: str,
    model_info: Dict[str, Any],
    requested_layer: str,
) -> Tuple[Dict[str, Any], str]:
    layered = model_info.get("embeddings_by_layer")
    layer_meta = model_info.get("layer_metadata", {})

    if requested_layer in {"selected", "default"}:
        return model_info["embeddings"], layer_meta.get("default_layer_key", "selected")
    if not layered:
        print(f"WARNING: {model_name} has no embeddings_by_layer; using selected/default embeddings.")
        return model_info["embeddings"], layer_meta.get("default_layer_key", "selected")
    if requested_layer in {"last", "-1"}:
        default_key = layer_meta.get("default_layer_key")
        if default_key in layered:
            return layered[default_key], default_key
        keys = sorted(layered.keys(), key=_numeric_layer_sort_key)
        return layered[keys[-1]], keys[-1]
    if requested_layer in layered:
        return layered[requested_layer], requested_layer
    print(
        f"WARNING: {model_name} missing requested layer '{requested_layer}'; "
        "using selected/default embeddings."
    )
    return model_info["embeddings"], layer_meta.get("default_layer_key", "selected")


def main(layer: str, data_file: str, output_dir: str) -> None:
    data_file = os.path.abspath(data_file)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        data = load_data(data_file)
    except FileNotFoundError:
        print(f"Error: {data_file} not found.")
        return
    concepts = data["concepts"]
    raw_models = data["models"]
    suffix = layer_suffix(layer)

    models = {}
    resolved_layers = {}
    for name, info in raw_models.items():
        embeddings, resolved = resolve_embeddings_for_layer(name, info, layer)
        updated = dict(info)
        updated["embeddings"] = embeddings
        models[name] = updated
        resolved_layers[name] = resolved

    base_concepts = [c for c in concepts if c not in
                     [cc[2] for cc in COMPOUND_CONCEPTS]]

    lang_models = {}
    for name, info in models.items():
        if info["config"]["type"] != "causal":
            continue
        size = get_language_model_size(name, info)
        if size is None:
            continue
        lang_models[name] = {"info": info, "size": size}
    vision_models = {
        name: info for name, info in models.items()
        if info["config"]["type"] == "vision"
    }

    if not lang_models:
        print("No language models found in replication data.")
        return
    if not vision_models:
        print("No vision models found in replication data.")
        return

    print(f"Language models: {list(lang_models.keys())}")
    print(f"Vision models:   {list(vision_models.keys())}")
    print(f"Layer mode: requested='{layer}'")
    print(f"Resolved layers: {resolved_layers}")

    vision_flat = {}
    for vname, vinfo in vision_models.items():
        mat = build_sim_matrix(vinfo["embeddings"], base_concepts)
        vision_flat[vname] = get_upper_triangle(mat)

    sizes = []
    balance_scores = []
    additive_scores = []
    rsa_vs_vision    = {vn: [] for vn in vision_models}
    rsa_ci_lo        = {vn: [] for vn in vision_models}
    rsa_ci_hi        = {vn: [] for vn in vision_models}

    sorted_lang = sorted(lang_models.items(), key=lambda kv: kv[1]["size"])

    for lname, lentry in sorted_lang:
        size = lentry["size"]
        embs = lentry["info"]["embeddings"]

        sizes.append(size)
        balance_scores.append(balance_compositionality(embs))
        additive_scores.append(additive_compositionality(embs))

        lang_mat = build_sim_matrix(embs, base_concepts)
        lang_flat = get_upper_triangle(lang_mat)

        for vname, vflat in vision_flat.items():
            rho, _ = spearmanr(lang_flat, vflat)
            rsa_vs_vision[vname].append(rho)
            lo, hi = bootstrap_rsa_ci(lang_flat, vflat)
            rsa_ci_lo[vname].append(lo)
            rsa_ci_hi[vname].append(hi)

    # --- Plot 1: Compositionality vs Scale ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(sizes, balance_scores, "o-", color="#E74C3C", linewidth=2,
            markersize=8, label="Balance metric")
    ax.plot(sizes, additive_scores, "s--", color="#2980B9", linewidth=2,
            markersize=8, label="Additive metric")
    for i, name in enumerate([n for n, _ in sorted_lang]):
        ax.annotate(name, (sizes[i], balance_scores[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8)
    ax.set_xlabel("Model Size (B params)", fontsize=11)
    ax.set_ylabel("Avg Compositionality Score", fontsize=11)
    ax.set_title("Compositionality vs Language Model Scale", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- Plot 2: RSA with vision models vs Scale (with 95% bootstrap CI) ---
    ax = axes[1]
    markers = ["o", "s", "D", "^", "v", "P", "X", "h", "*", "p", "d", "<", ">", "H"]
    colors = [
        "#9B59B6", "#27AE60", "#F39C12", "#1ABC9C", "#E74C3C",
        "#3498DB", "#E67E22", "#2ECC71", "#8E44AD", "#16A085",
        "#D35400", "#2980B9", "#C0392B", "#7F8C8D",
    ]
    sizes_arr = np.array(sizes)
    for idx, (vname, rhos) in enumerate(rsa_vs_vision.items()):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        rhos_arr = np.array(rhos)
        lo_arr   = np.array(rsa_ci_lo[vname])
        hi_arr   = np.array(rsa_ci_hi[vname])
        ax.plot(sizes_arr, rhos_arr, f"{m}-", color=c, linewidth=2,
                markersize=8, label=f"vs {vname}")
        ax.fill_between(sizes_arr, lo_arr, hi_arr, color=c, alpha=0.12)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Model Size (B params)", fontsize=11)
    ax.set_ylabel("Spearman ρ (RSA)", fontsize=11)
    ax.set_title(
        "Language–Vision RSA vs Language Model Scale\n"
        f"(shaded = 95% bootstrap CI) | layer={layer}",
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"scaling_analysis{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS SUMMARY")
    print("=" * 80)
    header = f"{'Model':<22} {'Size':>5} {'Balance':>8} {'Additive':>9}"
    for vn in vision_models:
        short = vn[:10]
        header += f"  {'ρ/' + short:>18}"
    print(header)
    print("-" * len(header))

    for i, (lname, _) in enumerate(sorted_lang):
        row = (f"{lname:<22} {sizes[i]:>5.1f} "
               f"{balance_scores[i]:>8.3f} {additive_scores[i]:>9.3f}")
        for vn in vision_models:
            rho = rsa_vs_vision[vn][i]
            lo  = rsa_ci_lo[vn][i]
            hi  = rsa_ci_hi[vn][i]
            cell = f"{rho:.2f}[{lo:.2f},{hi:.2f}]"
            row += f"  {cell:>18}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling analysis for replication results.")
    parser.add_argument(
        "--data-file",
        type=str,
        default=DEFAULT_DATA_FILE,
        help=f"Compiled replication JSON path (default: {DEFAULT_DATA_FILE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for scaling plots (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="selected",
        help="Layer key to analyze: selected|default|last|-1|layer_N.",
    )
    args = parser.parse_args()
    main(args.layer, data_file=args.data_file, output_dir=args.output_dir)
