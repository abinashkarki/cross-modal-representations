import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import rankdata


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_FILE = os.path.join(
    EXPERIMENT_DIR,
    "results",
    "scale250_full",
    "baseline25_extension",
    "replication_results.json",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    EXPERIMENT_DIR,
    "results",
    "scale250_full",
    "baseline25_extension",
    "architecture_analysis",
)

KNOWN_COMPOUNDS: set[str] = set()

FAMILY_ORDER = [
    "language",
    "contrastive_vlm",
    "autoregressive_vlm",
    "vision",
]

FAMILY_LABELS = {
    "language": "Language",
    "contrastive_vlm": "Contrastive VLM",
    "autoregressive_vlm": "Autoregressive VLM",
    "vision": "Vision",
}

FAMILY_COLORS = {
    "language": "#E74C3C",
    "contrastive_vlm": "#3498DB",
    "autoregressive_vlm": "#16A085",
    "vision": "#9B59B6",
}


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


def stack_embeddings(embeddings: Dict[str, Any], concepts: List[str]) -> np.ndarray:
    return np.stack([np.asarray(embeddings[concept], dtype=np.float32) for concept in concepts], axis=0)


def build_similarity_matrix_from_stacked(stacked_embeddings: np.ndarray) -> np.ndarray:
    normalized = _row_normalize(stacked_embeddings)
    matrix = normalized @ normalized.T
    np.clip(matrix, -1.0, 1.0, out=matrix)
    return matrix.astype(np.float32, copy=False)


def spearman_correlation_matrix(rows: np.ndarray) -> np.ndarray:
    ranked = np.empty(rows.shape, dtype=np.float32)
    for idx, row in enumerate(rows):
        ranked[idx] = rankdata(row, method="average").astype(np.float32)
    ranked -= ranked.mean(axis=1, keepdims=True)
    ranked = _row_normalize(ranked)
    corr = ranked @ ranked.T
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr.astype(np.float32, copy=False)


def model_family(model_type: str) -> str:
    if model_type == "causal":
        return "language"
    if model_type == "vision_language":
        return "contrastive_vlm"
    if model_type == "vision_language_autoregressive":
        return "autoregressive_vlm"
    return "vision"


def ordered_models(data_models: Dict[str, Any]) -> List[str]:
    grouped: Dict[str, List[str]] = {family: [] for family in FAMILY_ORDER}
    for name, payload in data_models.items():
        grouped[model_family(payload["config"]["type"])].append(name)
    ordered: List[str] = []
    for family in FAMILY_ORDER:
        ordered.extend(sorted(grouped[family]))
    return ordered


def pairwise_rsa(models_data: Dict[str, Any], concepts: List[str], model_names: List[str]) -> np.ndarray:
    triu_idx = np.triu_indices(len(concepts), k=1)
    flat_rows = []
    for name in model_names:
        embeddings = models_data[name]["embeddings"]
        stacked = stack_embeddings(embeddings, concepts)
        sim = build_similarity_matrix_from_stacked(stacked)
        flat_rows.append(sim[triu_idx].astype(np.float32, copy=False))
    flat_rows_arr = np.stack(flat_rows, axis=0)
    return spearman_correlation_matrix(flat_rows_arr)


def mean_to_family(
    rsa_matrix: np.ndarray,
    model_names: List[str],
    family_map: Dict[str, str],
    src_model: str,
    target_family: str,
) -> float:
    src_idx = model_names.index(src_model)
    targets = [
        model_names.index(name)
        for name in model_names
        if family_map[name] == target_family and name != src_model
    ]
    if not targets:
        return float("nan")
    vals = rsa_matrix[src_idx, targets]
    return float(np.mean(vals))


def build_family_block_matrix(
    rsa_matrix: np.ndarray,
    model_names: List[str],
    family_map: Dict[str, str],
) -> Tuple[np.ndarray, List[str]]:
    block = np.full((len(FAMILY_ORDER), len(FAMILY_ORDER)), np.nan, dtype=np.float32)
    for i, fam_a in enumerate(FAMILY_ORDER):
        idx_a = [k for k, name in enumerate(model_names) if family_map[name] == fam_a]
        for j, fam_b in enumerate(FAMILY_ORDER):
            idx_b = [k for k, name in enumerate(model_names) if family_map[name] == fam_b]
            vals = []
            for a in idx_a:
                for b in idx_b:
                    if fam_a == fam_b and a == b:
                        continue
                    vals.append(rsa_matrix[a, b])
            if vals:
                block[i, j] = float(np.mean(vals))
    return block, [FAMILY_LABELS[f] for f in FAMILY_ORDER]


def plot_bridge_bar(summary_rows: List[Dict[str, Any]], output_path: str) -> None:
    labels = [row["model"] for row in summary_rows]
    lang_vals = [row["mean_to_language"] for row in summary_rows]
    vis_vals = [row["mean_to_vision"] for row in summary_rows]
    colors = [FAMILY_COLORS[row["family"]] for row in summary_rows]

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(max(10, len(labels) * 0.9), 6))
    plt.bar(x - width / 2, lang_vals, width, color="#E74C3C", alpha=0.8, label="Mean RSA to language")
    plt.bar(x + width / 2, vis_vals, width, color="#9B59B6", alpha=0.8, label="Mean RSA to vision")
    for idx, row in enumerate(summary_rows):
        plt.scatter([], [], color=colors[idx], label=FAMILY_LABELS[row["family"]])
    handles, labels_seen = plt.gca().get_legend_handles_labels()
    dedup_handles = []
    dedup_labels = []
    for h, l in zip(handles, labels_seen):
        if l not in dedup_labels:
            dedup_handles.append(h)
            dedup_labels.append(l)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Mean RSA")
    plt.title("Bridge Models: Mean Similarity to Language vs Vision Families", fontweight="bold")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(dedup_handles, dedup_labels, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_family_block(block: np.ndarray, labels: List[str], output_path: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        block,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        vmin=np.nanmin(block),
        vmax=np.nanmax(block),
    )
    plt.title("Family-Level Mean RSA", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rsa_heatmap(rsa_matrix: np.ndarray, labels: List[str], output_path: str) -> None:
    fig_size = max(10, len(labels) * 0.9)
    ann_size = max(5, 11 - len(labels) // 4)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        rsa_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        vmin=float(np.min(rsa_matrix)),
        vmax=float(np.max(rsa_matrix)),
        annot_kws={"size": ann_size},
    )
    plt.title("25-Model RSA Heatmap (Selected Layer)", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=ann_size)
    plt.yticks(fontsize=ann_size)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(data_file: str, output_dir: str) -> None:
    data_file = os.path.abspath(data_file)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    concepts = [c for c in data["concepts"] if c not in KNOWN_COMPOUNDS]
    raw_models = data["models"]
    models = {}
    for name, payload in raw_models.items():
        updated = dict(payload)
        updated["embeddings"] = payload.get("embeddings") or payload["embeddings_by_layer"][payload["layer_metadata"]["default_layer_key"]]
        models[name] = updated

    model_names = ordered_models(models)
    family_map = {name: model_family(models[name]["config"]["type"]) for name in model_names}
    rsa_matrix = pairwise_rsa(models, concepts, model_names)

    bridge_rows: List[Dict[str, Any]] = []
    for name in model_names:
        family = family_map[name]
        if family not in {"contrastive_vlm", "autoregressive_vlm"}:
            continue
        row = {
            "model": name,
            "family": family,
            "mean_to_language": mean_to_family(rsa_matrix, model_names, family_map, name, "language"),
            "mean_to_vision": mean_to_family(rsa_matrix, model_names, family_map, name, "vision"),
            "mean_to_contrastive_vlm": mean_to_family(rsa_matrix, model_names, family_map, name, "contrastive_vlm"),
            "mean_to_autoregressive_vlm": mean_to_family(rsa_matrix, model_names, family_map, name, "autoregressive_vlm"),
        }
        row["vision_minus_language"] = row["mean_to_vision"] - row["mean_to_language"]
        bridge_rows.append(row)

    block, block_labels = build_family_block_matrix(rsa_matrix, model_names, family_map)

    summary = {
        "data_file": data_file,
        "n_models": len(model_names),
        "n_concepts": len(concepts),
        "model_order": model_names,
        "family_map": family_map,
        "bridge_summary": bridge_rows,
        "family_block_labels": block_labels,
        "family_block_mean_rsa": block.tolist(),
    }

    json_path = os.path.join(output_dir, "architecture_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    bridge_plot = os.path.join(output_dir, "bridge_language_vs_vision.png")
    plot_bridge_bar(bridge_rows, bridge_plot)

    block_plot = os.path.join(output_dir, "family_block_mean_rsa.png")
    plot_family_block(block, block_labels, block_plot)

    rsa_plot = os.path.join(output_dir, "rsa_matrix.png")
    plot_rsa_heatmap(rsa_matrix, model_names, rsa_plot)

    print(f"Saved summary JSON -> {json_path}")
    print(f"Saved bridge plot -> {bridge_plot}")
    print(f"Saved family block plot -> {block_plot}")
    print(f"Saved RSA heatmap -> {rsa_plot}")

    print("\nBridge summary:")
    for row in bridge_rows:
        print(
            f"{row['model']}: "
            f"lang={row['mean_to_language']:.4f}, "
            f"vision={row['mean_to_vision']:.4f}, "
            f"contrastive={row['mean_to_contrastive_vlm']:.4f}, "
            f"autoregressive={row['mean_to_autoregressive_vlm']:.4f}, "
            f"vision-language delta={row['vision_minus_language']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze autoregressive VLM extension in compiled results.")
    parser.add_argument("--data-file", type=str, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    main(args.data_file, args.output_dir)
