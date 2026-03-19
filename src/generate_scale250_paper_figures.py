import gzip
import json
import shutil
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "manuscript" / "figures" / "scale250"

OLD_BASELINE_REP = REPO_ROOT / "results" / "baseline" / "replication_results.json.gz"
OLD_BASELINE_ROB = REPO_ROOT / "results" / "baseline" / "robustness" / "robustness_stats.json"

NEW_BASELINE_REP = REPO_ROOT / "results" / "scale250_full" / "baseline" / "replication_results.json"
NEW_BASELINE_ROB = (
    REPO_ROOT / "results" / "scale250_full" / "baseline" / "robustness_opt_full" / "robustness_stats.json"
)
NEW_ALIGNED5_REP = REPO_ROOT / "results" / "scale250_full" / "aligned5" / "replication_results.json"
NEW_ALIGNED5_ROB = REPO_ROOT / "results" / "scale250_full" / "aligned5" / "robustness" / "robustness_stats.json"

BASELINE_HEATMAP = (
    REPO_ROOT / "results" / "scale250_full" / "baseline" / "visualizations" / "heatmaps" / "rsa_matrix.png"
)
BASELINE_SCALING = (
    REPO_ROOT / "results" / "scale250_full" / "baseline" / "scaling" / "scaling_analysis.png"
)
ALIGNED5_HEATMAP = (
    REPO_ROOT / "results" / "scale250_full" / "aligned5" / "visualizations" / "heatmaps" / "rsa_matrix.png"
)
ALIGNED5_SCALING = (
    REPO_ROOT / "results" / "scale250_full" / "aligned5" / "scaling" / "scaling_analysis.png"
)
ARVLM25_ARCH_DIR = (
    REPO_ROOT / "results" / "scale250_full" / "baseline25_extension" / "architecture_analysis"
)
ARVLM25_SUMMARY = ARVLM25_ARCH_DIR / "architecture_summary.json"
ARVLM25_BRIDGE = ARVLM25_ARCH_DIR / "bridge_language_vs_vision.png"
ARVLM25_FAMILY_BLOCK = ARVLM25_ARCH_DIR / "family_block_mean_rsa.png"
ARVLM25_RSA = ARVLM25_ARCH_DIR / "rsa_matrix.png"


def load_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def broad_pair_type(type_map, model_a: str, model_b: str) -> str:
    type_a = type_map[model_a]
    type_b = type_map[model_b]
    if type_a == "causal" and type_b == "causal":
        return "language-language"
    if type_a != "causal" and type_b != "causal":
        return "image-image"
    return "language-image"


def compute_broad_summary(replication, robustness):
    type_map = {name: info["config"]["type"] for name, info in replication["models"].items()}
    rows = robustness["rsa_significance"]["pairwise_results"]
    summary = {}
    for pair_type in ["language-language", "image-image", "language-image"]:
        selected = [
            row
            for row in rows
            if broad_pair_type(type_map, row["model_a"], row["model_b"]) == pair_type
        ]
        summary[pair_type] = {
            "count": len(selected),
            "mean_rho": mean(row["rho"] for row in selected),
            "sig_fraction": sum(1 for row in selected if row["q_bh_fdr"] < 0.05) / len(selected),
        }
    return summary


def compute_prompt_summary(robustness):
    rows = robustness["prompt_sensitivity"]["models"]
    return sorted(
        (
            {
                "model": model,
                "max_abs_delta": info["max_abs_delta_vs_baseline"],
            }
            for model, info in rows.items()
        ),
        key=lambda row: row["max_abs_delta"],
        reverse=True,
    )


def compute_depth_summary(replication, robustness):
    type_map = {name: info["config"]["type"] for name, info in replication["models"].items()}
    fractions = robustness["aligned_layer"]["pairwise_by_fraction"]
    out = {"overall": {}}
    for key, rows in fractions.items():
        out["overall"][key] = mean(row["rho"] for row in rows)
        for pair_type in ["language-language", "image-image", "language-image"]:
            selected = [
                row
                for row in rows
                if broad_pair_type(type_map, row["model_a"], row["model_b"]) == pair_type
            ]
            out.setdefault(pair_type, {})[key] = mean(row["rho"] for row in selected)
    return out


def compute_scale_summary(replication):
    concepts = replication["concepts"]
    models = replication["models"]

    def sim_flat(embeddings):
        stacked = np.stack([np.asarray(embeddings[concept], dtype=np.float32) for concept in concepts], axis=0)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        stacked = stacked / norms
        matrix = stacked @ stacked.T
        idx = np.triu_indices(len(concepts), k=1)
        return matrix[idx]

    vision = {
        name: sim_flat(info["embeddings"])
        for name, info in models.items()
        if info["config"]["type"] == "vision"
    }
    rows = []
    for name, info in models.items():
        if info["config"]["type"] != "causal":
            continue
        size = info["config"].get("param_size_b")
        if size is None:
            continue
        lang_flat = sim_flat(info["embeddings"])
        mean_vision_rho = mean(
            float(spearmanr(lang_flat, vision_flat).statistic)
            for vision_flat in vision.values()
        )
        rows.append({"model": name, "size_b": float(size), "mean_vision_rho": mean_vision_rho})
    rows.sort(key=lambda row: row["size_b"])
    return rows


def stack_feature_matrices(replication, row_normalize: bool = True):
    concepts = replication["concepts"]
    matrices = {}
    for name, info in replication["models"].items():
        stacked = np.stack(
            [np.asarray(info["embeddings"][concept], dtype=np.float32) for concept in concepts],
            axis=0,
        )
        if row_normalize:
            norms = np.linalg.norm(stacked, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            stacked = stacked / norms
        stacked = stacked - stacked.mean(axis=0, keepdims=True)
        matrices[name] = stacked
    return matrices


def compute_linear_cka_summary(replication, robustness, row_normalize: bool = True):
    type_map = {name: info["config"]["type"] for name, info in replication["models"].items()}
    features = stack_feature_matrices(replication, row_normalize=row_normalize)
    model_names = sorted(features)

    centered_grams = {}
    gram_norms = {}
    for name, matrix in features.items():
        gram = matrix @ matrix.T
        centered_grams[name] = gram
        gram_norms[name] = float(np.sqrt(np.sum(gram * gram)))

    pairwise = []
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1:]:
            numerator = float(np.sum(centered_grams[model_a] * centered_grams[model_b]))
            denom = gram_norms[model_a] * gram_norms[model_b]
            cka = numerator / denom if denom > 0.0 else float("nan")
            pairwise.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "cka": cka,
                    "pair_type": broad_pair_type(type_map, model_a, model_b),
                }
            )

    matrix = np.eye(len(model_names), dtype=np.float32)
    index = {name: idx for idx, name in enumerate(model_names)}
    for row in pairwise:
        i = index[row["model_a"]]
        j = index[row["model_b"]]
        matrix[i, j] = row["cka"]
        matrix[j, i] = row["cka"]

    broad_summary = {}
    for pair_type in ["language-language", "image-image", "language-image"]:
        selected = [row["cka"] for row in pairwise if row["pair_type"] == pair_type]
        broad_summary[pair_type] = {
            "count": len(selected),
            "mean_cka": mean(selected),
            "median_cka": float(np.median(selected)),
        }

    rsa_rows = {
        (row["model_a"], row["model_b"]): row["rho"]
        for row in robustness["rsa_significance"]["pairwise_results"]
    }
    rsa_values = []
    cka_values = []
    for row in pairwise:
        key = (row["model_a"], row["model_b"])
        rsa = rsa_rows.get(key, rsa_rows.get((key[1], key[0])))
        if rsa is None:
            continue
        rsa_values.append(float(rsa))
        cka_values.append(float(row["cka"]))

    rsa_arr = np.asarray(rsa_values, dtype=np.float32)
    cka_arr = np.asarray(cka_values, dtype=np.float32)
    agreement = {
        "pearson": float(np.corrcoef(rsa_arr, cka_arr)[0, 1]),
        "spearman": float(spearmanr(rsa_arr, cka_arr).statistic),
    }

    return {
        "row_normalize": row_normalize,
        "model_names": model_names,
        "pairwise": pairwise,
        "matrix": matrix.tolist(),
        "broad_summary": broad_summary,
        "rsa_agreement": agreement,
    }


def make_design_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    old_counts = [20, 8]
    new_counts = [250, 0]
    labels = ["Base", "Compound"]
    colors = ["#2C7FB8", "#F28E2B"]
    x = np.arange(2)
    axes[0].bar(x, [old_counts[0], new_counts[0]], color=colors[0], width=0.55, label="Base")
    axes[0].bar(x, [old_counts[1], new_counts[1]], bottom=[old_counts[0], new_counts[0]], color=colors[1], width=0.55, label="Compound")
    axes[0].set_xticks(x, ["Previous", "Scale250"])
    axes[0].set_ylabel("Concepts")
    axes[0].set_title("Benchmark composition")
    for idx, total in enumerate([sum(old_counts), sum(new_counts)]):
        axes[0].text(idx, total + 5, str(total), ha="center", fontsize=10, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    sources = ["ImageNet", "OpenImages", "Unsplash"]
    source_vals = [5, 5, 5]
    source_colors = ["#4E79A7", "#59A14F", "#E15759"]
    bottom = 0
    for name, value, color in zip(sources, source_vals, source_colors):
        axes[1].bar([0], [value], bottom=[bottom], color=color, width=0.55, label=name)
        axes[1].text(0, bottom + value / 2, f"{name}\n{value}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        bottom += value
    axes[1].set_xticks([0], ["Per concept"])
    axes[1].set_ylim(0, 16)
    axes[1].set_ylabel("Images")
    axes[1].set_title("Within-concept source balance")

    families = ["Language", "Vision SSL", "Vision-Language"]
    family_counts = [8, 10, 4]
    family_colors = ["#E15759", "#4E79A7", "#76B7B2"]
    axes[2].bar(families, family_counts, color=family_colors, width=0.6)
    axes[2].set_ylabel("Models")
    axes[2].set_title("Core panel")
    axes[2].tick_params(axis="x", rotation=18)
    for idx, value in enumerate(family_counts):
        axes[2].text(idx, value + 0.2, str(value), ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Scale250 confirmatory design", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "design_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_old_vs_new_summary(old_rep, old_rob, new_rep, new_rob):
    old_summary = compute_broad_summary(old_rep, old_rob)
    new_summary = compute_broad_summary(new_rep, new_rob)
    categories = ["language-language", "image-image", "language-image"]
    labels = ["Language\nLanguage", "Image\nImage", "Language\nImage"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(categories))
    width = 0.34

    old_means = [old_summary[cat]["mean_rho"] for cat in categories]
    new_means = [new_summary[cat]["mean_rho"] for cat in categories]
    axes[0].bar(x - width / 2, old_means, width=width, color="#A0CBE8", label="Previous benchmark")
    axes[0].bar(x + width / 2, new_means, width=width, color="#4E79A7", label="Scale250")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean RSA (Spearman rho)")
    axes[0].set_title("Mean RSA by family pair")
    axes[0].axhline(0.0, color="#666666", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)

    old_sig = [old_summary[cat]["sig_fraction"] for cat in categories]
    new_sig = [new_summary[cat]["sig_fraction"] for cat in categories]
    axes[1].bar(x - width / 2, old_sig, width=width, color="#FFBE7D", label="Previous benchmark")
    axes[1].bar(x + width / 2, new_sig, width=width, color="#F28E2B", label="Scale250")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("FDR-significant fraction")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Significance rate by family pair")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Benchmark scale changes the cross-modal story", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "old_vs_new_family_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_depth_profile(new_aligned_rep, new_aligned_rob):
    summary = compute_depth_summary(new_aligned_rep, new_aligned_rob)
    order = ["d00", "d25", "d50", "d75", "d100"]
    x = np.arange(len(order))
    labels = ["d00", "d25", "d50", "d75", "d100"]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for key, color in [
        ("overall", "#222222"),
        ("language-language", "#E15759"),
        ("image-image", "#4E79A7"),
        ("language-image", "#59A14F"),
    ]:
        y = [summary[key][depth] for depth in order]
        ax.plot(x, y, marker="o", linewidth=2.2, color=color, label=key.replace("-", " "))
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean RSA (Spearman rho)")
    ax.set_title("Aligned5 depth profile")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "aligned5_depth_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_prompt_sensitivity(new_rob):
    rows = compute_prompt_summary(new_rob)
    models = [row["model"] for row in rows]
    values = [row["max_abs_delta"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    y = np.arange(len(models))
    ax.barh(y, values, color="#B07AA1")
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlabel("Max absolute delta vs baseline template")
    ax.set_title("Prompt sensitivity across language models")
    for idx, value in enumerate(values):
        ax.text(value + 0.002, idx, f"{value:.3f}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prompt_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_cka_heatmap(cka_summary):
    model_names = cka_summary["model_names"]
    matrix = np.asarray(cka_summary["matrix"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(11.5, 9.6))
    im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(model_names, fontsize=8)
    ax.set_title("Baseline centered linear CKA across 250 concepts", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA", rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_cka_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_rsa_cka_scatter(cka_summary, robustness):
    rsa_rows = {
        (row["model_a"], row["model_b"]): row["rho"]
        for row in robustness["rsa_significance"]["pairwise_results"]
    }
    colors = {
        "language-language": "#E15759",
        "image-image": "#4E79A7",
        "language-image": "#59A14F",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for pair_type in ["language-language", "image-image", "language-image"]:
        xs = []
        ys = []
        for row in cka_summary["pairwise"]:
            if row["pair_type"] != pair_type:
                continue
            key = (row["model_a"], row["model_b"])
            rsa = rsa_rows.get(key, rsa_rows.get((key[1], key[0])))
            if rsa is None:
                continue
            xs.append(float(rsa))
            ys.append(float(row["cka"]))
        ax.scatter(xs, ys, s=28, alpha=0.75, color=colors[pair_type], label=pair_type.replace("-", " "))
    ax.set_xlabel("RSA (Spearman rho)")
    ax.set_ylabel("Centered linear CKA")
    agreement = cka_summary["rsa_agreement"]
    ax.set_title(
        "Metric triangulation: RSA vs CKA\n"
        f"Pearson={agreement['pearson']:.2f}, Spearman={agreement['spearman']:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rsa_vs_cka_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def copy_existing_figures():
    copies = {
        BASELINE_HEATMAP: FIG_DIR / "baseline_rsa_heatmap.png",
        BASELINE_SCALING: FIG_DIR / "baseline_scaling.png",
        ALIGNED5_HEATMAP: FIG_DIR / "aligned5_rsa_heatmap.png",
        ALIGNED5_SCALING: FIG_DIR / "aligned5_scaling.png",
        ARVLM25_BRIDGE: FIG_DIR / "arvlm_bridge_language_vs_vision.png",
        ARVLM25_FAMILY_BLOCK: FIG_DIR / "arvlm_family_block_mean_rsa.png",
        ARVLM25_RSA: FIG_DIR / "arvlm25_rsa_heatmap.png",
    }
    for src, dst in copies.items():
        shutil.copy2(src, dst)


def write_stats_summary(
    old_rep,
    old_rob,
    new_rep,
    new_rob,
    new_aligned_rep,
    new_aligned_rob,
    new_cka_summary,
):
    arvlm_summary = load_json(ARVLM25_SUMMARY)
    payload = {
        "old_broad_summary": compute_broad_summary(old_rep, old_rob),
        "new_broad_summary": compute_broad_summary(new_rep, new_rob),
        "prompt_sensitivity": compute_prompt_summary(new_rob),
        "aligned_depth_summary": compute_depth_summary(new_aligned_rep, new_aligned_rob),
        "scale_summary": compute_scale_summary(new_rep),
        "baseline_cka_summary": new_cka_summary,
        "arvlm25_extension_summary": arvlm_summary,
    }
    with open(FIG_DIR / "paper_stats_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    old_rep = load_json(OLD_BASELINE_REP)
    old_rob = load_json(OLD_BASELINE_ROB)
    new_rep = load_json(NEW_BASELINE_REP)
    new_rob = load_json(NEW_BASELINE_ROB)
    new_aligned_rep = load_json(NEW_ALIGNED5_REP)
    new_aligned_rob = load_json(NEW_ALIGNED5_ROB)

    make_design_overview()
    make_old_vs_new_summary(old_rep, old_rob, new_rep, new_rob)
    make_depth_profile(new_aligned_rep, new_aligned_rob)
    make_prompt_sensitivity(new_rob)
    new_cka_summary = compute_linear_cka_summary(new_rep, new_rob, row_normalize=True)
    make_cka_heatmap(new_cka_summary)
    make_rsa_cka_scatter(new_cka_summary, new_rob)
    copy_existing_figures()
    write_stats_summary(
        old_rep,
        old_rob,
        new_rep,
        new_rob,
        new_aligned_rep,
        new_aligned_rob,
        new_cka_summary,
    )

    print(f"Wrote paper figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
