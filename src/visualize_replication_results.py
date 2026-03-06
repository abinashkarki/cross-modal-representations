import argparse
import os
import json
import re
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "results", "replication")
DEFAULT_DATA_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "replication_results.json")

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def balance_score(emb_compound, emb1, emb2):
    """Balance compositionality: 1 = perfectly equidistant, 0 = fully biased."""
    s1 = cosine_similarity(emb_compound, emb1)[0][0]
    s2 = cosine_similarity(emb_compound, emb2)[0][0]
    total = s1 + s2
    ratio = s2 / total if total > 0 else 0.5
    return 1.0 - abs(0.5 - ratio) * 2


def additive_score(emb_compound, emb1, emb2):
    """Additive compositionality: cosine sim between compound and sum of parts."""
    v1 = emb1.flatten()
    v2 = emb2.flatten()
    vc = emb_compound.flatten()
    v_sum = v1 + v2
    return cosine_similarity(vc.reshape(1, -1), v_sum.reshape(1, -1))[0][0]


def compute_compo_scores(embeddings, compound_list):
    """Return (balance_scores, additive_scores) lists for a compound list."""
    bal, add = [], []
    for comp1, comp2, compound in compound_list:
        if compound not in embeddings:
            continue
        e1 = np.array(embeddings[comp1]).reshape(1, -1)
        e2 = np.array(embeddings[comp2]).reshape(1, -1)
        ec = np.array(embeddings[compound]).reshape(1, -1)
        bal.append(balance_score(ec, e1, e2))
        add.append(additive_score(ec, e1, e2))
    return bal, add


def bootstrap_rsa_ci(flat_a, flat_b, n_bootstrap=5000, ci=0.95):
    """Bootstrap 95% CI for Spearman ρ by resampling concept pairs."""
    n = len(flat_a)
    rhos = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho, _ = spearmanr(flat_a[idx], flat_b[idx])
        rhos.append(rho)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(rhos, [alpha, 1 - alpha])
    return float(lo), float(hi)


def model_category(mtype):
    if mtype == "causal":
        return "causal"
    if mtype == "vision_language":
        return "vision_language"
    return "vision"


def category_color(cat):
    return {"causal": "#E74C3C", "vision_language": "#3498DB", "vision": "#9B59B6"}[cat]


def ordered_models_and_colors(models_data):
    cats = {"causal": [], "vision_language": [], "vision": []}
    for m, mdata in models_data.items():
        cats[model_category(mdata["config"]["type"])].append(m)
    ordered = cats["causal"] + cats["vision_language"] + cats["vision"]
    colors = (
        [category_color("causal")] * len(cats["causal"])
        + [category_color("vision_language")] * len(cats["vision_language"])
        + [category_color("vision")] * len(cats["vision"])
    )
    return ordered, colors


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
    model_data: Dict[str, Any],
    requested_layer: str,
) -> Tuple[Dict[str, Any], str]:
    layer_meta = model_data.get("layer_metadata", {})
    layered = model_data.get("embeddings_by_layer")

    if requested_layer in {"selected", "default"}:
        return model_data["embeddings"], layer_meta.get("default_layer_key", "selected")

    if not layered:
        print(
            f"WARNING: {model_name} has no embeddings_by_layer; using selected/default embeddings."
        )
        return model_data["embeddings"], layer_meta.get("default_layer_key", "selected")

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
    return model_data["embeddings"], layer_meta.get("default_layer_key", "selected")


LEGEND_ELEMENTS = [
    Line2D([0], [0], color="#E74C3C", lw=4, label="Language Models (Causal)"),
    Line2D([0], [0], color="#3498DB", lw=4, label="Vision-Language"),
    Line2D([0], [0], color="#9B59B6", lw=4, label="Vision Models (SSL)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(layer: str, data_file: str, output_dir: str) -> None:
    data_file = os.path.abspath(data_file)
    output_dir = os.path.abspath(output_dir)
    heatmaps_dir = os.path.join(output_dir, "heatmaps")
    compositionality_dir = os.path.join(output_dir, "compositionality")
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(compositionality_dir, exist_ok=True)

    print(f"Loading data from {data_file}...")
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {data_file} not found.")
        return

    all_concepts = data["concepts"]
    raw_models_data = data["models"]

    models_data = {}
    resolved_layers = {}
    for model_name, model_data in raw_models_data.items():
        embeddings, resolved_layer = resolve_embeddings_for_layer(
            model_name, model_data, layer
        )
        updated = dict(model_data)
        updated["embeddings"] = embeddings
        models_data[model_name] = updated
        resolved_layers[model_name] = resolved_layer

    base_concepts = [c for c in all_concepts if c not in KNOWN_COMPOUNDS]
    model_names = list(models_data.keys())
    suffix = layer_suffix(layer)

    # Filter compound lists to what's actually in the dataset
    v1_list = [t for t in COMPOUND_CONCEPTS_V1 if t[2] in all_concepts]
    v2_list = [t for t in COMPOUND_CONCEPTS_V2 if t[2] in all_concepts]
    all_compound_list = v1_list + v2_list

    print(
        f"Loaded {len(all_concepts)} concepts ({len(base_concepts)} base, "
        f"{len(all_compound_list)} compounds) for {len(model_names)} models."
    )
    print(f"Layer mode: requested='{layer}'")
    print(f"Resolved layers: {resolved_layers}")

    # ── 1. Similarity matrices (base concepts only) ───────────────────────────
    print("Computing similarity matrices (base concepts only)...")
    sim_matrices = {}
    for model_name, mdata in models_data.items():
        embs = mdata["embeddings"]
        n = len(base_concepts)
        mat = np.zeros((n, n))
        for i, c1 in enumerate(base_concepts):
            for j, c2 in enumerate(base_concepts):
                e1 = np.array(embs[c1]).reshape(1, -1)
                e2 = np.array(embs[c2]).reshape(1, -1)
                mat[i, j] = cosine_similarity(e1, e2)[0][0]
        sim_matrices[model_name] = mat

    # ── 2. RSA matrix with bootstrap CIs ─────────────────────────────────────
    n_pairs = len(base_concepts) * (len(base_concepts) - 1) // 2
    print(f"Computing RSA ({len(base_concepts)} base concepts, {n_pairs} pairs)...")
    triu_idx = np.triu_indices(len(base_concepts), k=1)
    flat_sims = {m: sim_matrices[m][triu_idx] for m in model_names}

    rsa_matrix = np.zeros((len(model_names), len(model_names)))
    rsa_ci_lo  = np.zeros_like(rsa_matrix)
    rsa_ci_hi  = np.zeros_like(rsa_matrix)

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            rho, _ = spearmanr(flat_sims[m1], flat_sims[m2])
            rsa_matrix[i, j] = rho
            lo, hi = bootstrap_rsa_ci(flat_sims[m1], flat_sims[m2])
            rsa_ci_lo[i, j] = lo
            rsa_ci_hi[i, j] = hi

    # Heatmap
    n_models = len(model_names)
    fig_size = max(10, n_models * 0.9)
    ann_size = max(6, 12 - n_models // 4)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(rsa_matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=model_names, yticklabels=model_names,
                vmin=0, vmax=1, annot_kws={"size": ann_size})
    plt.title(
        "Representational Similarity Analysis (RSA)\n"
        f"Spearman ρ of Concept Similarities (base concepts only) | layer={layer}",
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=ann_size)
    plt.yticks(fontsize=ann_size)
    plt.tight_layout()
    plt.savefig(f"{heatmaps_dir}/rsa_matrix{suffix}.png", dpi=150)
    plt.close()
    print(f"Saved RSA Matrix → {heatmaps_dir}/rsa_matrix{suffix}.png")

    # Print RSA table with 95% CI
    print("\nRSA Point Estimates with 95% Bootstrap CI (upper triangle):")
    print(f"{'':>20}", end="")
    for m in model_names:
        print(f"  {m[:12]:>12}", end="")
    print()
    for i, m1 in enumerate(model_names):
        print(f"{m1[:20]:<20}", end="")
        for j, m2 in enumerate(model_names):
            if j < i:
                print(f"  {'':>12}", end="")
            else:
                rho = rsa_matrix[i, j]
                lo  = rsa_ci_lo[i, j]
                hi  = rsa_ci_hi[i, j]
                cell = f"{rho:.2f}[{lo:.2f},{hi:.2f}]"
                print(f"  {cell:>12}", end="")
        print()

    # ── 3. Compositionality Analysis ──────────────────────────────────────────
    print("\nComputing compositionality scores...")
    if not all_compound_list:
        print("No compound concepts found – skipping compositionality.")
    else:
        compo = {}
        for model_name, mdata in models_data.items():
            embs = mdata["embeddings"]
            v1_bal, v1_add = compute_compo_scores(embs, v1_list)
            v2_bal, v2_add = compute_compo_scores(embs, v2_list)
            all_bal, all_add = compute_compo_scores(embs, all_compound_list)
            compo[model_name] = {
                "v1_balance":   float(np.mean(v1_bal))  if v1_bal  else None,
                "v2_balance":   float(np.mean(v2_bal))  if v2_bal  else None,
                "all_balance":  float(np.mean(all_bal)) if all_bal else None,
                "v1_additive":  float(np.mean(v1_add))  if v1_add  else None,
                "v2_additive":  float(np.mean(v2_add))  if v2_add  else None,
                "all_additive": float(np.mean(all_add)) if all_add else None,
            }

        ordered, colors = ordered_models_and_colors(models_data)
        label_size = max(7, 11 - len(ordered) // 5)
        fig_w = max(14, len(ordered) * 1.0)

        # ── 3a. Overall summary: balance + additive side by side ──────────────
        all_bal_vals = [compo[m]["all_balance"]  or 0 for m in ordered]
        all_add_vals = [compo[m]["all_additive"] or 0 for m in ordered]

        x = np.arange(len(ordered))
        bar_w = 0.38
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        b1 = ax.bar(x - bar_w / 2, all_bal_vals, bar_w,
                    color=colors, alpha=0.85, edgecolor="black", linewidth=1.2,
                    label="Balance metric")
        b2 = ax.bar(x + bar_w / 2, all_add_vals, bar_w,
                    color=colors, alpha=0.45, edgecolor="black", linewidth=1.2,
                    hatch="//", label="Additive metric")
        for bar, v in zip(b1, all_bal_vals):
            ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=label_size, fontweight="bold")
        for bar, v in zip(b2, all_add_vals):
            ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=label_size - 1)
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.4)
        ax.axhline(y=0.5, color="orange", linestyle=":", alpha=0.4)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=40, ha="right", fontsize=label_size)
        ax.set_ylabel("Compositionality Score")
        ax.set_title("Compositionality — Balance & Additive Metrics (all compounds)",
                     fontsize=13, fontweight="bold")

        metric_handles = [
            Line2D([0], [0], color="gray", lw=6, alpha=0.85, label="Balance (solid)"),
            Line2D([0], [0], color="gray", lw=6, alpha=0.45, linestyle="--",
                   label="Additive (hatched)"),
        ]
        ax.legend(handles=LEGEND_ELEMENTS + metric_handles, loc="upper right",
                  fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(f"{compositionality_dir}/all_models_compositionality{suffix}.png", dpi=150)
        plt.close()
        print(f"Saved → {compositionality_dir}/all_models_compositionality{suffix}.png")

        # ── 3b. V1 vs V2 side-by-side bar charts ─────────────────────────────
        if v1_list and v2_list:
            v1_vals = [compo[m]["v1_balance"] or 0 for m in ordered]
            v2_vals = [compo[m]["v2_balance"] or 0 for m in ordered]

            fig, axes = plt.subplots(1, 2, figsize=(fig_w * 1.4, 6), sharey=True)
            for ax, vals, title, suffix in [
                (axes[0], v1_vals,
                 "V1 Compounds\n(linguistically composed, visually ambiguous)", "v1"),
                (axes[1], v2_vals,
                 "V2 Compounds\n(linguistically + visually composed)", "v2"),
            ]:
                bars = ax.bar(ordered, vals, color=colors, alpha=0.85,
                              edgecolor="black", linewidth=1.2)
                for bar, v in zip(bars, vals):
                    ax.annotate(f"{v:.2f}",
                                xy=(bar.get_x() + bar.get_width() / 2, v),
                                xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=label_size, fontweight="bold")
                ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.4)
                ax.axhline(y=0.5, color="orange", linestyle=":", alpha=0.4)
                ax.set_ylim(0, 1.15)
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_ylabel("Avg Balance Score")
                ax.set_xticks(range(len(ordered)))
                ax.set_xticklabels(ordered, rotation=40, ha="right",
                                   fontsize=label_size)
                ax.grid(axis="y", alpha=0.25)

            fig.suptitle("Compositionality: V1 vs V2 Compounds\n"
                         "Key test — does visual grounding change the result?",
                         fontsize=13, fontweight="bold")
            plt.legend(handles=LEGEND_ELEMENTS, loc="upper right", fontsize=8)
            plt.tight_layout()
            plt.savefig(f"{compositionality_dir}/v1_vs_v2_compositionality{suffix}.png",
                        dpi=150)
            plt.close()
            print(f"Saved → {compositionality_dir}/v1_vs_v2_compositionality{suffix}.png")

            # ── 3c. Delta chart: v2_score − v1_score ─────────────────────────
            deltas = [
                (compo[m]["v2_balance"] or 0) - (compo[m]["v1_balance"] or 0)
                for m in ordered
            ]
            delta_colors = ["#27AE60" if d >= 0 else "#E74C3C" for d in deltas]

            fig, ax = plt.subplots(figsize=(fig_w, 5))
            bars = ax.bar(ordered, deltas, color=delta_colors, alpha=0.85,
                          edgecolor="black", linewidth=1.2)
            for bar, d in zip(bars, deltas):
                ypos = d + 0.01 if d >= 0 else d - 0.03
                ax.annotate(f"{d:+.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, ypos),
                            ha="center", fontsize=label_size, fontweight="bold")
            ax.axhline(y=0, color="black", linewidth=1.2)
            ax.set_ylabel("Δ Balance Score  (V2 − V1)")
            ax.set_title("V2 − V1 Compositionality Delta\n"
                         "Positive = model benefits from visual grounding in stimuli",
                         fontsize=12, fontweight="bold")
            ax.set_xticks(range(len(ordered)))
            ax.set_xticklabels(ordered, rotation=40, ha="right",
                               fontsize=label_size)
            ax.grid(axis="y", alpha=0.25)

            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(color="#27AE60", alpha=0.85, label="V2 > V1 (benefits from grounding)"),
                Patch(color="#E74C3C", alpha=0.85, label="V1 > V2 (no benefit)"),
            ], fontsize=9)
            plt.tight_layout()
            plt.savefig(f"{compositionality_dir}/v1_v2_delta{suffix}.png", dpi=150)
            plt.close()
            print(f"Saved → {compositionality_dir}/v1_v2_delta{suffix}.png")

            # Print summary table
            print("\nCompositionality Summary (Balance metric):")
            print(f"{'Model':<22} {'V1':>6} {'V2':>6} {'Δ(V2-V1)':>10} "
                  f"{'All-Bal':>8} {'All-Add':>8}")
            print("-" * 65)
            for m in ordered:
                c = compo[m]
                v1 = c["v1_balance"]  or 0
                v2 = c["v2_balance"]  or 0
                ab = c["all_balance"] or 0
                aa = c["all_additive"] or 0
                print(f"{m:<22} {v1:>6.3f} {v2:>6.3f} {v2 - v1:>+10.3f} "
                      f"{ab:>8.3f} {aa:>8.3f}")

    print("\nSUCCESS: All charts generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize replication results.")
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
        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="selected",
        help="Layer key to visualize: selected|default|last|-1|layer_N.",
    )
    args = parser.parse_args()
    main(args.layer, data_file=args.data_file, output_dir=args.output_dir)
