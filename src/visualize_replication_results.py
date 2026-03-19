import argparse
import os
import json
import re
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, spearmanr
from matplotlib.lines import Line2D
from time import perf_counter

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
DEFAULT_BOOTSTRAP_DRAWS = 1000
DEFAULT_BOOTSTRAP_BATCH_SIZE = 64


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_1d(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def stack_embeddings(embeddings: Dict[str, Any], concepts) -> np.ndarray:
    return np.stack([np.asarray(embeddings[concept], dtype=np.float32) for concept in concepts], axis=0)


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


def spearman_correlation_matrix(rows: np.ndarray) -> np.ndarray:
    ranked = np.empty(rows.shape, dtype=np.float32)
    for idx, row in enumerate(rows):
        ranked[idx] = rankdata(row, method="average").astype(np.float32)
    ranked -= ranked.mean(axis=1, keepdims=True)
    ranked = _row_normalize(ranked)
    corr = ranked @ ranked.T
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr.astype(np.float32, copy=False)


def build_bootstrap_index_batches(
    n_items: int,
    n_bootstrap: int,
    batch_size: int,
    seed: int = 42,
):
    if n_bootstrap <= 0:
        return []
    rng = np.random.default_rng(seed)
    batches = []
    remaining = n_bootstrap
    while remaining > 0:
        size = min(batch_size, remaining)
        batches.append(rng.integers(0, n_items, size=(size, n_items), dtype=np.int32))
        remaining -= size
    return batches


def bootstrap_rsa_ci(
    flat_a: np.ndarray,
    flat_b: np.ndarray,
    index_batches,
    ci: float = 0.95,
):
    """Bootstrap 95% CI for Spearman ρ by resampling concept pairs."""
    if not index_batches:
        rho, _ = spearmanr(flat_a, flat_b)
        return float(rho), float(rho)

    draws = []
    for idx in index_batches:
        sample_a = np.take(flat_a, idx)
        sample_b = np.take(flat_b, idx)
        ranked_a = rankdata(sample_a, axis=1, method="average").astype(np.float32)
        ranked_b = rankdata(sample_b, axis=1, method="average").astype(np.float32)
        ranked_a -= ranked_a.mean(axis=1, keepdims=True)
        ranked_b -= ranked_b.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(ranked_a, axis=1) * np.linalg.norm(ranked_b, axis=1)
        numer = np.sum(ranked_a * ranked_b, axis=1)
        batch = np.divide(
            numer,
            denom,
            out=np.full(idx.shape[0], np.nan, dtype=np.float32),
            where=denom > 0.0,
        )
        draws.append(batch)

    rhos = np.concatenate(draws, axis=0)
    alpha = (1 - ci) / 2
    lo, hi = np.nanquantile(rhos, [alpha, 1 - alpha])
    return float(lo), float(hi)


def balance_score(emb_compound, emb1, emb2):
    """Balance compositionality: 1 = perfectly equidistant, 0 = fully biased."""
    s1 = cosine_similarity_1d(emb_compound, emb1)
    s2 = cosine_similarity_1d(emb_compound, emb2)
    total = s1 + s2
    ratio = s2 / total if total > 0 else 0.5
    return 1.0 - abs(0.5 - ratio) * 2


def additive_score(emb_compound, emb1, emb2):
    """Additive compositionality: cosine sim between compound and sum of parts."""
    v1 = np.asarray(emb1, dtype=np.float32).flatten()
    v2 = np.asarray(emb2, dtype=np.float32).flatten()
    vc = np.asarray(emb_compound, dtype=np.float32).flatten()
    v_sum = v1 + v2
    return cosine_similarity_1d(vc, v_sum)


def compute_compo_scores(embeddings, compound_list):
    """Return (balance_scores, additive_scores) lists for a compound list."""
    bal, add = [], []
    for comp1, comp2, compound in compound_list:
        if compound not in embeddings:
            continue
        e1 = np.asarray(embeddings[comp1], dtype=np.float32)
        e2 = np.asarray(embeddings[comp2], dtype=np.float32)
        ec = np.asarray(embeddings[compound], dtype=np.float32)
        bal.append(balance_score(ec, e1, e2))
        add.append(additive_score(ec, e1, e2))
    return bal, add


def model_category(mtype):
    if mtype == "causal":
        return "causal"
    if mtype == "vision_language":
        return "vision_language"
    if mtype == "vision_language_autoregressive":
        return "vision_language_autoregressive"
    return "vision"


def category_color(cat):
    return {
        "causal": "#E74C3C",
        "vision_language": "#3498DB",
        "vision_language_autoregressive": "#16A085",
        "vision": "#9B59B6",
    }[cat]


def ordered_models_and_colors(models_data):
    cats = {
        "causal": [],
        "vision_language": [],
        "vision_language_autoregressive": [],
        "vision": [],
    }
    for m, mdata in models_data.items():
        cats[model_category(mdata["config"]["type"])].append(m)
    ordered = (
        cats["causal"]
        + cats["vision_language"]
        + cats["vision_language_autoregressive"]
        + cats["vision"]
    )
    colors = (
        [category_color("causal")] * len(cats["causal"])
        + [category_color("vision_language")] * len(cats["vision_language"])
        + [category_color("vision_language_autoregressive")] * len(cats["vision_language_autoregressive"])
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
    Line2D([0], [0], color="#3498DB", lw=4, label="Vision-Language (Contrastive)"),
    Line2D([0], [0], color="#16A085", lw=4, label="Vision-Language (Autoregressive)"),
    Line2D([0], [0], color="#9B59B6", lw=4, label="Vision Models (SSL)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    layer: str,
    data_file: str,
    output_dir: str,
    bootstrap_draws: int,
    bootstrap_batch_size: int,
) -> None:
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
    print(
        f"Bootstrap config: draws={bootstrap_draws}, "
        f"batch_size={bootstrap_batch_size}"
    )

    # ── 1. Similarity matrices (base concepts only) ───────────────────────────
    print("Computing similarity matrices (base concepts only)...")
    sim_matrices = {}
    sim_start = perf_counter()
    for model_name, mdata in models_data.items():
        stacked = stack_embeddings(mdata["embeddings"], base_concepts)
        sim_matrices[model_name] = build_similarity_matrix_from_stacked(stacked)
    print(f"Built {len(sim_matrices)} similarity matrices in {perf_counter() - sim_start:.1f}s")

    # ── 2. RSA matrix with bootstrap CIs ─────────────────────────────────────
    n_pairs = len(base_concepts) * (len(base_concepts) - 1) // 2
    print(f"Computing RSA ({len(base_concepts)} base concepts, {n_pairs} pairs)...")
    triu_idx = np.triu_indices(len(base_concepts), k=1)
    flat_sims = {m: sim_matrices[m][triu_idx].astype(np.float32, copy=False) for m in model_names}
    flat_rows = np.stack([flat_sims[m] for m in model_names], axis=0)
    rsa_matrix = spearman_correlation_matrix(flat_rows)
    rsa_ci_lo = np.eye(len(model_names), dtype=np.float32)
    rsa_ci_hi = np.eye(len(model_names), dtype=np.float32)
    index_batches = build_bootstrap_index_batches(
        flat_rows.shape[1],
        bootstrap_draws,
        bootstrap_batch_size,
    )
    pair_total = len(model_names) * (len(model_names) - 1) // 2
    rsa_start = perf_counter()
    pair_idx = 0
    for i, m1 in enumerate(model_names):
        for j in range(i + 1, len(model_names)):
            m2 = model_names[j]
            lo, hi = bootstrap_rsa_ci(flat_sims[m1], flat_sims[m2], index_batches)
            rsa_ci_lo[i, j] = rsa_ci_lo[j, i] = lo
            rsa_ci_hi[i, j] = rsa_ci_hi[j, i] = hi
            pair_idx += 1
            if pair_idx % 25 == 0 or pair_idx == pair_total:
                print(
                    f"  bootstrap progress: {pair_idx}/{pair_total} pairs "
                    f"({perf_counter() - rsa_start:.1f}s)"
                )
    print(f"Completed RSA stage in {perf_counter() - rsa_start:.1f}s")

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
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=DEFAULT_BOOTSTRAP_DRAWS,
        help="Bootstrap draws for RSA confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=DEFAULT_BOOTSTRAP_BATCH_SIZE,
        help="Batch size for vectorized RSA bootstrap resampling.",
    )
    args = parser.parse_args()
    main(
        args.layer,
        data_file=args.data_file,
        output_dir=args.output_dir,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_batch_size=args.bootstrap_batch_size,
    )
