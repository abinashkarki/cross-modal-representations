import argparse
import gzip
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy.stats import rankdata


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_COMPILED_PATH = os.path.join(REPO_ROOT, "results", "baseline", "replication_results.json.gz")
DEFAULT_ROBUSTNESS_PATH = os.path.join(
    REPO_ROOT, "results", "baseline", "robustness", "robustness_stats.json"
)
DEFAULT_OUTPUT_JSON = os.path.join(
    REPO_ROOT, "results", "summaries", "SCALEUP_PILOT_2026-03-13.json"
)
DEFAULT_OUTPUT_MD = os.path.join(
    REPO_ROOT, "results", "summaries", "SCALEUP_PILOT_2026-03-13.md"
)

KNOWN_COMPOUNDS = {
    "forest fire",
    "space city",
    "water city",
    "city forest",
    "mountain road",
    "ocean bridge",
    "city bridge",
    "mountain forest",
}


def load_json(path: str) -> Dict[str, Any]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, path)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def upper_triangle_flat(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def build_similarity_matrix(embeddings: Dict[str, Iterable[float]], concepts: List[str]) -> np.ndarray:
    stacked = np.stack([np.asarray(embeddings[c], dtype=np.float32) for c in concepts], axis=0)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = stacked / norms
    return normalized @ normalized.T


def corrcoef_from_ranks(rank_matrix: np.ndarray) -> np.ndarray:
    centered = rank_matrix - rank_matrix.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(centered, axis=1, keepdims=True)
    zero_mask = denom.squeeze(axis=1) == 0
    denom = np.where(denom == 0, 1.0, denom)
    normalized = centered / denom
    corr = normalized @ normalized.T
    corr = np.clip(corr, -1.0, 1.0)
    corr[zero_mask, :] = 0.0
    corr[:, zero_mask] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float32)))


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def model_family(model_info: Dict[str, Any]) -> str:
    mtype = model_info.get("config", {}).get("type")
    if mtype == "causal":
        return "language"
    if mtype == "vision":
        return "vision"
    if mtype == "vision_language":
        return "vlm"
    return "other"


def pair_family(family_a: str, family_b: str) -> str:
    pair = tuple(sorted((family_a, family_b)))
    mapping = {
        ("language", "language"): "language-language",
        ("language", "vision"): "language-vision",
        ("language", "vlm"): "language-vlm",
        ("vision", "vision"): "vision-vision",
        ("vision", "vlm"): "vision-vlm",
        ("vlm", "vlm"): "vlm-vlm",
    }
    return mapping.get(pair, "-".join(pair))


def l2_normalized_average(vectors: List[np.ndarray]) -> np.ndarray:
    normalized = []
    for vector in vectors:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        normalized.append(arr)
    return np.mean(np.stack(normalized, axis=0), axis=0).astype(np.float32)


def resolve_template_embeddings(
    model_info: Dict[str, Any],
) -> Dict[str, Dict[str, np.ndarray]]:
    by_template = model_info.get("text_template_embeddings_by_layer", {})
    layer_default = model_info.get("layer_metadata", {}).get("default_layer_key")
    resolved: Dict[str, Dict[str, np.ndarray]] = {}
    for template_key, layered in by_template.items():
        chosen = None
        if layer_default and layer_default in layered:
            chosen = layered[layer_default]
        elif layered:
            chosen = layered[next(iter(layered.keys()))]
        if chosen is None:
            continue
        resolved[template_key] = {
            concept: np.asarray(vector, dtype=np.float32)
            for concept, vector in chosen.items()
        }
    return resolved


def family_medians_from_corr(
    corr: np.ndarray,
    pair_rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in pair_rows:
        grouped[row["family"]].append(float(corr[row["i"], row["j"]]))
    return {family: median(values) for family, values in grouped.items()}


def run_concept_subsampling_pilot(
    similarity_matrices: Dict[str, np.ndarray],
    pair_rows: List[Dict[str, Any]],
    concept_counts: List[int],
    draws: int,
    seed: int,
) -> Dict[str, Any]:
    model_names = sorted(similarity_matrices.keys())
    full_rank_matrix = np.stack(
        [
            rankdata(upper_triangle_flat(similarity_matrices[name]), method="average")
            for name in model_names
        ],
        axis=0,
    ).astype(np.float32)
    full_corr = corrcoef_from_ranks(full_rank_matrix)
    full_family_medians = family_medians_from_corr(full_corr, pair_rows)

    n_concepts = next(iter(similarity_matrices.values())).shape[0]
    rng = np.random.default_rng(seed)
    summaries: Dict[str, Any] = {}

    for k in concept_counts:
        pairwise_abs_deltas: List[float] = []
        family_pairwise_abs_deltas: Dict[str, List[float]] = defaultdict(list)
        family_median_abs_deltas: Dict[str, List[float]] = defaultdict(list)
        claim_counts = Counter()

        for _ in range(draws):
            subset_idx = np.sort(rng.choice(n_concepts, size=k, replace=False))
            subset_rank_matrix = np.stack(
                [
                    rankdata(
                        upper_triangle_flat(similarity_matrices[name][np.ix_(subset_idx, subset_idx)]),
                        method="average",
                    )
                    for name in model_names
                ],
                axis=0,
            ).astype(np.float32)
            subset_corr = corrcoef_from_ranks(subset_rank_matrix)
            subset_family_medians = family_medians_from_corr(subset_corr, pair_rows)

            for row in pair_rows:
                rho_full = float(full_corr[row["i"], row["j"]])
                rho_subset = float(subset_corr[row["i"], row["j"]])
                delta = abs(rho_subset - rho_full)
                pairwise_abs_deltas.append(delta)
                family_pairwise_abs_deltas[row["family"]].append(delta)

            for family, full_median in full_family_medians.items():
                delta = abs(subset_family_medians[family] - full_median)
                family_median_abs_deltas[family].append(delta)

            if subset_family_medians["language-language"] > subset_family_medians["language-vision"]:
                claim_counts["ll_gt_lv"] += 1
            if subset_family_medians["vision-vision"] > subset_family_medians["language-vision"]:
                claim_counts["vv_gt_lv"] += 1
            if subset_family_medians["vision-vlm"] > subset_family_medians["language-vlm"]:
                claim_counts["visionvlm_gt_langvlm"] += 1

        summaries[str(k)] = {
            "draws": draws,
            "pairwise_abs_delta_mean": mean(pairwise_abs_deltas),
            "pairwise_abs_delta_median": median(pairwise_abs_deltas),
            "pairwise_abs_delta_p95": quantile(pairwise_abs_deltas, 0.95),
            "family_pairwise_abs_delta_mean": {
                family: mean(values)
                for family, values in sorted(family_pairwise_abs_deltas.items())
            },
            "family_median_abs_delta_mean": {
                family: mean(values)
                for family, values in sorted(family_median_abs_deltas.items())
            },
            "claim_retention_rate": {
                "language_language_gt_language_vision": claim_counts["ll_gt_lv"] / draws,
                "vision_vision_gt_language_vision": claim_counts["vv_gt_lv"] / draws,
                "vision_vlm_gt_language_vlm": claim_counts["visionvlm_gt_langvlm"] / draws,
            },
        }

    return {
        "full_family_medians": full_family_medians,
        "counts": summaries,
    }


def summarize_image_bootstrap_pilot(
    robustness: Dict[str, Any],
    families_by_model: Dict[str, str],
) -> Dict[str, Any]:
    pairwise_rows = robustness.get("rsa_bootstrap_image", {}).get("pairwise_results", [])
    image_rows = []
    for row in pairwise_rows:
        if row.get("deterministic_non_image_pair"):
            continue
        family = pair_family(families_by_model[row["model_a"]], families_by_model[row["model_b"]])
        point = float(row["rho_point_estimate"])
        rho_mean = float(row["rho_mean"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        image_rows.append(
            {
                "family": family,
                "abs_delta_mean_vs_point": abs(rho_mean - point),
                "ci_width": ci_high - ci_low,
                "point_inside_ci": ci_low <= point <= ci_high,
            }
        )

    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        by_family[row["family"]].append(row)

    return {
        "sample_size_per_concept": robustness.get("metadata", {}).get("bootstrap_sample_size"),
        "draws": robustness.get("metadata", {}).get("bootstrap_draws"),
        "image_involved_pair_count": len(image_rows),
        "overall": {
            "abs_delta_mean_vs_point_mean": mean([r["abs_delta_mean_vs_point"] for r in image_rows]),
            "abs_delta_mean_vs_point_p95": quantile(
                [r["abs_delta_mean_vs_point"] for r in image_rows], 0.95
            ),
            "ci_width_mean": mean([r["ci_width"] for r in image_rows]),
            "ci_width_median": median([r["ci_width"] for r in image_rows]),
            "ci_width_p95": quantile([r["ci_width"] for r in image_rows], 0.95),
            "point_inside_ci_rate": mean([1.0 if r["point_inside_ci"] else 0.0 for r in image_rows]),
        },
        "by_family": {
            family: {
                "pair_count": len(rows),
                "abs_delta_mean_vs_point_mean": mean(
                    [r["abs_delta_mean_vs_point"] for r in rows]
                ),
                "ci_width_mean": mean([r["ci_width"] for r in rows]),
                "point_inside_ci_rate": mean(
                    [1.0 if r["point_inside_ci"] else 0.0 for r in rows]
                ),
            }
            for family, rows in sorted(by_family.items())
        },
    }


def run_prompt_analysis(
    compiled: Dict[str, Any],
    base_concepts: List[str],
) -> Dict[str, Any]:
    models = compiled["models"]
    families_by_model = {name: model_family(info) for name, info in models.items()}
    language_models = sorted([name for name, fam in families_by_model.items() if fam == "language"])
    vision_models = sorted([name for name, fam in families_by_model.items() if fam == "vision"])
    vlm_models = sorted([name for name, fam in families_by_model.items() if fam == "vlm"])
    image_models = vision_models + vlm_models

    image_flats = {
        name: upper_triangle_flat(build_similarity_matrix(models[name]["embeddings"], base_concepts))
        for name in image_models
    }

    language_flats_by_label: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for model_name in language_models:
        template_embeddings = resolve_template_embeddings(models[model_name])
        for template_key, emb in template_embeddings.items():
            language_flats_by_label[template_key][model_name] = upper_triangle_flat(
                build_similarity_matrix(emb, base_concepts)
            )

        if template_embeddings:
            ensemble_embeddings = {}
            for concept in base_concepts:
                vectors = [template_embeddings[key][concept] for key in sorted(template_embeddings)]
                ensemble_embeddings[concept] = l2_normalized_average(vectors)
            language_flats_by_label["ensemble3"][model_name] = upper_triangle_flat(
                build_similarity_matrix(ensemble_embeddings, base_concepts)
            )

    all_labels = ["t0", "t1", "t2", "ensemble3"]
    label_summary: Dict[str, Any] = {}
    per_model_cross_modal: Dict[str, Dict[str, float]] = {}
    per_model_same_geom: Dict[str, Dict[str, float]] = {}
    wins = Counter()

    for model_name in language_models:
        per_model_cross_modal[model_name] = {}
        per_model_same_geom[model_name] = {}
        baseline_vec = language_flats_by_label["t0"][model_name]
        for label in all_labels:
            vec = language_flats_by_label[label][model_name]
            all_image_rhos = [
                float(np.corrcoef(rankdata(vec, method="average"), rankdata(image_flats[img], method="average"))[0, 1])
                for img in image_models
            ]
            per_model_cross_modal[model_name][label] = mean(all_image_rhos)
            if label != "t0":
                same_rho = float(
                    np.corrcoef(
                        rankdata(baseline_vec, method="average"),
                        rankdata(vec, method="average"),
                    )[0, 1]
                )
                per_model_same_geom[model_name][f"t0_vs_{label}"] = same_rho

        best_label = max(
            all_labels,
            key=lambda label: per_model_cross_modal[model_name][label],
        )
        wins[best_label] += 1

    geometry_agreement_summary = {}
    for key in ["t0_vs_t1", "t0_vs_t2", "t0_vs_ensemble3"]:
        values = [per_model_same_geom[model_name][key] for model_name in language_models]
        geometry_agreement_summary[key] = {
            "mean": mean(values),
            "median": median(values),
        }

    for label in all_labels:
        ll = []
        lv = []
        l_vlm = []
        l_all_image = []

        for model_a, model_b in combinations(language_models, 2):
            vec_a = language_flats_by_label[label][model_a]
            vec_b = language_flats_by_label[label][model_b]
            ll.append(
                float(
                    np.corrcoef(
                        rankdata(vec_a, method="average"),
                        rankdata(vec_b, method="average"),
                    )[0, 1]
                )
            )

        for model_a in language_models:
            vec_a = language_flats_by_label[label][model_a]
            rank_a = rankdata(vec_a, method="average")
            for model_b in vision_models:
                rho = float(np.corrcoef(rank_a, rankdata(image_flats[model_b], method="average"))[0, 1])
                lv.append(rho)
                l_all_image.append(rho)
            for model_b in vlm_models:
                rho = float(np.corrcoef(rank_a, rankdata(image_flats[model_b], method="average"))[0, 1])
                l_vlm.append(rho)
                l_all_image.append(rho)

        label_summary[label] = {
            "language_language_median": median(ll),
            "language_vision_median": median(lv),
            "language_vlm_median": median(l_vlm),
            "language_all_image_mean": mean(l_all_image),
            "language_all_image_median": median(l_all_image),
        }

    delta_vs_t0: Dict[str, Any] = {}
    for label in ["t1", "t2", "ensemble3"]:
        deltas = []
        for model_name in language_models:
            deltas.append(
                per_model_cross_modal[model_name][label] - per_model_cross_modal[model_name]["t0"]
            )
        delta_vs_t0[label] = {
            "per_language_model_mean_delta": {
                model_name: float(per_model_cross_modal[model_name][label] - per_model_cross_modal[model_name]["t0"])
                for model_name in language_models
            },
            "mean_delta": mean(deltas),
            "median_delta": median(deltas),
        }

    return {
        "labels": label_summary,
        "per_model_cross_modal_mean": per_model_cross_modal,
        "per_model_geometry_agreement": per_model_same_geom,
        "geometry_agreement_summary": geometry_agreement_summary,
        "wins_by_label": {label: int(wins.get(label, 0)) for label in all_labels},
        "delta_vs_t0": delta_vs_t0,
    }


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_markdown_report(
    summary: Dict[str, Any],
    concept_counts: List[int],
) -> str:
    concept_pilot = summary["concept_subsampling_pilot"]
    image_pilot = summary["image_bootstrap_pilot"]
    prompt = summary["prompt_analysis"]
    prompt_existing = summary["existing_prompt_sensitivity"]

    lines = [
        "# Scale-Up Pilot (2026-03-13)",
        "",
        "This report uses the released baseline artifacts to estimate budget-sensitive drift before a 250-concept expansion.",
        "",
        "Scope:",
        "- Baseline compiled artifact: legacy compressed bundle externalized from the public repo",
        f"- Baseline robustness artifact: `results/baseline/robustness/robustness_stats.json`",
        f"- Base RSA concepts: {summary['base_concept_count']}",
        f"- Models: {summary['model_count']}",
        "",
        "Important limit:",
        "- The checked-in workspace does not include `results/baseline/cache/`, so no fresh image-cache resampling was rerun here.",
        f"- The image pilot below is recovered from the existing cached-bootstrap summary at {image_pilot['sample_size_per_concept']} images/concept and {image_pilot['draws']} draws.",
        "",
        "## 1. Concept-Count Pilot",
        "",
        "Reference full-run family medians on the current 20 base concepts:",
    ]

    for family, value in sorted(concept_pilot["full_family_medians"].items()):
        lines.append(f"- `{family}`: {format_float(value)}")

    lines.extend(
        [
            "",
            "| Concept count | Mean |Δρ| vs full | P95 |Δρ| vs full | Retain LL > LV | Retain VV > LV | Retain V-VLM > L-VLM |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for count in concept_counts:
        row = concept_pilot["counts"][str(count)]
        claims = row["claim_retention_rate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(count),
                    format_float(row["pairwise_abs_delta_mean"]),
                    format_float(row["pairwise_abs_delta_p95"]),
                    format_float(claims["language_language_gt_language_vision"]),
                    format_float(claims["vision_vision_gt_language_vision"]),
                    format_float(claims["vision_vlm_gt_language_vlm"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Mean absolute drift in family medians by concept count:",
            "",
            "| Concept count | LL | LV | L-VLM | VV | V-VLM | VLM-VLM |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for count in concept_counts:
        fam = concept_pilot["counts"][str(count)]["family_median_abs_delta_mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(count),
                    format_float(fam["language-language"]),
                    format_float(fam["language-vision"]),
                    format_float(fam["language-vlm"]),
                    format_float(fam["vision-vision"]),
                    format_float(fam["vision-vlm"]),
                    format_float(fam["vlm-vlm"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 2. Image-Count Pilot (Recovered From Existing 10-Image Bootstrap)",
            "",
            f"- Image-involved model pairs: {image_pilot['image_involved_pair_count']}",
            f"- Mean |bootstrap-mean ρ - full 30-image ρ|: {format_float(image_pilot['overall']['abs_delta_mean_vs_point_mean'])}",
            f"- P95 |bootstrap-mean ρ - full 30-image ρ|: {format_float(image_pilot['overall']['abs_delta_mean_vs_point_p95'])}",
            f"- Mean 95% CI width: {format_float(image_pilot['overall']['ci_width_mean'])}",
            f"- Median 95% CI width: {format_float(image_pilot['overall']['ci_width_median'])}",
            f"- P95 95% CI width: {format_float(image_pilot['overall']['ci_width_p95'])}",
            "",
            "| Family | Pairs | Mean |Δρ| | Mean CI width | Point estimate inside CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family, row in sorted(image_pilot["by_family"].items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    family,
                    str(row["pair_count"]),
                    format_float(row["abs_delta_mean_vs_point_mean"]),
                    format_float(row["ci_width_mean"]),
                    format_float(row["point_inside_ci_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 3. Prompt Study",
            "",
            f"- Existing robustness summary mean max-absolute template delta vs baseline `t0`: {format_float(prompt_existing['mean_max_abs_delta'])}",
            f"- Existing robustness summary median max-absolute template delta vs baseline `t0`: {format_float(prompt_existing['median_max_abs_delta'])}",
            f"- Existing robustness summary max model-level delta: {format_float(prompt_existing['max_max_abs_delta'])}",
            "",
            "Current prompt set:",
            "- `t0`: `The concept of {concept}`",
            "- `t1`: `An example of {concept}`",
            "- `t2`: `The meaning of {concept}`",
            "- `ensemble3`: L2-normalized average of `t0/t1/t2` embeddings per concept",
            "",
            "| Representation | LL median | LV median | L-VLM median | L-all-image mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in ["t0", "t1", "t2", "ensemble3"]:
        row = prompt["labels"][label]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    format_float(row["language_language_median"]),
                    format_float(row["language_vision_median"]),
                    format_float(row["language_vlm_median"]),
                    format_float(row["language_all_image_mean"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Mean per-language-model change in cross-modal agreement vs `t0`:",
        ]
    )
    for label in ["t1", "t2", "ensemble3"]:
        row = prompt["delta_vs_t0"][label]
        lines.append(
            f"- `{label}` vs `t0`: mean delta {format_float(row['mean_delta'])}, median delta {format_float(row['median_delta'])}"
        )

    lines.extend(
        [
            "",
            "Median same-model geometry agreement with `t0`:",
        ]
    )
    for key, row in prompt["geometry_agreement_summary"].items():
        lines.append(
            f"- `{key}`: mean {format_float(row['mean'])}, median {format_float(row['median'])}"
        )

    lines.extend(
        [
            "",
            "Wins by representation on per-language-model mean cross-modal RSA:",
        ]
    )
    for label in ["t0", "t1", "t2", "ensemble3"]:
        wins = prompt["wins_by_label"][label]
        lines.append(f"- `{label}`: {wins} / 8 models")

    lines.extend(
        [
            "",
            "## 4. Decision",
            "",
            "- Concept reductions to 8-10 concepts induce large pairwise drift; 12 is workable but still noisy; 15-18 is much closer to the full 20-concept reference.",
            "- The recovered 10-image pilot shows that image thinning is feasible only with explicit uncertainty reporting. A 10-image protocol is a real approximation to the 30-image run, not an interchangeable substitute.",
            "- Prompt variance is already too large to leave language extraction on a single untuned prompt.",
            "- No single current template dominates across models. The scientifically cleaner move is a preregistered prompt-family protocol: either report the full prompt range as a fixed factor or use a fixed ensemble such as `ensemble3` to avoid post hoc prompt selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a scale-up pilot from the released baseline artifacts."
    )
    parser.add_argument("--compiled-path", default=DEFAULT_COMPILED_PATH)
    parser.add_argument("--robustness-path", default=DEFAULT_ROBUSTNESS_PATH)
    parser.add_argument("--concept-counts", default="8,10,12,15,18")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    compiled = load_json(os.path.abspath(args.compiled_path))
    robustness = load_json(os.path.abspath(args.robustness_path))

    concept_counts = [int(part.strip()) for part in args.concept_counts.split(",") if part.strip()]
    base_concepts = [c for c in compiled["concepts"] if c not in KNOWN_COMPOUNDS]
    models = compiled["models"]
    families_by_model = {name: model_family(info) for name, info in models.items()}

    similarity_matrices = {
        name: build_similarity_matrix(info["embeddings"], base_concepts)
        for name, info in models.items()
    }
    model_names = sorted(models.keys())
    pair_rows = []
    for i, model_a in enumerate(model_names):
        for j in range(i + 1, len(model_names)):
            model_b = model_names[j]
            pair_rows.append(
                {
                    "i": i,
                    "j": j,
                    "model_a": model_a,
                    "model_b": model_b,
                    "family": pair_family(families_by_model[model_a], families_by_model[model_b]),
                }
            )

    prompt_payload = robustness.get("prompt_sensitivity", {}).get("models", {})
    existing_prompt_deltas = [
        float(row["max_abs_delta_vs_baseline"]) for row in prompt_payload.values()
    ]

    summary = {
        "compiled_path": os.path.relpath(os.path.abspath(args.compiled_path), REPO_ROOT),
        "robustness_path": os.path.relpath(os.path.abspath(args.robustness_path), REPO_ROOT),
        "base_concept_count": len(base_concepts),
        "model_count": len(models),
        "concept_subsampling_pilot": run_concept_subsampling_pilot(
            similarity_matrices=similarity_matrices,
            pair_rows=pair_rows,
            concept_counts=concept_counts,
            draws=args.draws,
            seed=args.seed,
        ),
        "image_bootstrap_pilot": summarize_image_bootstrap_pilot(
            robustness=robustness,
            families_by_model=families_by_model,
        ),
        "existing_prompt_sensitivity": {
            "mean_max_abs_delta": mean(existing_prompt_deltas),
            "median_max_abs_delta": median(existing_prompt_deltas),
            "max_max_abs_delta": max(existing_prompt_deltas) if existing_prompt_deltas else 0.0,
        },
        "prompt_analysis": run_prompt_analysis(
            compiled=compiled,
            base_concepts=base_concepts,
        ),
    }

    report = render_markdown_report(summary, concept_counts)
    atomic_write_json(os.path.abspath(args.output_json), summary)
    atomic_write_text(os.path.abspath(args.output_md), report)

    print(f"Wrote JSON summary -> {os.path.abspath(args.output_json)}")
    print(f"Wrote Markdown report -> {os.path.abspath(args.output_md)}")


if __name__ == "__main__":
    main()
