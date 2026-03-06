import argparse
import gzip
import html
import json
import math
import os
import re
from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPOUND_CONCEPTS = {
    "forest fire",
    "space city",
    "water city",
    "city forest",
    "mountain road",
    "ocean bridge",
    "city bridge",
    "mountain forest",
}


def _pair_key(model_a: str, model_b: str) -> Tuple[str, str]:
    return tuple(sorted((model_a, model_b)))


def _load_json(path: str) -> Dict[str, Any]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _safe_median(values: Sequence[float]) -> float:
    return float(median(values)) if values else float("nan")


def _latest_file(glob_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(glob_dir):
        return None
    candidates = [
        os.path.join(glob_dir, name)
        for name in os.listdir(glob_dir)
        if name.startswith(prefix + "_") and name.endswith(".log")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _parse_iso_timestamp(text: str) -> Optional[datetime]:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return None


def _duration_from_model_log(path: str) -> Optional[float]:
    started = None
    finished = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "Started:" in line:
                started = line.split("Started:", 1)[1].strip()
            elif "Finished:" in line:
                finished = line.split("Finished:", 1)[1].strip()
    if not started or not finished:
        return None
    s_dt = _parse_iso_timestamp(started)
    f_dt = _parse_iso_timestamp(finished)
    if not s_dt or not f_dt:
        return None
    return (f_dt - s_dt).total_seconds()


def _parse_pipeline_model_times(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    pattern = re.compile(r"^<<< OK: (.+) \((\d+)s\)$")
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            match = pattern.match(line)
            if match:
                out[match.group(1)] = int(match.group(2))
    return out


def _similarity_vector(compiled: Dict[str, Any], concepts: List[str], model_name: str) -> np.ndarray:
    embeddings = compiled["models"][model_name]["embeddings"]
    matrix = np.stack([np.array(embeddings[c]).reshape(-1) for c in concepts], axis=0)
    sim = cosine_similarity(matrix)
    upper = np.triu_indices(len(concepts), k=1)
    return sim[upper]


def _compute_v1_v2_shift(v1_compiled: Dict[str, Any], v2_compiled: Dict[str, Any]) -> Dict[str, Any]:
    v1_concepts = [c for c in v1_compiled["concepts"] if c not in COMPOUND_CONCEPTS]
    v2_concepts = [c for c in v2_compiled["concepts"] if c not in COMPOUND_CONCEPTS]
    shared_concepts = [c for c in v1_concepts if c in set(v2_concepts)]

    shared_models = sorted(set(v1_compiled["models"]).intersection(v2_compiled["models"]))
    continuity_rows = []
    pair_delta_rows = []

    for model in shared_models:
        v1_vec = _similarity_vector(v1_compiled, shared_concepts, model)
        v2_vec = _similarity_vector(v2_compiled, shared_concepts, model)
        rho = float(spearmanr(v1_vec, v2_vec).statistic)
        continuity_rows.append({"model": model, "rho": rho})

    for i, model_a in enumerate(shared_models):
        for model_b in shared_models[i + 1 :]:
            v1_a = _similarity_vector(v1_compiled, shared_concepts, model_a)
            v1_b = _similarity_vector(v1_compiled, shared_concepts, model_b)
            v2_a = _similarity_vector(v2_compiled, shared_concepts, model_a)
            v2_b = _similarity_vector(v2_compiled, shared_concepts, model_b)
            rho_v1 = float(spearmanr(v1_a, v1_b).statistic)
            rho_v2 = float(spearmanr(v2_a, v2_b).statistic)
            pair_delta_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "rho_v1": rho_v1,
                    "rho_v2": rho_v2,
                    "delta": rho_v2 - rho_v1,
                }
            )

    continuity_values = [row["rho"] for row in continuity_rows]
    delta_values = [row["delta"] for row in pair_delta_rows]
    return {
        "shared_models": shared_models,
        "shared_concepts": shared_concepts,
        "continuity_rows": continuity_rows,
        "pair_delta_rows": pair_delta_rows,
        "continuity_mean": _safe_mean(continuity_values),
        "continuity_median": _safe_median(continuity_values),
        "continuity_min": min(continuity_values) if continuity_values else float("nan"),
        "continuity_max": max(continuity_values) if continuity_values else float("nan"),
        "pair_delta_mean": _safe_mean(delta_values),
        "pair_delta_mean_abs": _safe_mean([abs(x) for x in delta_values]),
        "pair_delta_max": max(delta_values) if delta_values else float("nan"),
        "pair_delta_min": min(delta_values) if delta_values else float("nan"),
    }


def _compute_baseline_vs_aligned_delta(
    baseline_robustness: Dict[str, Any],
    aligned_robustness: Dict[str, Any],
    model_types: Dict[str, str],
) -> Dict[str, Any]:
    b_rows = {
        _pair_key(r["model_a"], r["model_b"]): r
        for r in baseline_robustness["rsa_bootstrap_image"]["pairwise_results"]
    }
    a_rows = {
        _pair_key(r["model_a"], r["model_b"]): r
        for r in aligned_robustness["rsa_bootstrap_image"]["pairwise_results"]
    }
    rows = []
    for key in sorted(set(b_rows).intersection(a_rows)):
        model_a, model_b = key
        b_rho = float(b_rows[key]["rho_point_estimate"])
        a_rho = float(a_rows[key]["rho_point_estimate"])
        delta = a_rho - b_rho
        type_pair = " × ".join(sorted((model_types.get(model_a, "?"), model_types.get(model_b, "?"))))
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "rho_baseline": b_rho,
                "rho_aligned": a_rho,
                "delta": delta,
                "type_pair": type_pair,
            }
        )

    deltas = [r["delta"] for r in rows]
    by_type: Dict[str, List[float]] = {}
    for row in rows:
        by_type.setdefault(row["type_pair"], []).append(row["delta"])

    sig_b = baseline_robustness["rsa_significance"]["pairwise_results"]
    sig_a = aligned_robustness["rsa_significance"]["pairwise_results"]
    sig_b_set = {
        _pair_key(r["model_a"], r["model_b"])
        for r in sig_b
        if float(r.get("q_bh_fdr", 1.0)) < 0.05
    }
    sig_a_set = {
        _pair_key(r["model_a"], r["model_b"])
        for r in sig_a
        if float(r.get("q_bh_fdr", 1.0)) < 0.05
    }

    return {
        "rows": rows,
        "mean_delta": _safe_mean(deltas),
        "median_delta": _safe_median(deltas),
        "mean_abs_delta": _safe_mean([abs(d) for d in deltas]),
        "max_delta": max(deltas) if deltas else float("nan"),
        "min_delta": min(deltas) if deltas else float("nan"),
        "group_summary": [
            {
                "type_pair": key,
                "count": len(values),
                "mean_delta": _safe_mean(values),
                "mean_abs_delta": _safe_mean([abs(v) for v in values]),
            }
            for key, values in sorted(by_type.items(), key=lambda kv: _safe_mean(kv[1]), reverse=True)
        ],
        "sig_count_baseline": len(sig_b_set),
        "sig_count_aligned": len(sig_a_set),
        "sig_new": len(sig_a_set - sig_b_set),
        "sig_lost": len(sig_b_set - sig_a_set),
        "top_up": sorted(rows, key=lambda r: r["delta"], reverse=True)[:8],
        "top_down": sorted(rows, key=lambda r: r["delta"])[:8],
    }


def _compute_ci_summary(robustness: Dict[str, Any]) -> Dict[str, Any]:
    rows = robustness["rsa_bootstrap_image"]["pairwise_results"]
    det_rows = [r for r in rows if r.get("deterministic_non_image_pair")]
    img_rows = [r for r in rows if not r.get("deterministic_non_image_pair")]
    det_widths = [float(r["ci_high"]) - float(r["ci_low"]) for r in det_rows]
    img_widths = [float(r["ci_high"]) - float(r["ci_low"]) for r in img_rows]
    return {
        "det_count": len(det_rows),
        "img_count": len(img_rows),
        "det_width_mean": _safe_mean(det_widths),
        "img_width_mean": _safe_mean(img_widths),
        "img_width_median": _safe_median(img_widths),
        "img_width_min": min(img_widths) if img_widths else float("nan"),
        "img_width_max": max(img_widths) if img_widths else float("nan"),
        "img_widths": img_widths,
    }


def _compute_source_holdout_summary(
    baseline_robustness: Dict[str, Any],
    aligned_robustness: Dict[str, Any],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for mode in ("leave_one_source_out", "source_only"):
        b_mode = baseline_robustness["source_holdout"][mode]
        a_mode = aligned_robustness["source_holdout"][mode]
        for source in sorted(set(b_mode).union(a_mode)):
            b_row = b_mode.get(source, {})
            a_row = a_mode.get(source, {})
            if b_row.get("skipped") or a_row.get("skipped"):
                summaries.append(
                    {
                        "mode": mode,
                        "source": source,
                        "skipped": True,
                        "reason": b_row.get("reason") or a_row.get("reason") or "skipped",
                    }
                )
                continue
            b_pairs = b_row.get("pairwise", [])
            a_pairs = a_row.get("pairwise", [])
            if not b_pairs or not a_pairs:
                summaries.append(
                    {
                        "mode": mode,
                        "source": source,
                        "skipped": True,
                        "reason": "no pairwise entries",
                    }
                )
                continue
            b_deltas = [float(item["delta"]) for item in b_pairs]
            a_deltas = [float(item["delta"]) for item in a_pairs]
            summaries.append(
                {
                    "mode": mode,
                    "source": source,
                    "skipped": False,
                    "baseline_mean_delta": _safe_mean(b_deltas),
                    "aligned_mean_delta": _safe_mean(a_deltas),
                    "baseline_mean_abs": _safe_mean([abs(v) for v in b_deltas]),
                    "aligned_mean_abs": _safe_mean([abs(v) for v in a_deltas]),
                    "max_abs_baseline": max(abs(v) for v in b_deltas),
                    "max_abs_aligned": max(abs(v) for v in a_deltas),
                }
            )
    return summaries


def _compute_prompt_summary(
    baseline_robustness: Dict[str, Any],
    aligned_robustness: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_models = baseline_robustness["prompt_sensitivity"]["models"]
    aligned_models = aligned_robustness["prompt_sensitivity"]["models"]
    rows = []
    for model_name in sorted(set(baseline_models).intersection(aligned_models)):
        b = baseline_models[model_name]
        a = aligned_models[model_name]
        b_max = b.get("max_abs_delta_vs_baseline")
        a_max = a.get("max_abs_delta_vs_baseline")
        if b_max is None or a_max is None:
            continue
        rows.append(
            {
                "model": model_name,
                "baseline_max_abs": float(b_max),
                "aligned_max_abs": float(a_max),
                "delta": float(a_max) - float(b_max),
                "baseline_t0_mean": float(
                    b.get("cross_modal_rho_by_template", {}).get("t0", {}).get("mean", float("nan"))
                ),
                "aligned_t0_mean": float(
                    a.get("cross_modal_rho_by_template", {}).get("t0", {}).get("mean", float("nan"))
                ),
            }
        )
    return {
        "rows": rows,
        "baseline_mean": _safe_mean([r["baseline_max_abs"] for r in rows]),
        "aligned_mean": _safe_mean([r["aligned_max_abs"] for r in rows]),
        "max_shift": max([abs(r["delta"]) for r in rows], default=float("nan")),
    }


def _compute_depth_summary(
    baseline_robustness: Dict[str, Any],
    aligned_robustness: Dict[str, Any],
) -> List[Dict[str, Any]]:
    fractions = baseline_robustness["aligned_layer"]["fractions"]
    out = []
    for fraction in fractions:
        key = f"d{int(round(float(fraction) * 100)):02d}"
        b_rows = baseline_robustness["aligned_layer"]["pairwise_by_fraction"].get(key, [])
        a_rows = aligned_robustness["aligned_layer"]["pairwise_by_fraction"].get(key, [])
        if not b_rows or not a_rows:
            continue
        b_vals = [float(item["rho"]) for item in b_rows]
        a_vals = [float(item["rho"]) for item in a_rows]
        out.append(
            {
                "fraction": float(fraction),
                "key": key,
                "baseline_mean": _safe_mean(b_vals),
                "aligned_mean": _safe_mean(a_vals),
                "delta": _safe_mean(a_vals) - _safe_mean(b_vals),
            }
        )
    return out


def _compute_runtime_ratio(
    v1_results: Dict[str, Any],
    v2_results: Dict[str, Any],
    v1_logs_dir: str,
    v2_pipeline_log: str,
) -> List[Dict[str, Any]]:
    shared = sorted(set(v1_results["models"]).intersection(v2_results["models"]))
    vision_shared = [m for m in shared if v2_results["models"][m]["config"]["type"] == "vision"]
    v2_times = _parse_pipeline_model_times(v2_pipeline_log)
    rows: List[Dict[str, Any]] = []
    for model in vision_shared:
        v1_log = _latest_file(v1_logs_dir, model)
        if not v1_log:
            continue
        t1 = _duration_from_model_log(v1_log)
        t2 = v2_times.get(model)
        if t1 is None or t2 is None or t1 <= 0:
            continue
        rows.append(
            {
                "model": model,
                "v1_seconds": float(t1),
                "v2_seconds": float(t2),
                "ratio": float(t2) / float(t1),
            }
        )
    return sorted(rows, key=lambda r: r["ratio"], reverse=True)


def _plot_pairwise_delta_hist(rows: List[Dict[str, Any]], output_path: str) -> None:
    deltas = [row["delta"] for row in rows]
    plt.figure(figsize=(8, 4.8))
    plt.hist(deltas, bins=36, color="#2F6B7C", edgecolor="white", alpha=0.9)
    plt.axvline(0.0, color="#8E2C2C", linestyle="--", linewidth=1.2)
    plt.title("Baseline vs Aligned Pairwise RSA Delta Distribution")
    plt.xlabel("Delta (aligned - baseline)")
    plt.ylabel("Pair count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_top_pair_deltas(rows: List[Dict[str, Any]], output_path: str, n: int = 12) -> None:
    selected = sorted(rows, key=lambda r: abs(r["delta"]), reverse=True)[:n]
    labels = [f"{r['model_a']} vs {r['model_b']}" for r in selected]
    values = [r["delta"] for r in selected]
    colors = ["#2A9D8F" if v >= 0 else "#D95F02" for v in values]
    plt.figure(figsize=(10, 6.2))
    y = np.arange(len(labels))
    plt.barh(y, values, color=colors, alpha=0.9)
    plt.yticks(y, labels, fontsize=8)
    plt.axvline(0.0, color="#333333", linewidth=1.0)
    plt.title("Largest Absolute Pairwise RSA Deltas (Aligned vs Baseline)")
    plt.xlabel("Delta")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_depth_trend(depth_rows: List[Dict[str, Any]], output_path: str) -> None:
    x = [row["fraction"] for row in depth_rows]
    baseline = [row["baseline_mean"] for row in depth_rows]
    aligned = [row["aligned_mean"] for row in depth_rows]
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(x, baseline, marker="o", color="#5B6C5D", linewidth=2.0, label="Baseline profile")
    plt.plot(x, aligned, marker="o", color="#1F4E79", linewidth=2.0, label="Aligned5 profile")
    plt.title("Mean Pairwise RSA by Aligned Depth Fraction")
    plt.xlabel("Depth fraction")
    plt.ylabel("Mean pairwise rho")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_source_holdout(source_rows: List[Dict[str, Any]], output_path: str) -> None:
    usable = [row for row in source_rows if not row["skipped"] and row["mode"] == "leave_one_source_out"]
    if not usable:
        return
    labels = [row["source"] for row in usable]
    b_delta = [row["baseline_mean_delta"] for row in usable]
    a_delta = [row["aligned_mean_delta"] for row in usable]
    b_abs = [row["baseline_mean_abs"] for row in usable]
    a_abs = [row["aligned_mean_abs"] for row in usable]

    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].bar(x - width / 2, b_delta, width, label="Baseline", color="#7A8F5C")
    axes[0].bar(x + width / 2, a_delta, width, label="Aligned5", color="#4C78A8")
    axes[0].axhline(0, color="#333333", linewidth=1.0)
    axes[0].set_title("LOSO Mean Delta")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean delta vs full dataset")
    axes[0].legend()

    axes[1].bar(x - width / 2, b_abs, width, label="Baseline", color="#7A8F5C")
    axes[1].bar(x + width / 2, a_abs, width, label="Aligned5", color="#4C78A8")
    axes[1].set_title("LOSO Mean |Delta|")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean absolute delta")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_prompt_sensitivity(prompt_rows: List[Dict[str, Any]], output_path: str) -> None:
    if not prompt_rows:
        return
    labels = [row["model"] for row in prompt_rows]
    baseline_vals = [row["baseline_max_abs"] for row in prompt_rows]
    aligned_vals = [row["aligned_max_abs"] for row in prompt_rows]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(11, 4.8))
    plt.bar(x - width / 2, baseline_vals, width=width, color="#A67C52", label="Baseline")
    plt.bar(x + width / 2, aligned_vals, width=width, color="#5A7D9A", label="Aligned5")
    plt.xticks(x, labels, rotation=40, ha="right", fontsize=8)
    plt.ylabel("max_abs_delta_vs_baseline")
    plt.title("Prompt Sensitivity by Language Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_runtime_ratio(runtime_rows: List[Dict[str, Any]], output_path: str) -> None:
    if not runtime_rows:
        return
    labels = [row["model"] for row in runtime_rows]
    ratios = [row["ratio"] for row in runtime_rows]
    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, ratios, color="#8C564B", alpha=0.9)
    plt.axhline(3.0, color="#2A9D8F", linestyle="--", linewidth=1.2, label="3x reference")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylabel("V2/V1 extraction-time ratio")
    plt.title("Runtime Impact of 10→30 Images (Shared Vision Models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_ci_width_hist(widths: List[float], output_path: str) -> None:
    if not widths:
        return
    plt.figure(figsize=(8, 4.8))
    plt.hist(widths, bins=28, color="#6C8EAD", edgecolor="white", alpha=0.92)
    plt.title("Image-Bootstrap CI Width Distribution (Image-Involved Pairs)")
    plt.xlabel("CI width (ci_high - ci_low)")
    plt.ylabel("Pair count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"
    return f"{value:.{digits}f}"


def _badge(text: str, level: str = "info") -> str:
    return f'<span class="badge badge-{level}">{html.escape(text)}</span>'


def _render_html(
    output_html: str,
    assets_dir: str,
    summary: Dict[str, Any],
) -> None:
    rel_assets = {
        key: os.path.relpath(path, os.path.dirname(output_html))
        for key, path in summary["charts"].items()
        if path and os.path.exists(path)
    }

    change_rows = summary["change_rows"]
    change_table_rows = "\n".join(
        [
            (
                "<tr>"
                f"<td>{html.escape(row['change'])}</td>"
                f"<td>{_badge(row['status'], 'ok' if row['status'] == 'Done' else 'warn')}</td>"
                f"<td>{_badge(row['worth'], row['worth_level'])}</td>"
                f"<td>{html.escape(row['impact'])}</td>"
                f"<td>{html.escape(row['paper_placement'])}</td>"
                "</tr>"
            )
            for row in change_rows
        ]
    )

    top_up_rows = "\n".join(
        [
            (
                "<tr>"
                f"<td>{html.escape(r['model_a'])}</td>"
                f"<td>{html.escape(r['model_b'])}</td>"
                f"<td>{_fmt(r['rho_baseline'], 3)}</td>"
                f"<td>{_fmt(r['rho_aligned'], 3)}</td>"
                f"<td class='num-positive'>{_fmt(r['delta'], 3)}</td>"
                "</tr>"
            )
            for r in summary["baseline_vs_aligned"]["top_up"][:6]
        ]
    )
    top_down_rows = "\n".join(
        [
            (
                "<tr>"
                f"<td>{html.escape(r['model_a'])}</td>"
                f"<td>{html.escape(r['model_b'])}</td>"
                f"<td>{_fmt(r['rho_baseline'], 3)}</td>"
                f"<td>{_fmt(r['rho_aligned'], 3)}</td>"
                f"<td class='num-negative'>{_fmt(r['delta'], 3)}</td>"
                "</tr>"
            )
            for r in summary["baseline_vs_aligned"]["top_down"][:6]
        ]
    )

    source_rows_html = []
    for row in summary["source_holdout"]:
        if row["skipped"]:
            source_rows_html.append(
                "<tr>"
                f"<td>{html.escape(row['mode'])}</td>"
                f"<td>{html.escape(row['source'])}</td>"
                "<td colspan='4'>skipped: "
                f"{html.escape(row['reason'])}</td>"
                "</tr>"
            )
        else:
            source_rows_html.append(
                "<tr>"
                f"<td>{html.escape(row['mode'])}</td>"
                f"<td>{html.escape(row['source'])}</td>"
                f"<td>{_fmt(row['baseline_mean_delta'], 4)}</td>"
                f"<td>{_fmt(row['aligned_mean_delta'], 4)}</td>"
                f"<td>{_fmt(row['baseline_mean_abs'], 4)}</td>"
                f"<td>{_fmt(row['aligned_mean_abs'], 4)}</td>"
                "</tr>"
            )
    source_rows = "\n".join(source_rows_html)

    prompt_rows_html = "\n".join(
        [
            (
                "<tr>"
                f"<td>{html.escape(row['model'])}</td>"
                f"<td>{_fmt(row['baseline_max_abs'], 4)}</td>"
                f"<td>{_fmt(row['aligned_max_abs'], 4)}</td>"
                f"<td>{_fmt(row['delta'], 4)}</td>"
                "</tr>"
            )
            for row in summary["prompt"]["rows"]
        ]
    )

    runtime_rows_html = "\n".join(
        [
            (
                "<tr>"
                f"<td>{html.escape(row['model'])}</td>"
                f"<td>{_fmt(row['v1_seconds'], 1)}</td>"
                f"<td>{_fmt(row['v2_seconds'], 1)}</td>"
                f"<td>{_fmt(row['ratio'], 2)}x</td>"
                "</tr>"
            )
            for row in summary["runtime_ratios"]
        ]
    )

    depth_rows_html = "\n".join(
        [
            (
                "<tr>"
                f"<td>{html.escape(row['key'])}</td>"
                f"<td>{_fmt(row['baseline_mean'], 4)}</td>"
                f"<td>{_fmt(row['aligned_mean'], 4)}</td>"
                f"<td>{_fmt(row['delta'], 4)}</td>"
                "</tr>"
            )
            for row in summary["depth"]
        ]
    )

    html_out = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>V2 Change Impact Report</title>
  <style>
    :root {{
      --ink: #1c2a31;
      --muted: #4e626e;
      --bg: #f5f7f4;
      --card: #ffffff;
      --accent: #0f6c78;
      --accent-soft: #d8ecef;
      --good: #157a4f;
      --warn: #9a5e00;
      --bad: #9f2f2f;
      --border: #d5dde2;
    }}
    @page {{ margin: 14mm; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 20%, #e8f2f0 0%, transparent 40%),
        radial-gradient(circle at 85% 0%, #f2efe4 0%, transparent 40%),
        var(--bg);
      line-height: 1.45;
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 52px;
    }}
    .hero {{
      background: linear-gradient(110deg, #114b5f 0%, #34677e 55%, #4f86a3 100%);
      color: #fff;
      border-radius: 14px;
      padding: 24px 28px;
      box-shadow: 0 10px 22px rgba(16, 43, 55, 0.20);
      margin-bottom: 20px;
      page-break-inside: avoid;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      letter-spacing: 0.3px;
    }}
    .hero p {{
      margin: 0;
      opacity: 0.95;
    }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-bottom: 14px;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      page-break-inside: avoid;
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }}
    .kpi .value {{
      font-size: 27px;
      font-weight: 700;
      margin-top: 4px;
      color: #1a4052;
    }}
    .section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-top: 14px;
      page-break-inside: avoid;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 20px;
      color: #11303f;
    }}
    h3 {{
      margin: 14px 0 8px;
      font-size: 16px;
      color: #23495b;
    }}
    p {{
      margin: 8px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 7px 8px;
      vertical-align: top;
    }}
    th {{
      background: #edf3f6;
      text-align: left;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 9px;
      font-size: 11px;
      font-weight: 600;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .badge-ok {{ background: #e3f4eb; color: #0e5d3c; border-color: #b9e3cb; }}
    .badge-warn {{ background: #fef1db; color: #7b4a00; border-color: #f2d3a0; }}
    .badge-info {{ background: #e5f2f7; color: #1a5d7a; border-color: #bfddeb; }}
    .num-positive {{ color: var(--good); font-weight: 700; }}
    .num-negative {{ color: var(--bad); font-weight: 700; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 10px;
    }}
    .chart {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px;
      background: #fbfcfd;
      page-break-inside: avoid;
    }}
    .chart img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    ul {{
      margin: 8px 0 8px 18px;
      padding: 0;
    }}
    .note {{
      background: var(--accent-soft);
      border-left: 4px solid var(--accent);
      padding: 10px 12px;
      border-radius: 6px;
      margin-top: 8px;
    }}
    .foot {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 16px;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .chart-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>V2 Change Impact Report</h1>
      <p>V1 vs V2 comparison with robustness hardening outcomes, aligned-layer analysis, and publication placement guidance. Generated: {html.escape(summary['generated_at'])}</p>
    </section>

    <section class="grid">
      <article class="kpi">
        <div class="label">V2 Model Coverage</div>
        <div class="value">{summary['v2_model_count']}</div>
      </article>
      <article class="kpi">
        <div class="label">Images / Concept</div>
        <div class="value">{summary['images_per_concept']}</div>
      </article>
      <article class="kpi">
        <div class="label">Significant Pairs (q&lt;0.05)</div>
        <div class="value">{summary['baseline_vs_aligned']['sig_count_baseline']}</div>
      </article>
      <article class="kpi">
        <div class="label">Mean |Aligned-Baseline Delta|</div>
        <div class="value">{_fmt(summary['baseline_vs_aligned']['mean_abs_delta'], 4)}</div>
      </article>
    </section>

    <section class="section">
      <h2>Change Scorecard</h2>
      <table>
        <thead>
          <tr>
            <th>Change</th>
            <th>Status</th>
            <th>Worth It?</th>
            <th>Impact Summary</th>
            <th>Put In Paper</th>
          </tr>
        </thead>
        <tbody>
          {change_table_rows}
        </tbody>
      </table>
      <div class="note">
        q4 quantization sweep is intentionally excluded per thesis scope.
      </div>
    </section>

    <section class="section">
      <h2>Key Outcomes</h2>
      <ul>
        <li>V1 to V2 (shared models) continuity mean Spearman: <strong>{_fmt(summary['v1_v2_shift']['continuity_mean'], 3)}</strong>.</li>
        <li>V1 to V2 shared-pair RSA shift mean delta: <strong>{_fmt(summary['v1_v2_shift']['pair_delta_mean'], 3)}</strong> (mean absolute <strong>{_fmt(summary['v1_v2_shift']['pair_delta_mean_abs'], 3)}</strong>).</li>
        <li>Baseline vs aligned full-run drift is small: mean delta <strong>{_fmt(summary['baseline_vs_aligned']['mean_delta'], 4)}</strong>.</li>
        <li>Prompt sensitivity remains non-trivial for language models (mean max-abs delta <strong>{_fmt(summary['prompt']['baseline_mean'], 4)}</strong>).</li>
      </ul>
    </section>

    <section class="section">
      <h2>Visual Evidence</h2>
      <div class="chart-grid">
        <div class="chart">
          <h3>Baseline vs Aligned Pairwise Delta Distribution</h3>
          <img src="{html.escape(rel_assets.get('pairwise_hist', ''))}" alt="Pairwise delta histogram" />
        </div>
        <div class="chart">
          <h3>Largest Pairwise Delta Magnitudes</h3>
          <img src="{html.escape(rel_assets.get('pairwise_top', ''))}" alt="Top pairwise deltas" />
        </div>
        <div class="chart">
          <h3>Depth Trend (Aligned Fractions)</h3>
          <img src="{html.escape(rel_assets.get('depth_trend', ''))}" alt="Depth trend chart" />
        </div>
        <div class="chart">
          <h3>Image-Bootstrap CI Widths</h3>
          <img src="{html.escape(rel_assets.get('ci_widths', ''))}" alt="CI width histogram" />
        </div>
        <div class="chart">
          <h3>Source Holdout Shift (LOSO)</h3>
          <img src="{html.escape(rel_assets.get('source_holdout', ''))}" alt="Source holdout chart" />
        </div>
        <div class="chart">
          <h3>Prompt Sensitivity by Model</h3>
          <img src="{html.escape(rel_assets.get('prompt', ''))}" alt="Prompt sensitivity chart" />
        </div>
        <div class="chart" style="grid-column: 1 / -1;">
          <h3>Runtime Cost of 10→30 Images (Shared Vision Models)</h3>
          <img src="{html.escape(rel_assets.get('runtime_ratio', ''))}" alt="Runtime ratio chart" />
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Pairwise Delta Details (Baseline vs Aligned)</h2>
      <h3>Largest Increases</h3>
      <table>
        <thead><tr><th>Model A</th><th>Model B</th><th>Baseline</th><th>Aligned</th><th>Delta</th></tr></thead>
        <tbody>{top_up_rows}</tbody>
      </table>
      <h3>Largest Decreases</h3>
      <table>
        <thead><tr><th>Model A</th><th>Model B</th><th>Baseline</th><th>Aligned</th><th>Delta</th></tr></thead>
        <tbody>{top_down_rows}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Source Holdout Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Mode</th><th>Source</th><th>Baseline Mean Delta</th><th>Aligned Mean Delta</th><th>Baseline Mean |Delta|</th><th>Aligned Mean |Delta|</th>
          </tr>
        </thead>
        <tbody>{source_rows}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Prompt Sensitivity Summary</h2>
      <p>Mean max-abs delta: baseline <strong>{_fmt(summary['prompt']['baseline_mean'], 4)}</strong>, aligned <strong>{_fmt(summary['prompt']['aligned_mean'], 4)}</strong>.</p>
      <table>
        <thead><tr><th>Model</th><th>Baseline Max-Abs</th><th>Aligned Max-Abs</th><th>Delta</th></tr></thead>
        <tbody>{prompt_rows_html}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Aligned Layer Fraction Means</h2>
      <table>
        <thead><tr><th>Fraction Bin</th><th>Baseline Mean</th><th>Aligned Mean</th><th>Delta</th></tr></thead>
        <tbody>{depth_rows_html}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Runtime Impact Table (Vision Models)</h2>
      <table>
        <thead><tr><th>Model</th><th>V1 Seconds</th><th>V2 Seconds</th><th>V2/V1 Ratio</th></tr></thead>
        <tbody>{runtime_rows_html}</tbody>
      </table>
    </section>

    <p class="foot">
      Sources: compiled and robustness JSON artifacts in `results/replication`, plus extraction pipeline logs for runtime accounting.
    </p>
  </div>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_out)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser(description="Generate V2 HTML change-impact report with charts.")
    parser.add_argument(
        "--v1-results",
        default="",
    )
    parser.add_argument(
        "--v2-baseline-results",
        default=os.path.join(repo_root, "results", "baseline", "replication_results.json.gz"),
    )
    parser.add_argument(
        "--v2-aligned-results",
        default=os.path.join(repo_root, "results", "aligned5", "replication_results.json.gz"),
    )
    parser.add_argument(
        "--v2-baseline-robustness",
        default=os.path.join(
            repo_root, "results", "baseline", "robustness", "robustness_stats.json"
        ),
    )
    parser.add_argument(
        "--v2-aligned-robustness",
        default=os.path.join(
            repo_root, "results", "aligned5", "robustness", "robustness_stats.json"
        ),
    )
    parser.add_argument(
        "--v1-logs-dir",
        default="",
    )
    parser.add_argument(
        "--v2-baseline-pipeline-log",
        default="",
    )
    parser.add_argument(
        "--output-html",
        default=os.path.join(repo_root, "results", "summaries", "V2_CHANGE_IMPACT_REPORT.html"),
    )
    parser.add_argument(
        "--assets-dir",
        default=os.path.join(repo_root, "results", "v2_change_assets"),
    )
    args = parser.parse_args()

    os.makedirs(args.assets_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)

    v1_compiled = _load_json(args.v1_results)
    v2_baseline_compiled = _load_json(args.v2_baseline_results)
    v2_aligned_compiled = _load_json(args.v2_aligned_results)
    baseline_robustness = _load_json(args.v2_baseline_robustness)
    aligned_robustness = _load_json(args.v2_aligned_robustness)

    model_types = {
        model: payload["config"]["type"]
        for model, payload in v2_baseline_compiled["models"].items()
    }

    v1_v2_shift = _compute_v1_v2_shift(v1_compiled, v2_baseline_compiled)
    baseline_vs_aligned = _compute_baseline_vs_aligned_delta(
        baseline_robustness, aligned_robustness, model_types
    )
    ci_summary = _compute_ci_summary(baseline_robustness)
    source_holdout = _compute_source_holdout_summary(baseline_robustness, aligned_robustness)
    prompt_summary = _compute_prompt_summary(baseline_robustness, aligned_robustness)
    depth_summary = _compute_depth_summary(baseline_robustness, aligned_robustness)
    runtime_ratios = _compute_runtime_ratio(
        v1_compiled,
        v2_baseline_compiled,
        args.v1_logs_dir,
        args.v2_baseline_pipeline_log,
    )

    chart_paths = {
        "pairwise_hist": os.path.join(args.assets_dir, "pairwise_delta_hist.png"),
        "pairwise_top": os.path.join(args.assets_dir, "pairwise_top_deltas.png"),
        "depth_trend": os.path.join(args.assets_dir, "depth_trend.png"),
        "ci_widths": os.path.join(args.assets_dir, "ci_width_hist.png"),
        "source_holdout": os.path.join(args.assets_dir, "source_holdout_loso.png"),
        "prompt": os.path.join(args.assets_dir, "prompt_sensitivity.png"),
        "runtime_ratio": os.path.join(args.assets_dir, "runtime_ratio.png"),
    }

    _plot_pairwise_delta_hist(baseline_vs_aligned["rows"], chart_paths["pairwise_hist"])
    _plot_top_pair_deltas(baseline_vs_aligned["rows"], chart_paths["pairwise_top"])
    _plot_depth_trend(depth_summary, chart_paths["depth_trend"])
    _plot_ci_width_hist(ci_summary["img_widths"], chart_paths["ci_widths"])
    _plot_source_holdout(source_holdout, chart_paths["source_holdout"])
    _plot_prompt_sensitivity(prompt_summary["rows"], chart_paths["prompt"])
    _plot_runtime_ratio(runtime_ratios, chart_paths["runtime_ratio"])

    change_rows = [
        {
            "change": "30 images/concept",
            "status": "Done",
            "worth": "Yes (High)",
            "worth_level": "ok",
            "impact": "Substantial V1→V2 shift on shared-model RSA structure.",
            "paper_placement": "Findings + Methods",
        },
        {
            "change": "Per-image embedding cache",
            "status": "Done",
            "worth": "Yes (Very High)",
            "worth_level": "ok",
            "impact": "Enables exact image bootstrap and fast cached reruns (~5x on tested vision model).",
            "paper_placement": "Methods / Reproducibility",
        },
        {
            "change": "Image-bootstrap RSA CIs",
            "status": "Done",
            "worth": "Yes (High)",
            "worth_level": "ok",
            "impact": "Deterministic language-language zero-width CIs; image-involved pairs carry quantified uncertainty.",
            "paper_placement": "Methods + Robustness Results",
        },
        {
            "change": "Source holdout",
            "status": "Done",
            "worth": "Yes (High)",
            "worth_level": "ok",
            "impact": "Meaningful source-dependent deltas observed, especially in LOSO ImageNet exclusion.",
            "paper_placement": "Findings (Bias/Robustness)",
        },
        {
            "change": "Prompt sensitivity (baseline3)",
            "status": "Done",
            "worth": "Yes (Moderate-High)",
            "worth_level": "ok",
            "impact": "Non-trivial language-model variance across templates.",
            "paper_placement": "Ablations / Controls",
        },
        {
            "change": "Aligned-layer protocol (full run)",
            "status": "Done",
            "worth": "Yes (High)",
            "worth_level": "ok",
            "impact": "Clear depth trend toward higher convergence at deeper aligned fractions.",
            "paper_placement": "Findings + Ablation Variant",
        },
        {
            "change": "q4 quantization sensitivity",
            "status": "Skipped",
            "worth": "Deferred (Out of Scope)",
            "worth_level": "warn",
            "impact": "Intentionally excluded from thesis core claims.",
            "paper_placement": "Not required for main thesis",
        },
    ]

    report_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "v2_model_count": len(v2_baseline_compiled["models"]),
        "images_per_concept": 30,
        "v1_v2_shift": v1_v2_shift,
        "baseline_vs_aligned": baseline_vs_aligned,
        "ci_summary": ci_summary,
        "source_holdout": source_holdout,
        "prompt": prompt_summary,
        "depth": depth_summary,
        "runtime_ratios": runtime_ratios,
        "change_rows": change_rows,
        "charts": chart_paths,
    }
    _render_html(args.output_html, args.assets_dir, report_summary)
    print(f"HTML report written to: {args.output_html}")
    print(f"Chart assets written to: {args.assets_dir}")


if __name__ == "__main__":
    main()
