#!/usr/bin/env python3
"""Create a publication figure for ImageNet leave-one-source-out impact."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_REPLICATION = DEFAULT_RESULTS / "baseline" / "replication_results.json.gz"
DEFAULT_ROBUSTNESS = DEFAULT_RESULTS / "baseline" / "robustness" / "robustness_stats.json"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "v2_change_assets" / "imagenet_holdout_impact.png"

FAMILY_ORDER = [
    "Language-Vision-Language",
    "Language-Vision",
    "Language-Language",
    "Vision-Vision",
    "Vision-Vision-Language",
    "Vision-Language-Vision-Language",
]
FAMILY_COLORS = {
    "Language-Language": "#4E79A7",
    "Vision-Vision": "#59A14F",
    "Vision-Language-Vision-Language": "#7AA974",
    "Language-Vision": "#C44E52",
    "Language-Vision-Language": "#B07AA1",
    "Vision-Vision-Language": "#76B7B2",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_family(raw_type: str) -> str:
    return "language" if raw_type == "causal" else raw_type


def _pair_family_label(model_a: str, model_b: str, model_families: Dict[str, str]) -> str:
    ordered = tuple(sorted((_normalize_family(model_families[model_a]), _normalize_family(model_families[model_b]))))
    labels = {
        ("language", "language"): "Language-Language",
        ("language", "vision"): "Language-Vision",
        ("language", "vision_language"): "Language-Vision-Language",
        ("vision", "vision"): "Vision-Vision",
        ("vision", "vision_language"): "Vision-Vision-Language",
        ("vision_language", "vision_language"): "Vision-Language-Vision-Language",
    }
    return labels[ordered]


def _collect_rows(
    replication: Dict[str, Any],
    robustness: Dict[str, Any],
) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]], Counter[str]]:
    model_families = {
        name: meta["config"]["type"]
        for name, meta in replication["models"].items()
    }
    imagenet_rows = robustness["source_holdout"]["leave_one_source_out"]["imagenet"]["pairwise"]

    grouped: Dict[str, List[float]] = defaultdict(list)
    enriched_rows: List[Dict[str, Any]] = []
    for row in imagenet_rows:
        family = _pair_family_label(row["model_a"], row["model_b"], model_families)
        delta = float(row["delta"])
        grouped[family].append(delta)
        enriched_rows.append(
            {
                "family": family,
                "label": f"{row['model_a']} vs {row['model_b']}",
                "delta": delta,
            }
        )

    top_20 = sorted(enriched_rows, key=lambda item: abs(item["delta"]), reverse=True)[:20]
    return grouped, enriched_rows, Counter(row["family"] for row in top_20)


def _build_figure(
    grouped: Dict[str, List[float]],
    rows: List[Dict[str, Any]],
    top20_counts: Counter[str],
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.9),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    order = [family for family in FAMILY_ORDER if family in grouped]
    box_data = [grouped[family] for family in order]
    positions = np.arange(len(order))
    boxplot = axes[0].boxplot(
        box_data,
        vert=False,
        positions=positions,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.4},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
    )

    for patch, family in zip(boxplot["boxes"], order):
        patch.set_facecolor(FAMILY_COLORS[family])
        patch.set_alpha(0.78)
        patch.set_edgecolor("#333333")

    rng = np.random.default_rng(7)
    for idx, family in enumerate(order):
        values = grouped[family]
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        axes[0].scatter(
            values,
            np.full(len(values), idx) + jitter,
            s=12,
            alpha=0.25,
            color=FAMILY_COLORS[family],
            edgecolors="none",
        )
        mean_delta = float(np.mean(values))
        mean_abs = float(np.mean(np.abs(values)))
        axes[0].text(
            0.34,
            idx,
            f"mean {mean_delta:+.3f} | |Δ| {mean_abs:.3f}",
            va="center",
            ha="right",
            fontsize=8,
            color="#333333",
        )

    axes[0].axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.75, 0.34)
    axes[0].set_xlabel("Δρ after removing ImageNet")
    axes[0].set_title("Family-level shift distribution")
    axes[0].grid(axis="x", alpha=0.18)

    top_rows = sorted(rows, key=lambda item: abs(item["delta"]), reverse=True)[:12]
    top_rows.reverse()
    y = np.arange(len(top_rows))
    values = [item["delta"] for item in top_rows]
    colors = [FAMILY_COLORS[item["family"]] for item in top_rows]
    labels = [item["label"] for item in top_rows]

    axes[1].barh(y, values, color=colors, alpha=0.9)
    axes[1].axvline(0.0, color="#222222", linewidth=1.0)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].set_xlim(-0.75, 0.03)
    axes[1].set_xlabel("Δρ after removing ImageNet")
    axes[1].set_title("Largest absolute pairwise shifts")
    axes[1].grid(axis="x", alpha=0.18)

    legend_handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[family], lw=8, label=family)
        for family in order
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=7.5,
    )

    summary_text = (
        "Top 20 |Δρ| pairs:\n"
        f"{top20_counts.get('Language-Vision', 0)} language-vision\n"
        f"{top20_counts.get('Language-Vision-Language', 0)} language-VLM\n"
        f"{top20_counts.get('Vision-Vision', 0) + top20_counts.get('Vision-Vision-Language', 0) + top20_counts.get('Vision-Language-Vision-Language', 0)} vision-side only"
    )
    axes[1].text(
        0.97,
        0.08,
        summary_text,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "#F7F2E8", "edgecolor": "#BDAE9A", "boxstyle": "round,pad=0.35"},
    )

    fig.tight_layout(pad=0.7, w_pad=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manuscript-ready ImageNet holdout impact figure.",
    )
    parser.add_argument(
        "--replication",
        type=Path,
        default=DEFAULT_REPLICATION,
        help=f"Compiled baseline replication artifact (default: {DEFAULT_REPLICATION})",
    )
    parser.add_argument(
        "--robustness",
        type=Path,
        default=DEFAULT_ROBUSTNESS,
        help=f"Baseline robustness artifact (default: {DEFAULT_ROBUSTNESS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for the figure (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replication = _load_json(args.replication)
    robustness = _load_json(args.robustness)
    grouped, rows, top20_counts = _collect_rows(replication, robustness)
    _build_figure(grouped, rows, top20_counts, args.output)
    print(f"Saved ImageNet holdout impact figure -> {args.output}")


if __name__ == "__main__":
    main()
