#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import source_scale250_manifest as ssm


@dataclass
class QueueItem:
    concept: str
    image_count: int
    openimages_hit_count: int
    openimages_class_count: int
    openimages_label_count: int
    openimages_labels: list[tuple[str, str]]
    openimages_classes: list[str]
    stratum: str
    feasibility: str


def complete_count(manifest: dict) -> int:
    return sum(1 for images in manifest["concept_to_images"].values() if len(images) == 15)


def partial_count(manifest: dict) -> int:
    return sum(1 for images in manifest["concept_to_images"].values() if 0 < len(images) < 15)


def empty_count(manifest: dict) -> int:
    return sum(1 for images in manifest["concept_to_images"].values() if len(images) == 0)


def openimages_hit_count(
    concept: str,
    label_catalog: dict[str, tuple[str, str]],
    cache: dict,
) -> tuple[int, list[tuple[str, str]]]:
    labels = ssm.select_openimages_image_labels(concept, label_catalog)
    label_cache = cache.get("labels", {})
    hit_count = 0
    for class_id, _label_name in labels:
        image_ids = label_cache.get(class_id, {}).get("image_ids", [])
        hit_count = max(hit_count, len(image_ids))
    return hit_count, labels


def ranked_underfilled_concepts(
    manifest: dict,
    openimages_classes: Sequence[str],
    label_catalog: dict[str, tuple[str, str]],
    cache: dict,
    include_zero_hit: bool,
) -> list[QueueItem]:
    concept_to_images = manifest["concept_to_images"]
    concept_metadata = manifest["concept_metadata"]
    queue: list[QueueItem] = []
    for concept, images in concept_to_images.items():
        image_count = len(images)
        if image_count >= 15:
            continue
        hit_count, labels = openimages_hit_count(concept, label_catalog, cache)
        class_matches = ssm.select_openimages_classes(concept, openimages_classes)
        if hit_count == 0 and not class_matches and not labels and not include_zero_hit:
            continue
        md = concept_metadata[concept]
        queue.append(
            QueueItem(
                concept=concept,
                image_count=image_count,
                openimages_hit_count=hit_count,
                openimages_class_count=len(class_matches),
                openimages_label_count=len(labels),
                openimages_labels=labels,
                openimages_classes=class_matches,
                stratum=md.get("stratum", ""),
                feasibility=md.get("source_feasibility", ""),
            )
        )
    queue.sort(
        key=lambda item: (
            item.openimages_hit_count > 0,
            item.openimages_class_count > 0,
            item.openimages_label_count > 0,
            item.feasibility == "high",
            item.image_count,
            item.openimages_hit_count,
            item.openimages_class_count,
            item.openimages_label_count,
            item.concept,
        ),
        reverse=True,
    )
    return queue


def collect_underfilled_openimages_label_ids(
    manifest: dict,
    label_catalog: dict[str, tuple[str, str]],
) -> list[str]:
    label_ids: list[str] = []
    seen = set()
    for concept, images in manifest["concept_to_images"].items():
        if len(images) >= 15:
            continue
        for class_id, _label_name in ssm.select_openimages_image_labels(concept, label_catalog):
            if class_id in seen:
                continue
            seen.add(class_id)
            label_ids.append(class_id)
    return label_ids


def format_queue_item(item: QueueItem) -> str:
    label_names = ", ".join(label_name for _class_id, label_name in item.openimages_labels) or "-"
    class_names = ", ".join(item.openimages_classes) or "-"
    return (
        f"{item.concept}: images={item.image_count} "
        f"oi_hits={item.openimages_hit_count} "
        f"oi_classes={item.openimages_class_count} "
        f"oi_labels={item.openimages_label_count} "
        f"feasibility={item.feasibility or '-'} "
        f"stratum={item.stratum or '-'} "
        f"labels=[{label_names}] "
        f"classes=[{class_names}]"
    )


def run_batch(
    script_path: Path,
    manifest_path: Path,
    batch: Sequence[str],
    python_bin: str,
    sources: Sequence[str],
) -> int:
    cmd = [
        python_bin,
        str(script_path),
        "--manifest-path",
        str(manifest_path),
        "--write",
        "--sources",
        *sources,
        "--concepts",
        *batch,
    ]
    print(f"[runner] launching batch ({len(batch)} concepts)")
    print("[runner] concepts:", ", ".join(batch))
    print("[runner] cmd:", " ".join(cmd))
    sys.stdout.flush()
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Drain the scale250 fresh manifest in ranked sourcing batches.")
    parser.add_argument("--manifest-path", default="data/data_manifest_250_fresh.json")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--max-batches", type=int, default=999)
    parser.add_argument("--max-stagnant-batches", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--include-zero-hit", action="store_true")
    parser.add_argument("--prewarm-openimages-label-hits", action="store_true")
    parser.add_argument("--sources", nargs="*", default=["imagenet", "openimages", "unsplash"])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).resolve()
    script_path = THIS_DIR / "source_scale250_manifest.py"
    openimages_classes = ssm.load_openimages_classes()
    label_catalog = ssm.load_openimages_label_catalog()
    cache = ssm.load_cached_openimages_label_hits()

    stagnant_batches = 0
    for batch_index in range(1, args.max_batches + 1):
        manifest = ssm.load_json(str(manifest_path))
        if args.prewarm_openimages_label_hits:
            label_ids = collect_underfilled_openimages_label_ids(manifest, label_catalog)
            if label_ids:
                print(f"[runner] prewarming {len(label_ids)} OpenImages image-label ids")
                sys.stdout.flush()
                ssm.ensure_openimages_label_hit_cache(label_ids)
                cache = ssm.load_cached_openimages_label_hits()
        before_complete = complete_count(manifest)
        before_partial = partial_count(manifest)
        before_empty = empty_count(manifest)
        queue = ranked_underfilled_concepts(
            manifest=manifest,
            openimages_classes=openimages_classes,
            label_catalog=label_catalog,
            cache=cache,
            include_zero_hit=args.include_zero_hit,
        )

        print(
            f"[runner] batch={batch_index} complete={before_complete} "
            f"partial={before_partial} empty={before_empty} queue={len(queue)}"
        )
        if not queue:
            print("[runner] no ranked underfilled concepts remain")
            return 0

        batch = queue[: args.batch_size]
        for item in batch:
            print("[runner] candidate", format_queue_item(item))
        sys.stdout.flush()

        if args.dry_run:
            return 0

        rc = run_batch(
            script_path=script_path,
            manifest_path=manifest_path,
            batch=[item.concept for item in batch],
            python_bin=args.python_bin,
            sources=args.sources,
        )
        if rc != 0:
            print(f"[runner] source batch failed with exit code {rc}")
            return rc

        manifest = ssm.load_json(str(manifest_path))
        after_complete = complete_count(manifest)
        after_partial = partial_count(manifest)
        after_empty = empty_count(manifest)
        gained = after_complete - before_complete
        print(
            f"[runner] batch={batch_index} result gained={gained} "
            f"complete={after_complete} partial={after_partial} empty={after_empty}"
        )
        sys.stdout.flush()

        if gained <= 0:
            stagnant_batches += 1
            print(f"[runner] stagnant_batches={stagnant_batches}")
            if stagnant_batches >= args.max_stagnant_batches:
                print("[runner] stopping after repeated no-progress batches")
                return 3
        else:
            stagnant_batches = 0

        time.sleep(args.sleep_seconds)

    print("[runner] reached max_batches limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
