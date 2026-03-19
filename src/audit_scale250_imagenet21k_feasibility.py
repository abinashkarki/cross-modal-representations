import importlib.util
import json
import os
import tempfile
import time
import urllib.parse
import argparse
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import requests
from nltk.corpus import wordnet as wn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_MODULE_PATH = os.path.join(SCRIPT_DIR, "source_scale250_manifest.py")
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "data_manifest_250_fresh.json")
DEFAULT_CACHE_PATH = os.path.join(REPO_ROOT, "data", "imagenet21k_filter_cache.json")
DEFAULT_OUTPUT_JSON = os.path.join(
    REPO_ROOT,
    "results",
    "summaries",
    "SCALE250_IMAGENET21K_FEASIBILITY_2026-03-15.json",
)
DEFAULT_OUTPUT_MD = os.path.join(
    REPO_ROOT,
    "results",
    "summaries",
    "SCALE250_IMAGENET21K_FEASIBILITY_2026-03-15.md",
)
DATASET_SERVER_FILTER_URL = "https://datasets-server.huggingface.co/filter"
PRIORITY_STRATA = (
    "plants_fungi",
    "food_drink",
    "natural_landforms_waterscapes",
    "buildings_infrastructure",
)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_i21k_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_scale250_i21k_", suffix=".md", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_source_module():
    spec = importlib.util.spec_from_file_location("source_scale250_manifest", SOURCE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_cache(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    return load_json(path)


def synset_label(synset: Any) -> str:
    return ", ".join(lemma.replace("_", " ") for lemma in synset.lemma_names())


def concept_synset_candidates(concept: str, max_depth: int = 1, max_nodes: int = 16) -> List[Dict[str, Any]]:
    queue: List[Tuple[Any, int]] = []
    seen = set()
    candidates: List[Dict[str, Any]] = []

    for term in (concept.replace(" ", "_"), concept):
        for synset in wn.synsets(term, pos=wn.NOUN):
            if synset.name() in seen:
                continue
            seen.add(synset.name())
            queue.append((synset, 0))

    while queue and len(candidates) < max_nodes:
        synset, depth = queue.pop(0)
        label = synset_label(synset)
        if label:
            candidates.append(
                {
                    "synset": synset.name(),
                    "label": label,
                    "depth": depth,
                }
            )
        if depth >= max_depth:
            continue
        for child in synset.hyponyms():
            if child.name() in seen:
                continue
            seen.add(child.name())
            queue.append((child, depth + 1))

    return candidates


def fetch_label_rows(label: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cached = cache.get(label)
    if cached is not None:
        return cached

    escaped_label = label.replace("'", "''")
    where = f"\"class\"='{escaped_label}'"
    params = {
        "dataset": "gmongaras/Imagenet21K",
        "config": "default",
        "split": "train",
        "where": where,
        "offset": 0,
        "length": 5,
    }

    last_error = ""
    for attempt in range(2):
        try:
            response = requests.get(DATASET_SERVER_FILTER_URL, params=params, timeout=8)
            status_code = response.status_code
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            rows = payload.get("rows", []) if status_code == 200 else []
            result = {
                "status_code": status_code,
                "hit": bool(rows),
                "num_rows": len(rows),
                "classes": sorted({row["row"]["class"] for row in rows}) if rows else [],
            }
            cache[label] = result
            return result
        except requests.RequestException as exc:
            last_error = repr(exc)
            time.sleep(1.0 + attempt)

    result = {
        "status_code": "request_error",
        "hit": False,
        "num_rows": 0,
        "classes": [],
        "error": last_error,
    }
    cache[label] = result
    return result


def iter_empty_concepts(manifest: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    ordered = []
    for concept, images in manifest.get("concept_to_images", {}).items():
        if images:
            continue
        meta = manifest["concept_metadata"][concept]
        stratum = meta.get("stratum", "")
        feasibility = meta.get("source_feasibility", "")
        stratum_rank = PRIORITY_STRATA.index(stratum) if stratum in PRIORITY_STRATA else len(PRIORITY_STRATA)
        feasibility_rank = 0 if feasibility == "high" else 1
        ordered.append((stratum_rank, feasibility_rank, stratum, concept, meta))
    for _, _, _, concept, meta in sorted(ordered):
        yield concept, meta


def audit_empty_concepts(
    manifest: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    concept_limit: int | None,
    max_depth: int,
    max_nodes: int,
    cache_path: str,
) -> Tuple[List[Dict[str, Any]], Counter]:
    rows: List[Dict[str, Any]] = []
    counts = Counter()
    processed = 0

    for concept, meta in iter_empty_concepts(manifest):
        if concept_limit is not None and processed >= concept_limit:
            break
        print(f"[imagenet21k-audit] start concept={concept!r}", flush=True)
        matches = []
        for candidate in concept_synset_candidates(concept, max_depth=max_depth, max_nodes=max_nodes):
            fetched = fetch_label_rows(candidate["label"], cache)
            if not fetched.get("hit"):
                continue
            matches.append(
                {
                    "synset": candidate["synset"],
                    "label": candidate["label"],
                    "depth": candidate["depth"],
                    "status_code": fetched.get("status_code"),
                    "num_rows": fetched.get("num_rows", 0),
                    "classes": fetched.get("classes", []),
                }
            )
            if len(matches) >= 12:
                break

        status = "imagenet21k_candidate" if matches else "no_imagenet21k_match_found"
        row = {
            "concept": concept,
            "stratum": meta.get("stratum", ""),
            "selection_status": meta.get("selection_status", ""),
            "source_feasibility": meta.get("source_feasibility", ""),
            "status": status,
            "match_count": len(matches),
            "matches": matches,
        }
        rows.append(row)
        counts[status] += 1
        processed += 1
        atomic_write_json(cache_path, cache)
        print(
            f"[imagenet21k-audit] concept={concept!r} status={status} matches={len(matches)} cache_size={len(cache)}",
            flush=True,
        )

    rows.sort(key=lambda row: (row["status"], row["stratum"], row["concept"]))
    return rows, counts


def render_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Scale250 ImageNet21K Feasibility Audit (2026-03-15)",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in report["status_counts"].items():
        lines.append(f"- `{key}`: `{value}`")

    candidate_rows = [row for row in report["concept_rows"] if row["status"] == "imagenet21k_candidate"]
    blocked_rows = [row for row in report["concept_rows"] if row["status"] == "no_imagenet21k_match_found"]

    lines.extend(
        [
            "",
            "## Candidate Matches",
            "",
        ]
    )
    if not candidate_rows:
        lines.append("- None")
    else:
        for row in candidate_rows[:40]:
            preview = "; ".join(match["label"] for match in row["matches"][:5])
            lines.append(
                f"- `{row['concept']}` ({row['stratum']}, {row['source_feasibility']}): "
                f"{row['match_count']} candidate labels; sample: {preview}"
            )

    lines.extend(
        [
            "",
            "## No Match Found",
            "",
        ]
    )
    if not blocked_rows:
        lines.append("- None")
    else:
        lines.append("- " + ", ".join(row["concept"] for row in blocked_rows[:60]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit empty concepts against ImageNet21K class availability.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--concept-limit", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=16)
    args = parser.parse_args()

    source_mod = load_source_module()
    source_mod.ensure_wordnet()
    manifest = load_json(args.manifest_path)
    cache = load_cache(args.cache_path)

    concept_rows, counts = audit_empty_concepts(
        manifest,
        cache,
        concept_limit=args.concept_limit,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        cache_path=args.cache_path,
    )
    report = {
        "status_counts": dict(counts),
        "concept_rows": concept_rows,
        "cache_size": len(cache),
        "concept_limit": args.concept_limit,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
    }

    atomic_write_json(args.cache_path, cache)
    atomic_write_json(args.output_json, report)
    atomic_write_text(args.output_md, render_md(report))

    print(f"Cache JSON: {args.cache_path}")
    print(f"Report JSON: {args.output_json}")
    print(f"Report MD: {args.output_md}")
    print(f"Status counts: {dict(counts)}")


if __name__ == "__main__":
    main()
