import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_SCRIPT = SCRIPT_DIR / "source_scale250_manifest.py"


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def concept_status(manifest: dict, concept: str) -> str:
    meta = manifest["concept_metadata"][concept]
    actual = meta["source_mix_actual"]
    if meta["num_images"] == 15 and actual == {"imagenet": 5, "openimages": 5, "unsplash": 5}:
        return "complete"
    if meta["num_images"] == 0:
        return "empty"
    return f"partial:{meta['num_images']}:{actual}"


def run_concept(manifest_path: Path, concept: str, timeout_seconds: int, sources: list[str] | None) -> str:
    cmd = [
        sys.executable,
        str(SOURCE_SCRIPT),
        "--manifest-path",
        str(manifest_path),
        "--write",
        "--concepts",
        concept,
    ]
    if sources:
        cmd.extend(["--sources", *sources])

    started = time.time()
    print(f"[runner] start concept={concept!r}")
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.time() - started
        print(f"[runner] end concept={concept!r} exit={completed.returncode} elapsed={elapsed:.1f}s")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        print(f"[runner] timeout concept={concept!r} elapsed={elapsed:.1f}s")

    manifest = load_manifest(manifest_path)
    status = concept_status(manifest, concept)
    print(f"[runner] manifest concept={concept!r} status={status}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scale250 sourcing one concept at a time with timeouts.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=480)
    parser.add_argument("--sources", nargs="*")
    parser.add_argument("--concepts", nargs="+", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).resolve()
    initial_manifest = load_manifest(manifest_path)
    queue: list[str] = []
    for concept in args.concepts:
        if concept_status(initial_manifest, concept) == "complete":
            print(f"[runner] skip concept={concept!r} reason=already_complete")
            continue
        queue.append(concept)

    results: dict[str, str] = {}
    for concept in queue:
        results[concept] = run_concept(manifest_path, concept, args.timeout_seconds, args.sources)

    print("[runner] summary")
    for concept in queue:
        print(f"[runner] {concept!r} -> {results[concept]}")


if __name__ == "__main__":
    main()
