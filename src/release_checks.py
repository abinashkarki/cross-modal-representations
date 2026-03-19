import json
import subprocess
from fnmatch import fnmatch
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_TRACKED_BYTES = 100 * 1024 * 1024

FORBIDDEN_TRACKED_PATTERNS = [
    ".local_artifacts/*",
    "data/images_multi/*",
    "data/images_250/*",
    "data/images_250_fresh/*",
    "results/arvlm_smoke_qwen/*",
    "results/scale250_complete_subset/*",
    "results/source_runs/*",
    "results/summaries/*",
    "results/scale250_full/*/raw_data/*",
    "results/scale250_full/*/cache/*",
    "results/scale250_full/*/logs/*",
    "results/scale250_full/*/nohup.out",
    "results/scale250_full/*/replication_results.json",
    "results/scale250_full/*/run.log",
    "*.safetensors",
    "*.incomplete",
]

REQUIRED_FILES = [
    "README.md",
    "CITATION.cff",
    "RELEASE_NOTES.md",
    ".gitignore",
    ".gitattributes",
    "environment.yml",
    "requirements.txt",
    "requirements-curation.txt",
    "Makefile",
    "docs/release_reproducibility.md",
    "artifacts/release_manifest.json",
    "artifacts/SHA256SUMS.txt",
    "data/README.md",
    "data/data_manifest_250.json",
    "results/README.md",
    "results/scale250_full/baseline/robustness_opt_full/robustness_stats.json",
    "results/scale250_full/aligned5/robustness/robustness_stats.json",
    "results/scale250_full/baseline25_extension/architecture_analysis/architecture_summary.json",
    "manuscript/paper.md",
    "manuscript/paper.html",
    "manuscript/paper.tex",
    "manuscript/legacy/where_representations_diverge/paper.md",
    "src/materialize_release_artifacts.py",
]


def git_ls_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    return [line for line in output.splitlines() if line.strip()]


def check_required_files() -> list[str]:
    errors = []
    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"Missing required file: {rel}")
    return errors


def check_forbidden_tracked(tracked: list[str]) -> list[str]:
    errors = []
    for rel in tracked:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        for pattern in FORBIDDEN_TRACKED_PATTERNS:
            if fnmatch(rel, pattern):
                errors.append(f"Forbidden tracked path: {rel}")
                break
    return errors


def check_large_tracked_files(tracked: list[str]) -> list[str]:
    errors = []
    for rel in tracked:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        if path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"Tracked file exceeds {MAX_TRACKED_BYTES} bytes: {rel}")
    return errors


def check_manifest_shape() -> list[str]:
    errors = []
    manifest_path = REPO_ROOT / "artifacts" / "release_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "external_artifacts" not in manifest:
        errors.append("release_manifest.json missing external_artifacts")
    if "local_archives" not in manifest:
        errors.append("release_manifest.json missing local_archives")
    for artifact in manifest.get("external_artifacts", []):
        for key in ["id", "checkout_path", "local_archive_path", "size_bytes", "sha256"]:
            if key not in artifact:
                errors.append(f"Artifact entry missing {key}: {artifact}")
        checkout = REPO_ROOT / artifact["checkout_path"]
        if checkout.exists() and checkout.stat().st_size == 0:
            errors.append(f"Empty checkout artifact placeholder: {artifact['checkout_path']}")
    return errors


def main() -> None:
    tracked = git_ls_files()
    errors = []
    errors.extend(check_required_files())
    errors.extend(check_forbidden_tracked(tracked))
    errors.extend(check_large_tracked_files(tracked))
    errors.extend(check_manifest_shape())

    if errors:
        print("RELEASE CHECK FAILED")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("RELEASE CHECK PASSED")
    print(f"Tracked files: {len(tracked)}")


if __name__ == "__main__":
    main()
