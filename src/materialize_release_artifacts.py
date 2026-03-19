import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "release_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_artifacts(manifest: dict, artifact_ids: list[str], all_flag: bool) -> list[dict]:
    artifacts = manifest["external_artifacts"]
    if all_flag:
        return artifacts
    wanted = set(artifact_ids)
    selected = [artifact for artifact in artifacts if artifact["id"] in wanted]
    missing = wanted - {artifact["id"] for artifact in selected}
    if missing:
        raise SystemExit(f"Unknown artifact ids: {', '.join(sorted(missing))}")
    return selected


def copy_from_local_archive(artifact: dict, force: bool) -> None:
    src = REPO_ROOT / artifact["local_archive_path"]
    dst = REPO_ROOT / artifact["checkout_path"]
    if not src.exists():
        raise SystemExit(f"Local archive source missing: {src}")
    if dst.exists() and not force:
        raise SystemExit(f"Destination exists: {dst}. Use --force to overwrite.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def download_from_url(artifact: dict, force: bool) -> None:
    url = artifact.get("url")
    if not url:
        raise SystemExit(f"Artifact {artifact['id']} has no URL in the manifest.")
    dst = REPO_ROOT / artifact["checkout_path"]
    if dst.exists() and not force:
        raise SystemExit(f"Destination exists: {dst}. Use --force to overwrite.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dst, "wb") as handle:
        shutil.copyfileobj(response, handle)


def verify_artifact(artifact: dict) -> None:
    dst = REPO_ROOT / artifact["checkout_path"]
    if not dst.exists():
        raise SystemExit(f"Materialized artifact missing after copy/download: {dst}")
    actual_size = dst.stat().st_size
    if actual_size != artifact["size_bytes"]:
        raise SystemExit(
            f"Size mismatch for {artifact['id']}: expected {artifact['size_bytes']}, got {actual_size}"
        )
    actual_hash = sha256_file(dst)
    if actual_hash != artifact["sha256"]:
        raise SystemExit(
            f"SHA256 mismatch for {artifact['id']}: expected {artifact['sha256']}, got {actual_hash}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore heavy release artifacts into canonical checkout paths.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--artifact", action="append", default=[], help="Artifact id to materialize.")
    parser.add_argument("--all", action="store_true", help="Materialize all indexed artifacts.")
    parser.add_argument(
        "--from-local-archive",
        action="store_true",
        help="Copy artifacts from .local_artifacts instead of downloading from manifest URLs.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing destination files.")
    args = parser.parse_args()

    if not args.all and not args.artifact:
        raise SystemExit("Specify --all or at least one --artifact id.")

    manifest = load_manifest(Path(args.manifest))
    artifacts = select_artifacts(manifest, args.artifact, args.all)

    for artifact in artifacts:
        print(f"Materializing {artifact['id']} -> {artifact['checkout_path']}")
        if args.from_local_archive:
            copy_from_local_archive(artifact, force=args.force)
        else:
            download_from_url(artifact, force=args.force)
        verify_artifact(artifact)
        print("  OK")


if __name__ == "__main__":
    main()

