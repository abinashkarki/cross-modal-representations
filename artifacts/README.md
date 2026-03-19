# Release Artifact Index

Heavy compiled Scale250 artifacts are intentionally excluded from git.

Use:

- [release_manifest.json](./release_manifest.json) for paths, sizes, and checksums
- [SHA256SUMS.txt](./SHA256SUMS.txt) for quick verification

In this local cleaned workspace, the excluded files still exist under `.local_artifacts/` and can be
restored into their canonical checkout paths with:

```bash
python src/materialize_release_artifacts.py --from-local-archive --all
```

When publishing a GitHub release, attach the heavy compiled artifacts using the `checkout_path` or a
clear derivative asset name, then add the asset URLs back into `release_manifest.json`.
