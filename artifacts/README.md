# Release Artifact Index

Heavy compiled Scale250 artifacts are intentionally excluded from git.

Current release status:

- `release_manifest.json` is marked `local_archive_only`
- external download URLs have not been published yet
- `src/materialize_release_artifacts.py` therefore works only with `--from-local-archive`

Use:

- [release_manifest.json](./release_manifest.json) for paths, sizes, and checksums
- [SHA256SUMS.txt](./SHA256SUMS.txt) for quick verification

In this local cleaned workspace, the excluded files still exist under `.local_artifacts/` and can be
restored into their canonical checkout paths with:

```bash
python src/materialize_release_artifacts.py --from-local-archive --all
```

When publishing external bundles, attach the heavy compiled artifacts using the `checkout_path` or a
clear derivative asset name, then switch the manifest to `urls_published` and add the artifact URLs
back into `release_manifest.json`.
