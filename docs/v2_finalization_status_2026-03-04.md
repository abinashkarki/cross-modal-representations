# V2 Finalization Status - March 4, 2026

## Scope
Attempted strict V2 finalization gate for `01_embeddings_convergence_basics`:
- Full-sweep policy
- MLX-blocking healthcheck
- 30 images/concept prerequisite
- Baseline profile (`RUN_PROFILE=baseline`) runner validation

## Environment Work
1. Created fresh environment:
- `venv_v2`
- Installed repo requirements from `requirements.txt`

2. Verified MLX in `venv_v2`:
- `python -c "import mlx.core as mx; print(float(mx.sum(mx.array([1.0,2.0])).item()))"`
- Result: `3.0`

3. Strict healthcheck in `venv_v2`:
- `python healthcheck_replication.py --strict ...`
- Result: `Required probes passed: True`

## Gate Results
## Gate A: Data readiness (30 images/concept)
Command:
- `python preflight_replication.py --phase pre --min-images-per-concept 30 --require-image-source-metadata true --require-clip-scores`

Result:
- **FAIL**
- All 28 concepts are still at 10 images each.

## Gate B: Baseline orchestrator
Command:
- `PYTHON_BIN=.../venv_v2/bin/python RUN_PROFILE=baseline bash docs/run_all_models.sh`

Result:
- Healthcheck: **PASS**
- Preflight(30): **FAIL** immediately on image counts.

## Data Expansion Attempt
Implemented and ran:
- `src/topup_images_to_target.py`
- Intended behavior: Unsplash public search + CLIP gate + manifest update.

Observed runtime behavior:
- Unsplash search endpoint returned `HTTP 403 Forbidden` for all tested concepts/queries/pages.
- No new images were acquired.

Hardening applied:
- `topup_images_to_target.py` now avoids writing manifest metadata on incomplete runs unless `--allow-partial` is explicitly set.
- Manifest target restored to truthful current value (`images_per_concept_target=10`).

## Current State
- MLX/runtime blocker: **Resolved** via `venv_v2`.
- Network reachability blocker: **Resolved** for healthcheck targets.
- 30-image data blocker: **Unresolved**.
- Full baseline/aligned 22-model sweeps: **Blocked** by 30-image prerequisite.

## Required Next Action
Acquire + curate 20 additional images per concept from an allowed source that is not blocked, then rerun:
1. `RUN_PROFILE=baseline ... run_all_models.sh`
2. `RUN_PROFILE=aligned5 ... run_all_models.sh`

No partial sign-off artifacts were generated.
