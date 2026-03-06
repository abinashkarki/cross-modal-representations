#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$EXPERIMENT_DIR/src"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-1200}"
RUN_PROFILE="${RUN_PROFILE:-baseline}"
ANALYSIS_LAYER="${ANALYSIS_LAYER:-selected}"
RUN_ROBUSTNESS="$(echo "${RUN_ROBUSTNESS:-1}" | tr '[:upper:]' '[:lower:]')"
LOCAL_FILES_ONLY="$(echo "${LOCAL_FILES_ONLY:-false}" | tr '[:upper:]' '[:lower:]')"
CACHE_IMAGE_EMBEDDINGS="$(echo "${CACHE_IMAGE_EMBEDDINGS:-true}" | tr '[:upper:]' '[:lower:]')"
FORCE_CACHE_REBUILD="$(echo "${FORCE_CACHE_REBUILD:-true}" | tr '[:upper:]' '[:lower:]')"
TEXT_TEMPLATE_SET="${TEXT_TEMPLATE_SET:-baseline3}"

HEALTHCHECK_REQUIRE_NETWORK="$(echo "${HEALTHCHECK_REQUIRE_NETWORK:-true}" | tr '[:upper:]' '[:lower:]')"
HEALTHCHECK_REQUIRE_MLX="$(echo "${HEALTHCHECK_REQUIRE_MLX:-true}" | tr '[:upper:]' '[:lower:]')"
HEALTHCHECK_REQUIRE_MODEL_CACHE="$(echo "${HEALTHCHECK_REQUIRE_MODEL_CACHE:-false}" | tr '[:upper:]' '[:lower:]')"

MIN_IMAGES_PER_CONCEPT="${MIN_IMAGES_PER_CONCEPT:-30}"
REQUIRE_CLIP_SCORES="$(echo "${REQUIRE_CLIP_SCORES:-true}" | tr '[:upper:]' '[:lower:]')"
REQUIRE_IMAGE_SOURCE_METADATA="$(echo "${REQUIRE_IMAGE_SOURCE_METADATA:-true}" | tr '[:upper:]' '[:lower:]')"

BOOTSTRAP_DRAWS="${BOOTSTRAP_DRAWS:-300}"
BOOTSTRAP_SAMPLE_SIZE="${BOOTSTRAP_SAMPLE_SIZE:-10}"
BOOTSTRAP_REPLACEMENT="$(echo "${BOOTSTRAP_REPLACEMENT:-true}" | tr '[:upper:]' '[:lower:]')"
MANTEL_PERMUTATIONS="${MANTEL_PERMUTATIONS:-3000}"
MIN_CONCEPTS_FOR_RSA="${MIN_CONCEPTS_FOR_RSA:-8}"

MANIFEST_PATH="${MANIFEST_PATH:-$EXPERIMENT_DIR/data/data_manifest_multi.json}"

if [ "$RUN_PROFILE" = "baseline" ]; then
    LAYERS="-1"
    EXPECTED_LAYER_PROFILE_ID="baseline_last"
    REQUESTED_LAYERS_SPEC="-1"
elif [ "$RUN_PROFILE" = "aligned5" ]; then
    LAYERS="aligned5"
    EXPECTED_LAYER_PROFILE_ID="aligned5"
    REQUESTED_LAYERS_SPEC="aligned5"
else
    echo "ERROR: RUN_PROFILE must be one of: baseline, aligned5"
    exit 1
fi

RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/replication/$RUN_PROFILE}"
LOG_DIR="$RESULTS_ROOT/logs"
RAW_DIR="$RESULTS_ROOT/raw_data"
CACHE_DIR="$RESULTS_ROOT/cache"
COMPILED_JSON="$RESULTS_ROOT/replication_results.json"
HEALTHCHECK_JSON="$RESULTS_ROOT/healthcheck.json"
ROBUSTNESS_OUTPUT_DIR="$RESULTS_ROOT/robustness"
SCALING_OUTPUT_DIR="$RESULTS_ROOT/scaling"

mkdir -p "$LOG_DIR" "$RAW_DIR" "$CACHE_DIR" "$ROBUSTNESS_OUTPUT_DIR" "$SCALING_OUTPUT_DIR"

PIPELINE_LOG="$LOG_DIR/pipeline_${RUN_PROFILE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

MODELS=(
    "Qwen3-0.6B-MLX-8bit"
    "Qwen3-1.7B-MLX-8bit"
    "Qwen3-4B-MLX-8bit"
    "LFM2-2.6B-Exp-8bit"
    "SmolLM3-3B-8bit"
    "Qwen2.5-1.5B-Instruct-8bit"
    "Falcon3-1B-Instruct-8bit"
    "Granite-3.3-2B-Instruct-8bit"
    "DINOv2-small"
    "DINOv2-base"
    "ViT-MAE-base"
    "BEiT-base"
    "data2vec-vision"
    "Hiera-base"
    "ConvNeXt-v2"
    "ViT-MSN-base"
    "DINOv3-ConvNeXt-tiny"
    "I-JEPA"
    "CLIP-ViT-B32"
    "MetaCLIP-B32-400m"
    "SigLIP"
    "SigLIP2"
)
CORE_MODEL_COUNT="${#MODELS[@]}"

if [ "$CACHE_IMAGE_EMBEDDINGS" != "true" ]; then
    echo "ERROR: CACHE_IMAGE_EMBEDDINGS must be true for V2 finalization runs."
    exit 1
fi

echo "Pipeline log: $PIPELINE_LOG"
echo "Python: $PYTHON_BIN"
echo "Run profile: $RUN_PROFILE"
echo "Layer spec: $LAYERS"
echo "Analysis layer: $ANALYSIS_LAYER"
echo "Model count (core): $CORE_MODEL_COUNT"
echo "Results root: $RESULTS_ROOT"
echo "Raw dir: $RAW_DIR"
echo "Cache dir: $CACHE_DIR"
echo "Compiled JSON: $COMPILED_JSON"
echo "Robustness output: $ROBUSTNESS_OUTPUT_DIR"

cd "$SRC_DIR"

echo "Running healthcheck..."
"$PYTHON_BIN" healthcheck_replication.py \
    --models-file "$SCRIPT_DIR/run_all_models.sh" \
    --require-network "$HEALTHCHECK_REQUIRE_NETWORK" \
    --require-mlx "$HEALTHCHECK_REQUIRE_MLX" \
    --require-model-cache "$HEALTHCHECK_REQUIRE_MODEL_CACHE" \
    --local-files-only "$LOCAL_FILES_ONLY" \
    --cache-dir "$CACHE_DIR" \
    --output-json "$HEALTHCHECK_JSON" \
    --strict

echo "Running preflight (pre)..."
PREFLIGHT_PRE_ARGS=(
    "$PYTHON_BIN" preflight_replication.py
    --phase pre
    --manifest "$MANIFEST_PATH"
    --raw-dir "$RAW_DIR"
    --models-file "$SCRIPT_DIR/run_all_models.sh"
    --cache-dir "$CACHE_DIR"
    --min-images-per-concept "$MIN_IMAGES_PER_CONCEPT"
    --require-image-source-metadata "$REQUIRE_IMAGE_SOURCE_METADATA"
)
if [ "$REQUIRE_CLIP_SCORES" = "true" ]; then
    PREFLIGHT_PRE_ARGS+=(--require-clip-scores)
fi
"${PREFLIGHT_PRE_ARGS[@]}"

FAILED_MODELS=()
TIMED_OUT_MODELS=()

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo ">>> [$RUN_PROFILE][$MODEL] starting"
    MODEL_START=$(date +%s)
    set +e
    MODEL_ARGS=(
        "$PYTHON_BIN" main_replication.py
        --model "$MODEL"
        --force
        --layers "$LAYERS"
        --output-dir "$RAW_DIR"
        --log-dir "$LOG_DIR"
        --cache-image-embeddings "$CACHE_IMAGE_EMBEDDINGS"
        --cache-dir "$CACHE_DIR"
        --text-template-set "$TEXT_TEMPLATE_SET"
        --local-files-only "$LOCAL_FILES_ONLY"
    )
    if [ "$FORCE_CACHE_REBUILD" = "true" ]; then
        MODEL_ARGS+=(--force-cache-rebuild)
    fi
    perl -e "alarm $MODEL_TIMEOUT; exec @ARGV" -- "${MODEL_ARGS[@]}"
    EXIT_CODE=$?
    set -e
    ELAPSED=$(( $(date +%s) - MODEL_START ))

    if [ $EXIT_CODE -eq 255 ]; then
        echo "!!! TIMEOUT: $MODEL (${MODEL_TIMEOUT}s)"
        TIMED_OUT_MODELS+=("$MODEL")
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "!!! FAILED: $MODEL (exit=$EXIT_CODE)"
        FAILED_MODELS+=("$MODEL")
    else
        echo "<<< OK: $MODEL (${ELAPSED}s)"
    fi
done

echo ""
echo "Failed: ${FAILED_MODELS[*]:-none}"
echo "Timed out: ${TIMED_OUT_MODELS[*]:-none}"

if [ ${#FAILED_MODELS[@]} -ne 0 ] || [ ${#TIMED_OUT_MODELS[@]} -ne 0 ]; then
    echo "ERROR: Full-sweep policy violated. Fix failed/timed-out models and rerun."
    exit 1
fi

echo "Running preflight (post)..."
PREFLIGHT_POST_ARGS=(
    "$PYTHON_BIN" preflight_replication.py
    --phase post
    --manifest "$MANIFEST_PATH"
    --raw-dir "$RAW_DIR"
    --models-file "$SCRIPT_DIR/run_all_models.sh"
    --cache-dir "$CACHE_DIR"
    --min-images-per-concept "$MIN_IMAGES_PER_CONCEPT"
    --require-image-source-metadata "$REQUIRE_IMAGE_SOURCE_METADATA"
    --require-cache
)
if [ "$REQUIRE_CLIP_SCORES" = "true" ]; then
    PREFLIGHT_POST_ARGS+=(--require-clip-scores)
fi
"${PREFLIGHT_POST_ARGS[@]}"

echo "Compiling (strict full sweep)..."
"$PYTHON_BIN" compile_results.py \
    --strict \
    --raw-dir "$RAW_DIR" \
    --output-file "$COMPILED_JSON" \
    --manifest "$MANIFEST_PATH" \
    --min-models "$CORE_MODEL_COUNT"

echo "Visualizing..."
"$PYTHON_BIN" visualize_replication_results.py \
    --data-file "$COMPILED_JSON" \
    --output-dir "$RESULTS_ROOT" \
    --layer "$ANALYSIS_LAYER"

echo "Scaling analysis..."
"$PYTHON_BIN" scaling_analysis.py \
    --data-file "$COMPILED_JSON" \
    --output-dir "$SCALING_OUTPUT_DIR" \
    --layer "$ANALYSIS_LAYER"

if [ "$RUN_ROBUSTNESS" = "1" ] || [ "$RUN_ROBUSTNESS" = "true" ]; then
    echo "Robustness analysis..."
    "$PYTHON_BIN" robustness_analysis.py \
        --data-file "$COMPILED_JSON" \
        --manifest-path "$MANIFEST_PATH" \
        --layer "$ANALYSIS_LAYER" \
        --bootstrap-draws "$BOOTSTRAP_DRAWS" \
        --bootstrap-sample-size "$BOOTSTRAP_SAMPLE_SIZE" \
        --bootstrap-replacement "$BOOTSTRAP_REPLACEMENT" \
        --mantel-permutations "$MANTEL_PERMUTATIONS" \
        --min-concepts-for-rsa "$MIN_CONCEPTS_FOR_RSA" \
        --seed 42 \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$ROBUSTNESS_OUTPUT_DIR" \
        --expected-layer-profile-id "$EXPECTED_LAYER_PROFILE_ID" \
        --requested-layers-spec "$REQUESTED_LAYERS_SPEC"
fi

echo "Done."
