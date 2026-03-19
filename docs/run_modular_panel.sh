#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$EXPERIMENT_DIR/src"

resolve_repo_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "$EXPERIMENT_DIR/$path"
    fi
}

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PROFILE="${RUN_PROFILE:-baseline}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-7200}"
MANIFEST_PATH="${MANIFEST_PATH:-$EXPERIMENT_DIR/data/data_manifest_250.json}"
MODELS_FILE="${MODELS_FILE:-$SCRIPT_DIR/model_panel_core22.txt}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/scale250_full/$RUN_PROFILE}"
TEXT_TEMPLATE_SET="${TEXT_TEMPLATE_SET:-baseline3}"
LOCAL_FILES_ONLY="$(echo "${LOCAL_FILES_ONLY:-false}" | tr '[:upper:]' '[:lower:]')"
CACHE_IMAGE_EMBEDDINGS="$(echo "${CACHE_IMAGE_EMBEDDINGS:-true}" | tr '[:upper:]' '[:lower:]')"
FORCE_CACHE_REBUILD="$(echo "${FORCE_CACHE_REBUILD:-false}" | tr '[:upper:]' '[:lower:]')"
FORCE_OVERWRITE="$(echo "${FORCE_OVERWRITE:-false}" | tr '[:upper:]' '[:lower:]')"

MANIFEST_PATH="$(resolve_repo_path "$MANIFEST_PATH")"
MODELS_FILE="$(resolve_repo_path "$MODELS_FILE")"
RESULTS_ROOT="$(resolve_repo_path "$RESULTS_ROOT")"

if [ "$RUN_PROFILE" = "baseline" ]; then
    LAYERS="-1"
elif [ "$RUN_PROFILE" = "aligned5" ]; then
    LAYERS="aligned5"
else
    echo "ERROR: RUN_PROFILE must be baseline or aligned5"
    exit 1
fi

RAW_DIR="$RESULTS_ROOT/raw_data"
LOG_DIR="$RESULTS_ROOT/logs"
CACHE_DIR="$RESULTS_ROOT/cache"
mkdir -p "$RAW_DIR" "$LOG_DIR" "$CACHE_DIR"

echo "Python: $PYTHON_BIN"
echo "Run profile: $RUN_PROFILE"
echo "Layer spec: $LAYERS"
echo "Manifest path: $MANIFEST_PATH"
echo "Models file: $MODELS_FILE"
echo "Results root: $RESULTS_ROOT"
echo "Raw dir: $RAW_DIR"
echo "Log dir: $LOG_DIR"
echo "Cache dir: $CACHE_DIR"
echo "Text template set: $TEXT_TEMPLATE_SET"
echo "Local files only: $LOCAL_FILES_ONLY"
echo "Cache image embeddings: $CACHE_IMAGE_EMBEDDINGS"
echo "Force cache rebuild: $FORCE_CACHE_REBUILD"
echo "Force overwrite: $FORCE_OVERWRITE"

cd "$SRC_DIR"

FAILED_MODELS=()
TIMED_OUT_MODELS=()

while IFS= read -r RAW_LINE || [[ -n "$RAW_LINE" ]]; do
    MODEL="$(printf '%s' "$RAW_LINE" | sed 's/#.*$//' | xargs)"
    if [ -z "$MODEL" ]; then
        continue
    fi

    echo ""
    echo ">>> [$RUN_PROFILE][$MODEL] starting"
    MODEL_START=$(date +%s)
    set +e
    MODEL_ARGS=(
        "$PYTHON_BIN" main_replication.py
        --model "$MODEL"
        --layers "$LAYERS"
        --manifest-path "$MANIFEST_PATH"
        --output-dir "$RAW_DIR"
        --log-dir "$LOG_DIR"
        --cache-image-embeddings "$CACHE_IMAGE_EMBEDDINGS"
        --cache-dir "$CACHE_DIR"
        --text-template-set "$TEXT_TEMPLATE_SET"
        --local-files-only "$LOCAL_FILES_ONLY"
    )
    if [ "$FORCE_OVERWRITE" = "true" ]; then
        MODEL_ARGS+=(--force)
    fi
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
done < "$MODELS_FILE"

echo ""
echo "Failed: ${FAILED_MODELS[*]:-none}"
echo "Timed out: ${TIMED_OUT_MODELS[*]:-none}"

if [ ${#FAILED_MODELS[@]} -ne 0 ] || [ ${#TIMED_OUT_MODELS[@]} -ne 0 ]; then
    exit 1
fi

echo "Panel completed successfully."
