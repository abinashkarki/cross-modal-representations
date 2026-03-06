#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_ROOT="$REPO_ROOT"
RAW_DIR="$EXP_ROOT/results/replication/baseline/raw_data"
CACHE_ROOT="$HOME/.cache/huggingface/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=180

if [[ -f "$EXP_ROOT/venv/bin/activate" ]]; then
  source "$EXP_ROOT/venv/bin/activate"
fi

# Stop if extraction pipeline is running.
if pgrep -f "main_replication.py --model" >/dev/null; then
  echo "ERROR: main_replication.py is running. Stop the pipeline first."
  exit 1
fi

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

declare -A ID=(
  ["Qwen3-0.6B-MLX-8bit"]="Qwen/Qwen3-0.6B-MLX-8bit"
  ["Qwen3-1.7B-MLX-8bit"]="Qwen/Qwen3-1.7B-MLX-8bit"
  ["Qwen3-4B-MLX-8bit"]="Qwen/Qwen3-4B-MLX-8bit"
  ["LFM2-2.6B-Exp-8bit"]="mlx-community/LFM2-2.6B-Exp-8bit"
  ["SmolLM3-3B-8bit"]="mlx-community/SmolLM3-3B-8bit"
  ["Qwen2.5-1.5B-Instruct-8bit"]="mlx-community/Qwen2.5-1.5B-Instruct-8bit"
  ["Falcon3-1B-Instruct-8bit"]="mlx-community/Falcon3-1B-Instruct-8bit"
  ["Granite-3.3-2B-Instruct-8bit"]="mlx-community/granite-3.3-2b-instruct-8bit"
  ["DINOv2-small"]="facebook/dinov2-small"
  ["DINOv2-base"]="facebook/dinov2-base"
  ["ViT-MAE-base"]="facebook/vit-mae-base"
  ["BEiT-base"]="microsoft/beit-base-patch16-224"
  ["data2vec-vision"]="facebook/data2vec-vision-base"
  ["Hiera-base"]="facebook/hiera-base-224-in1k-hf"
  ["ConvNeXt-v2"]="facebook/convnextv2-base-1k-224"
  ["ViT-MSN-base"]="facebook/vit-msn-base"
  ["DINOv3-ConvNeXt-tiny"]="facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
  ["I-JEPA"]="facebook/ijepa_vith14_1k"
  ["CLIP-ViT-B32"]="openai/clip-vit-base-patch32"
  ["MetaCLIP-B32-400m"]="facebook/metaclip-b32-400m"
  ["SigLIP"]="google/siglip-base-patch16-224"
  ["SigLIP2"]="google/siglip2-base-patch16-224"
)

# local overrides already present in your repo (skip HF download for these)
has_local_override() {
  local model="$1"
  local models_root="${LOCAL_MODELS_ROOT:-$HOME/models}"
  case "$model" in
    "Qwen3-1.7B-MLX-8bit")
      [[ -f "$models_root/qwen3-1.7B-mlx-8bit/model.safetensors" ]]
      ;;
    "Qwen3-4B-MLX-8bit")
      [[ -f "$models_root/qwen3-4B-mlx-8bit/model.safetensors" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

for model in "${MODELS[@]}"; do
  if [[ -f "$RAW_DIR/$model.json" ]]; then
    echo "[SKIP] $model (already extracted)"
    continue
  fi

  repo="${ID[$model]}"
  if has_local_override "$model"; then
    echo "[SKIP] $model (using local override path)"
    continue
  fi

  cache_repo_dir="$CACHE_ROOT/models--${repo//\//--}"

  echo ""
  echo "=== $model ==="
  echo "repo: $repo"

  # clear stale partials/locks for this repo (safe because pipeline is stopped)
  if [[ -d "$cache_repo_dir" ]]; then
    find "$cache_repo_dir" -type f -name '*.incomplete' -delete || true
    find "$CACHE_ROOT/.locks/models--${repo//\//--}" -type f -name '*.lock' -delete 2>/dev/null || true
  fi

  ok=0
  for attempt in 1 2 3; do
    echo "attempt $attempt..."
    if hf download "$repo" --repo-type model --max-workers 1; then
      ok=1
      break
    fi
    sleep $((attempt * 10))
  done

  if [[ "$ok" -ne 1 ]]; then
    echo "[FAIL] $model ($repo)"
    exit 1
  fi

  echo "[OK] $model"
done

echo ""
echo "All pending model downloads completed."
