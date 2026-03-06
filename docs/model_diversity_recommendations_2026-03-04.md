# Model Diversity Recommendations (2026-03-04)

This is the recommended model set for the clean replication run, optimized for:

- language-family diversity
- objective diversity in vision encoders
- quantization-matched controls
- local hardware constraints (M1/16GB, primary language sweep at `<=4B`)

## Core language set (primary analysis)

- `Qwen3-0.6B-MLX-8bit`
- `Qwen3-1.7B-MLX-8bit`
- `Qwen3-4B-MLX-8bit`
- `Qwen2.5-1.5B-Instruct-8bit`
- `Falcon3-1B-Instruct-8bit`
- `Granite-3.3-2B-Instruct-8bit`
- `LFM2-2.6B-Exp-8bit`
- `SmolLM3-3B-8bit`

Rationale:
- Main analysis keeps a consistent high-fidelity quantization baseline (`Q8`) for causal comparisons.
- Falcon and Granite add non-Qwen families with different training recipes.
- LFM and SmolLM3 preserve continuity with prior runs and provide stronger-size anchors.

## Optional quantization sensitivity (same runner)

- `Qwen3-1.7B-MLX-4bit`
- `Qwen3-4B-MLX-4bit`

Use:

```bash
INCLUDE_QUANT_SENSITIVITY=1 bash experiments/01_embeddings_convergence_basics/docs/run_all_models.sh
```

Rationale:
- Keeps the default run clean while preserving a one-command robustness check.

## Cross-layer support (now available)

- extraction: `python .../main_replication.py --layers all` (or `--layers 0,4,8,-1`)
- pipeline: `LAYERS=all` in `docs/run_all_models.sh`
- analysis: `--layer selected|last|layer_N` in both visualization/scaling scripts

Backend caveat:
- MLX language backend and current VLM `get_image_features` path return final-layer features only.

## Vision SSL set

- `DINOv2-small`
- `DINOv2-base`
- `ViT-MAE-base`
- `BEiT-base`
- `data2vec-vision`
- `Hiera-base`
- `ConvNeXt-v2`
- `ViT-MSN-base`
- `I-JEPA`
- `DINOv3-ConvNeXt-tiny`

Rationale:
- Covers multiple objective families: DINO distillation, MAE-style masking, BEiT token prediction, JEPA/MSN paradigms.
- Includes both ViT-like and ConvNeXt/hierarchical backbones.

## Vision-language image encoders

- `CLIP-ViT-B32`
- `MetaCLIP-B32-400m`
- `SigLIP`
- `SigLIP2`

Rationale:
- Improves robustness against single-family VLM encoder bias.
- CLIP vs SigLIP variants test whether alignment-loss differences alter concept geometry.

## Sidecar models (separate analysis track)

- `Qwen3.5-0.8B`
- `Qwen3.5-2B`

Rationale:
- Qwen3.5 model cards indicate a vision-encoder component; treat as multimodal sidecar, not pure-language control.
- Useful for sensitivity checks and future multimodal comparison runs.

## Excluded for core run

- `>4B` language models in primary sweep (methodological constraint).
- gated models (for example Llama/Gemma variants) unless auth workflow is set up in advance.

## Source links

- Qwen3.5 model cards: [0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Qwen2.5 MLX q8: [mlx-community/Qwen2.5-1.5B-Instruct-8bit](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-8bit)
- Falcon3 MLX q8: [mlx-community/Falcon3-1B-Instruct-8bit](https://huggingface.co/mlx-community/Falcon3-1B-Instruct-8bit)
- Granite MLX q8: [mlx-community/granite-3.3-2b-instruct-8bit](https://huggingface.co/mlx-community/granite-3.3-2b-instruct-8bit)
- LFM2 MLX q8: [mlx-community/LFM2-2.6B-Exp-8bit](https://huggingface.co/mlx-community/LFM2-2.6B-Exp-8bit)
- SmolLM3 MLX q8: [mlx-community/SmolLM3-3B-8bit](https://huggingface.co/mlx-community/SmolLM3-3B-8bit)
- DINOv2: [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small), [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base)
- DINOv3 ConvNeXt: [facebook/dinov3-convnext-tiny-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m)
- ViT-MSN: [facebook/vit-msn-base](https://huggingface.co/facebook/vit-msn-base)
- CLIP/MetaCLIP/SigLIP: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32), [facebook/metaclip-b32-400m](https://huggingface.co/facebook/metaclip-b32-400m), [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224), [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224)
