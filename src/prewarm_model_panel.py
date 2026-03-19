import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from huggingface_hub import snapshot_download

import main_replication as mr


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODELS_FILE = os.path.join(EXPERIMENT_DIR, "docs", "model_panel_core22.txt")

WEIGHT_CANDIDATE_FILES = {
    "model.safetensors",
    "pytorch_model.bin",
    "flax_model.msgpack",
    "tf_model.h5",
}
TOKENIZER_CANDIDATE_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "merges.txt",
    "vocab.json",
    "spiece.model",
}
PROCESSOR_CANDIDATE_FILES = {
    "preprocessor_config.json",
    "processor_config.json",
}


def parse_models_file(path: str) -> List[str]:
    models: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                models.append(line)
    return models


def get_model_configs() -> Dict[str, Dict[str, object]]:
    return {
        **mr.LANGUAGE_MODELS,
        **mr.VISION_MODELS_SSL,
        **mr.VISION_MODELS_VLM,
        **mr.VISION_MODELS_AR_VLM,
    }


def mlx_override_ready(model_name: str) -> Tuple[bool, str]:
    path = mr.LOCAL_MLX_MODEL_OVERRIDES.get(model_name)
    if not path:
        return False, ""
    required_files = [
        "model.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    if all(os.path.isfile(os.path.join(path, name)) for name in required_files):
        return True, path
    return False, path


def verify_weight_files(snapshot_path: str) -> List[str]:
    root = Path(snapshot_path)
    index_candidates = [
        root / "model.safetensors.index.json",
        root / "pytorch_model.bin.index.json",
    ]
    for index_path in index_candidates:
        if not index_path.is_file():
            continue
        with open(index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        shards = sorted(set(payload.get("weight_map", {}).values()))
        return [name for name in shards if not (root / name).is_file()]

    if any((root / name).is_file() for name in WEIGHT_CANDIDATE_FILES):
        return []

    weights = list(root.glob("*.safetensors")) + list(root.glob("*.bin"))
    if weights:
        return []
    return ["<missing model weights>"]


def verify_snapshot(snapshot_path: str, model_type: str) -> List[str]:
    root = Path(snapshot_path)
    missing: List[str] = []
    if not (root / "config.json").is_file():
        missing.append("config.json")
    missing.extend(verify_weight_files(snapshot_path))

    if model_type == "causal":
        if not any((root / name).is_file() for name in TOKENIZER_CANDIDATE_FILES):
            missing.append("<missing tokenizer files>")
    elif model_type == "vision_language_autoregressive":
        if not any((root / name).is_file() for name in TOKENIZER_CANDIDATE_FILES):
            missing.append("<missing tokenizer files>")
        if not any((root / name).is_file() for name in PROCESSOR_CANDIDATE_FILES):
            missing.append("<missing processor config>")
    else:
        if not any((root / name).is_file() for name in PROCESSOR_CANDIDATE_FILES):
            missing.append("<missing processor config>")
    return missing


def prewarm_model(
    model_name: str,
    config: Dict[str, object],
    local_files_only: bool,
) -> Dict[str, object]:
    ready_override, override_path = mlx_override_ready(model_name)
    if ready_override:
        return {
            "model": model_name,
            "status": "ok",
            "source": "local_override",
            "path": override_path,
        }

    repo_id = str(config["id"])
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        return {
            "model": model_name,
            "status": "missing",
            "source": "huggingface",
            "repo_id": repo_id,
            "error": str(exc),
            "override_path": override_path or None,
        }

    missing = verify_snapshot(snapshot_path, str(config["type"]))
    return {
        "model": model_name,
        "status": "ok" if not missing else "incomplete",
        "source": "huggingface",
        "repo_id": repo_id,
        "path": snapshot_path,
        "missing": missing,
        "override_path": override_path or None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prewarm Hugging Face cache for a model panel.")
    parser.add_argument("--models-file", default=DEFAULT_MODELS_FILE)
    parser.add_argument(
        "--local-files-only",
        type=str,
        default="false",
        help="Audit cache only without downloading missing files (true|false).",
    )
    args = parser.parse_args()

    local_files_only = mr.parse_bool_arg(args.local_files_only)
    model_configs = get_model_configs()
    models = parse_models_file(os.path.abspath(args.models_file))

    results: List[Dict[str, object]] = []
    for model_name in models:
        if model_name not in model_configs:
            raise SystemExit(f"Unknown model in panel: {model_name}")
        config = model_configs[model_name]
        print(f"[prewarm] {model_name}")
        result = prewarm_model(model_name, config, local_files_only=local_files_only)
        results.append(result)
        print(json.dumps(result, ensure_ascii=True))

    bad = [item for item in results if item["status"] != "ok"]
    print("")
    print(
        json.dumps(
            {
                "models": len(results),
                "ok": len(results) - len(bad),
                "bad": len(bad),
                "local_files_only": local_files_only,
            },
            ensure_ascii=True,
        )
    )
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
