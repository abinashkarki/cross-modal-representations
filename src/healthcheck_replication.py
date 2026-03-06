import argparse
import json
import os
import re
import socket
import subprocess
import sys
import traceback
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODELS_FILE = os.path.join(EXPERIMENT_DIR, "docs", "run_all_models.sh")
DEFAULT_CACHE_DIR = os.path.join(EXPERIMENT_DIR, "results", "replication", "cache")
DEFAULT_OUTPUT_JSON = os.path.join(EXPERIMENT_DIR, "results", "replication", "healthcheck.json")
CACHE_SCHEMA_VERSION = "1.1.0"


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_models_from_shell_script(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    match = re.search(r"MODELS=\((.*?)\)", text, re.DOTALL)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def probe_network() -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {}
    try:
        resolved = socket.gethostbyname("huggingface.co")
        details["dns_huggingface_co"] = resolved
    except Exception as exc:
        details["dns_error"] = str(exc)
        return False, details

    try:
        req = urllib.request.Request("https://huggingface.co", method="GET")
        with urllib.request.urlopen(req, timeout=12) as response:
            code = int(getattr(response, "status", 0) or response.getcode())
            details["http_status"] = code
            ok = 200 <= code < 400
            return ok, details
    except Exception as exc:
        details["http_error"] = str(exc)
        return False, details


def probe_mlx() -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {}
    probe_code = (
        "import json\n"
        "import mlx.core as mx\n"
        "x = mx.array([1.0, 2.0], dtype=mx.float32)\n"
        "y = float(mx.sum(x).item())\n"
        "print(json.dumps({'sum_check': y, 'version': getattr(mx, '__version__', 'unknown')}))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
        )
        details["return_code"] = proc.returncode
        if proc.returncode != 0:
            details["stderr"] = proc.stderr.strip()
            details["stdout"] = proc.stdout.strip()
            return False, details
        parsed = json.loads(proc.stdout.strip() or "{}")
        details.update(parsed)
        y = float(parsed.get("sum_check", 0.0))
        if abs(y - 3.0) > 1e-6:
            details["error"] = f"unexpected sum result: {y}"
            return False, details
        return True, details
    except Exception as exc:
        details["error"] = str(exc)
        details["traceback"] = traceback.format_exc(limit=2)
        return False, details


def probe_python_torch() -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {
        "python_version": ".".join(map(str, sys.version_info[:3]))
    }
    probe_code = (
        "import json\n"
        "import torch\n"
        "device = 'mps' if torch.backends.mps.is_available() else 'cpu'\n"
        "t = torch.tensor([1.0, 2.0], device=device)\n"
        "v = float((t * t).sum().item())\n"
        "print(json.dumps({'torch_version': torch.__version__, 'device': device, 'torch_sanity_value': v}))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
        )
        details["return_code"] = proc.returncode
        if proc.returncode != 0:
            details["stderr"] = proc.stderr.strip()
            details["stdout"] = proc.stdout.strip()
            return False, details
        parsed = json.loads(proc.stdout.strip() or "{}")
        details.update(parsed)
        v = float(parsed.get("torch_sanity_value", 0.0))
        if abs(v - 5.0) > 1e-6:
            details["error"] = f"unexpected torch sanity value: {v}"
            return False, details
        return True, details
    except Exception as exc:
        details["error"] = str(exc)
        details["traceback"] = traceback.format_exc(limit=2)
        return False, details


def _validate_cache_manifest_fields(payload: Dict[str, Any], model_name: str) -> None:
    required_fields = [
        "schema_version",
        "model_name",
        "model_config_fingerprint",
        "manifest_fingerprint",
        "layer_keys",
        "layer_profile_id",
        "requested_layers_spec",
        "concept_to_images",
        "dtype",
        "embedding_dim",
        "created_at",
    ]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"{model_name}: cache manifest missing fields {missing}")
    if payload["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"{model_name}: cache schema {payload['schema_version']} != {CACHE_SCHEMA_VERSION}"
        )
    if payload["model_name"] != model_name:
        raise ValueError(f"{model_name}: cache manifest model mismatch ({payload['model_name']})")
    if not isinstance(payload.get("layer_keys"), list) or not payload["layer_keys"]:
        raise ValueError(f"{model_name}: cache manifest layer_keys invalid")
    if not isinstance(payload.get("concept_to_images"), dict) or not payload["concept_to_images"]:
        raise ValueError(f"{model_name}: cache manifest concept_to_images invalid")


def probe_model_cache(cache_dir: str, models: List[str]) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {
        "cache_dir": os.path.abspath(cache_dir),
        "model_count": len(models),
        "missing_manifests": [],
        "missing_shards": [],
        "checked_models": [],
    }
    if not models:
        details["error"] = "No models parsed from --models-file; cannot verify cache coverage."
        return False, details
    ok = True
    for model_name in models:
        manifest_path = os.path.join(cache_dir, model_name, "cache_manifest.json")
        if not os.path.exists(manifest_path):
            ok = False
            details["missing_manifests"].append(manifest_path)
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            _validate_cache_manifest_fields(payload, model_name)
        except Exception as exc:
            ok = False
            details["missing_manifests"].append(f"{manifest_path} ({exc})")
            continue

        details["checked_models"].append(model_name)
        layer_keys = payload.get("layer_keys", [])
        concept_to_images = payload.get("concept_to_images", {})
        is_language_cache = payload.get("text_templates") is not None
        if is_language_cache:
            continue

        for concept in sorted(concept_to_images.keys()):
            for layer_key in layer_keys:
                shard_path = os.path.join(cache_dir, model_name, layer_key, f"{concept}.npy")
                if not os.path.exists(shard_path):
                    ok = False
                    details["missing_shards"].append(shard_path)

    details["missing_manifest_count"] = len(details["missing_manifests"])
    details["missing_shard_count"] = len(details["missing_shards"])
    if details["missing_shards"]:
        details["regeneration_hint"] = (
            "Regenerate missing cache with main_replication.py --force-cache-rebuild "
            "--cache-image-embeddings true for affected models."
        )
    return ok, details


def run_healthcheck(
    models_file: str,
    require_network: bool,
    require_mlx: bool,
    require_model_cache: bool,
    local_files_only: bool,
    cache_dir: str,
) -> Dict[str, Any]:
    models = parse_models_from_shell_script(models_file)
    if local_files_only:
        require_model_cache = True

    probes: List[Dict[str, Any]] = []
    probe_specs = [
        ("python_torch", True, probe_python_torch),
        ("network", require_network, probe_network),
        ("mlx", require_mlx, probe_mlx),
    ]
    if require_model_cache:
        probe_specs.append(
            ("model_cache", True, lambda: probe_model_cache(cache_dir, models))
        )

    for name, required, fn in probe_specs:
        ok, details = fn()
        probes.append(
            {
                "name": name,
                "required": required,
                "ok": bool(ok),
                "details": details,
            }
        )

    all_required_ok = all((not p["required"]) or p["ok"] for p in probes)
    return {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "models_file": os.path.abspath(models_file),
            "cache_dir": os.path.abspath(cache_dir),
            "local_files_only": local_files_only,
            "require_network": require_network,
            "require_mlx": require_mlx,
            "require_model_cache": require_model_cache,
            "parsed_model_count": len(models),
            "parsed_models": models,
        },
        "probes": probes,
        "all_required_ok": all_required_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcheck runner for replication sweep.")
    parser.add_argument("--models-file", default=DEFAULT_MODELS_FILE)
    parser.add_argument("--require-network", type=str, default="true")
    parser.add_argument("--require-mlx", type=str, default="true")
    parser.add_argument("--require-model-cache", type=str, default="false")
    parser.add_argument("--local-files-only", type=str, default="false")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any required probe fails.",
    )
    args = parser.parse_args()

    require_network = parse_bool(args.require_network)
    require_mlx = parse_bool(args.require_mlx)
    require_model_cache = parse_bool(args.require_model_cache)
    local_files_only = parse_bool(args.local_files_only)

    payload = run_healthcheck(
        models_file=args.models_file,
        require_network=require_network,
        require_mlx=require_mlx,
        require_model_cache=require_model_cache,
        local_files_only=local_files_only,
        cache_dir=args.cache_dir,
    )

    output_json = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    failed_required = [
        p for p in payload["probes"] if p["required"] and not p["ok"]
    ]
    print(f"Healthcheck output: {output_json}")
    print(f"Required probes passed: {payload['all_required_ok']}")
    if failed_required:
        print("Failed required probes:")
        for probe in failed_required:
            print(f"  - {probe['name']}: {probe['details']}")

    if args.strict and not payload["all_required_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
