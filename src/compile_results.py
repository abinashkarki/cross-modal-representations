import argparse
import json
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(EXPERIMENT_DIR, "results", "replication", "raw_data")
COMPILED_OUTPUT_PATH = os.path.join(
    EXPERIMENT_DIR, "results", "replication", "replication_results.json"
)
DEFAULT_MANIFEST_PATH = os.path.join(EXPERIMENT_DIR, "data", "data_manifest_250.json")


def load_expected_concepts(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    concept_to_images = manifest.get("concept_to_images", {})
    return list(concept_to_images.keys())


def validate_model_metadata(model_name, model_info):
    """Lightweight pass-through checks for new robustness metadata."""
    if "layer_metadata" not in model_info:
        return [f"{model_name}: missing layer_metadata"]
    warnings = []
    layer_meta = model_info.get("layer_metadata", {})
    if "default_layer_key" not in layer_meta:
        warnings.append(f"{model_name}: layer_metadata.default_layer_key missing")
    if "selected_layer_keys" not in layer_meta:
        warnings.append(f"{model_name}: layer_metadata.selected_layer_keys missing")

    template_meta = model_info.get("text_template_metadata")
    template_embs = model_info.get("text_template_embeddings_by_layer")
    if template_meta is not None and template_embs is None:
        warnings.append(
            f"{model_name}: text_template_metadata present but text_template_embeddings_by_layer missing"
        )
    if template_meta:
        baseline_key = template_meta.get("baseline_template_key")
        templates = template_meta.get("templates", {})
        if baseline_key and baseline_key not in templates:
            warnings.append(
                f"{model_name}: baseline template '{baseline_key}' missing from templates"
            )
    return warnings


def compile_results(
    strict=True,
    manifest_path=None,
    min_models=None,
    partial=False,
    raw_dir=RAW_DATA_DIR,
    output_file=COMPILED_OUTPUT_PATH,
):
    input_dir = os.path.abspath(raw_dir)
    output_file = os.path.abspath(output_file)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Individual JSON directory {input_dir} does not exist.")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    expected_concepts = None
    if manifest_path:
        manifest_path = os.path.abspath(manifest_path)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        expected_concepts = load_expected_concepts(manifest_path)

    compiled_results = {
        "timestamp": datetime.now().isoformat(),
        "concepts": [],
        "models": {},
        "metadata": {
            "strict_mode": strict,
            "partial": partial,
            "manifest_path": manifest_path,
            "raw_dir": input_dir,
            "output_file": output_file,
            "source_files": [],
        },
    }

    reference_concepts = None
    concept_mismatches = []
    model_name_collisions = []
    metadata_warnings = []

    raw_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))
    if not raw_files:
        raise RuntimeError(f"No JSON files found in {input_dir}.")

    for filename in raw_files:
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, "r") as f:
                model_data = json.load(f)
        except Exception as e:
            msg = f"Error reading {filename}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            print(f"WARNING: {msg}")
            continue

        file_concepts = model_data.get("concepts", [])
        if not file_concepts:
            msg = f"{filename} has no 'concepts' list."
            if strict:
                raise ValueError(msg)
            print(f"WARNING: {msg}")
            continue

        if reference_concepts is None:
            reference_concepts = file_concepts
            compiled_results["concepts"] = file_concepts
        elif file_concepts != reference_concepts:
            ref_set = set(reference_concepts)
            file_set = set(file_concepts)
            mismatch = {
                "file": filename,
                "reference_count": len(reference_concepts),
                "file_count": len(file_concepts),
                "extra_concepts": sorted(file_set - ref_set),
                "missing_concepts": sorted(ref_set - file_set),
            }
            concept_mismatches.append(mismatch)
            if strict:
                raise ValueError(
                    f"Concept mismatch in {filename}: "
                    f"{len(file_concepts)} vs reference {len(reference_concepts)}."
                )
            print(f"WARNING: concept mismatch in {filename}: {mismatch}")

        if "models" not in model_data:
            msg = f"{filename} has no 'models' key."
            if strict:
                raise ValueError(msg)
            print(f"WARNING: {msg}")
            continue

        compiled_results["metadata"]["source_files"].append(filename)
        for model_name, model_info in model_data["models"].items():
            if model_name in compiled_results["models"]:
                model_name_collisions.append(model_name)
                msg = f"Duplicate model name encountered: {model_name}"
                if strict:
                    raise ValueError(msg)
                print(f"WARNING: {msg} (last one wins)")
            warnings = validate_model_metadata(model_name, model_info)
            if warnings:
                metadata_warnings.extend(warnings)
                if strict:
                    raise ValueError(
                        "Metadata validation failed:\n" + "\n".join(warnings)
                    )
            compiled_results["models"][model_name] = model_info
            print(f"Merged results for {model_name}")

    if expected_concepts is not None:
        expected_set = set(expected_concepts)
        current_set = set(compiled_results["concepts"])
        extra = sorted(current_set - expected_set)
        missing = sorted(expected_set - current_set)
        if extra or missing:
            msg = (
                "Compiled concept list does not match manifest.\n"
                f"  compiled_count={len(compiled_results['concepts'])}, "
                f"manifest_count={len(expected_concepts)}\n"
                f"  extra_in_compiled={extra}\n"
                f"  missing_from_compiled={missing}"
            )
            if strict:
                raise ValueError(msg)
            print(f"WARNING: {msg}")

    num_models = len(compiled_results["models"])
    compiled_results["metadata"]["num_models"] = num_models
    compiled_results["metadata"]["num_concepts"] = len(compiled_results["concepts"])
    compiled_results["metadata"]["concept_mismatches"] = concept_mismatches
    compiled_results["metadata"]["model_name_collisions"] = sorted(set(model_name_collisions))
    compiled_results["metadata"]["metadata_warnings"] = sorted(set(metadata_warnings))

    if metadata_warnings and not strict:
        print("WARNING: metadata validation warnings:")
        for warning in sorted(set(metadata_warnings)):
            print(f"  - {warning}")

    # Enforce minimum model count — refuse to compile garbage partial results
    if min_models is not None and num_models < min_models:
        raise RuntimeError(
            f"Only {num_models} model(s) compiled, but --min-models={min_models} required. "
            f"Found: {sorted(compiled_results['models'].keys())}"
        )

    if partial:
        print(f"WARNING: Compiling PARTIAL results ({num_models} models). "
              f"Not all configured models are present.")

    with open(output_file, "w") as f:
        json.dump(compiled_results, f, indent=4)

    status = "PARTIAL" if partial else "complete"
    print(f"\nSuccessfully compiled {status} results into {output_file}")
    print(
        f"  models={num_models}, "
        f"concepts={compiled_results['metadata']['num_concepts']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile per-model replication JSONs.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=RAW_DATA_DIR,
        help=f"Directory containing per-model JSON files (default: {RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=COMPILED_OUTPUT_PATH,
        help=f"Compiled output JSON path (default: {COMPILED_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed files or concept/model mismatches.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=DEFAULT_MANIFEST_PATH,
        help="Optional concept manifest to enforce.",
    )
    parser.add_argument(
        "--no-manifest-check",
        action="store_true",
        help="Disable concept check against manifest file.",
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=None,
        help="Refuse to compile if fewer than N models are present.",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Allow partial results (not all configured models required). Stamps output as partial.",
    )
    args = parser.parse_args()

    manifest_path = None if args.no_manifest_check else args.manifest
    compile_results(
        strict=args.strict,
        manifest_path=manifest_path,
        min_models=args.min_models,
        partial=args.partial,
        raw_dir=args.raw_dir,
        output_file=args.output_file,
    )
