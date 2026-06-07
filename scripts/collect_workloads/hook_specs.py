"""Definition-to-hook spec parsing for the workload collector."""

from __future__ import annotations

import json
from pathlib import Path

from artifact_schemas import validate_definition

INT_DTYPES = {"int32", "int64"}
WRAPPER_EXEC_METHODS = {"run", "forward", "forward_return_lse"}

FULL_TENSOR_INPUTS_BY_OP = {
    "sampling": {"probs", "top_p"},
    "gqa_ragged": {"q", "k", "v"},
    "rmsnorm": {"hidden_states", "weight", "residual"},
}
FULL_TENSOR_ALL_INPUT_OPS = {
    "gdn",
    "moe",
}

PLAN_KWARG_MAP = {
    "qo_indptr": "qo_indptr",
    "paged_kv_indptr": "kv_indptr",
    "paged_kv_indices": "kv_indices",
    "paged_kv_last_page_len": "kv_last_page_len",
    "kv_indptr": "kv_indptr",
    "kv_indices": "kv_indices",
    "indptr": "kv_indptr",
    "indices": "kv_indices",
    "last_page_len": "kv_last_page_len",
}


def normalize_hook_api(api: str) -> str:
    """Normalize method-level fi_api tags to class-level hook targets."""
    parts = api.rsplit(".", 2)
    if len(parts) == 3:
        mod_path, cls_name, method_name = parts
        if method_name in WRAPPER_EXEC_METHODS and cls_name[:1].isupper():
            return f"{mod_path}.{cls_name}"
    return api


def should_store_tensor(op_type: str, input_name: str, dtype: str) -> bool:
    """Return True when this workload input should keep captured tensor values."""
    if dtype in INT_DTYPES:
        return True
    if "scale" in input_name.lower():
        return True
    if op_type in FULL_TENSOR_ALL_INPUT_OPS:
        return True
    if input_name in FULL_TENSOR_INPUTS_BY_OP.get(op_type, set()):
        return True
    return False


def extract_hook_api(defn: dict) -> str | None:
    """Return the Python API path this hook collector should patch."""
    tags = defn.get("tags", [])
    hook_api = next(
        (
            tag.split(":", 1)[1]
            for tag in tags
            if tag.startswith("fi_api:")
        ),
        None,
    )
    return normalize_hook_api(hook_api) if hook_api else None


def extract_official_fi_api(defn: dict) -> str | None:
    """Return a real FlashInfer fi_api tag, ignoring target_api extension tags."""
    tags = defn.get("tags", [])
    api = next(
        (
            tag.split(":", 1)[1]
            for tag in tags
            if tag.startswith("fi_api:")
        ),
        None,
    )
    return api if isinstance(api, str) and api.startswith("flashinfer.") else None


def parse_def_specs(def_files: list[Path]) -> dict:
    """Parse definition files into hook specs keyed by FlashInfer API path."""
    specs: dict = {}
    for path in def_files:
        defn = json.loads(path.read_text())
        validate_definition(defn)
        hook_api = extract_hook_api(defn)
        if not hook_api:
            continue

        last_component = hook_api.rsplit(".", 1)[-1]
        is_wrapper = last_component[0].isupper()
        has_int_inputs = any(
            inp.get("dtype") in INT_DTYPES
            for inp in defn.get("inputs", {}).values()
            if inp.get("shape") is not None
        )

        if hook_api not in specs:
            specs[hook_api] = {
                "is_wrapper": is_wrapper,
                "needs_plan": False,
                "definitions": [],
            }
        if has_int_inputs:
            specs[hook_api]["needs_plan"] = True

        specs[hook_api]["definitions"].append({
            "name": defn["name"],
            "op_type": defn.get("op_type", "unknown"),
            "inputs": defn.get("inputs", {}),
            "axes": defn.get("axes", {}),
            "constraints": defn.get("constraints", []),
            "full_tensor_inputs": [
                name
                for name, inp in defn.get("inputs", {}).items()
                if inp.get("shape") is not None
                and should_store_tensor(
                    defn.get("op_type", "unknown"),
                    name,
                    inp.get("dtype", "float32"),
                )
            ],
        })

    return specs
