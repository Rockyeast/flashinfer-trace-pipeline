"""Adapter for FlashInfer FP4/FP8 quantization kernels."""

from __future__ import annotations

from typing import Any

from adapters.official_templates import render_definition
from ._solution_utils import direct_function_solution
from .extractors import extract_observed_kwargs, extract_observed_param_names
from parse.inventory_helpers import observation_fields

OP_TYPE = "quantization"
OP_TYPES = {"quantization", "quantize_fp4"}


def pr_reference_source() -> str:
    return "FlashInfer quantization unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} quantization"


def pr_baseline_description(_def_name: str) -> str:
    return "flashinfer quantization wrapper"


_API_VARIANTS = {
    "flashinfer.quantization.fp4_quantization.fp4_quantize": (
        "quantization",
        "fp4",
        "fp4_quantize_trace",
    ),
    "flashinfer.fp4_quantization.fp4_quantize": (
        "quantization",
        "fp4",
        "fp4_quantize_trace",
    ),
    "flashinfer.quantization.fp4_quantization.nvfp4_quantize": (
        "quantization",
        "nvfp4",
        "nvfp4_quantize_trace",
    ),
    "flashinfer.fp4_quantization.nvfp4_quantize": (
        "quantization",
        "nvfp4",
        "nvfp4_quantize_trace",
    ),
    "flashinfer.quantization.fp4_quantization.mxfp4_quantize": (
        "quantization",
        "mxfp4",
        "mxfp4_quantize_trace",
    ),
    "flashinfer.fp4_quantization.mxfp4_quantize": (
        "quantization",
        "mxfp4",
        "mxfp4_quantize_trace",
    ),
    "flashinfer.quantization.fp8_quantization.mxfp8_quantize": (
        "quantization",
        "mxfp8",
        "mxfp8_quantize_trace",
    ),
    "flashinfer.fp8_quantization.mxfp8_quantize": (
        "quantization",
        "mxfp8",
        "mxfp8_quantize_trace",
    ),
    "flashinfer.quantization.fp4_quantization.nvfp4_kv_quantize": (
        "quantize_fp4",
        "nvfp4_kv",
        "nvfp4_kv_quantize_trace",
    ),
    "flashinfer.fp4_quantization.nvfp4_kv_quantize": (
        "quantize_fp4",
        "nvfp4_kv",
        "nvfp4_kv_quantize_trace",
    ),
}


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    mapped = _API_VARIANTS.get(trace_id)
    if mapped is None:
        return None
    op_type, variant, _template_name = mapped
    return (op_type, variant)


def _signature_body(sig: dict) -> dict:
    body = sig.get("signature", sig)
    return body if isinstance(body, dict) else {}


def _tensor_shape(value: Any) -> list[Any] | None:
    if isinstance(value, dict) and value.get("type") == "tensor":
        shape = value.get("shape")
        if isinstance(shape, list):
            return shape
    return None


def _input_shape(sig: dict, names: tuple[str, ...]) -> list[Any] | None:
    body = _signature_body(sig)
    kwargs = body.get("kwargs")
    if isinstance(kwargs, dict):
        for name in names:
            shape = _tensor_shape(kwargs.get(name))
            if shape:
                return shape

    param_names = sig.get("param_names")
    args = body.get("args")
    if isinstance(param_names, list) and isinstance(args, list):
        for name in names:
            if name in param_names:
                idx = param_names.index(name)
                if idx < len(args):
                    shape = _tensor_shape(args[idx])
                    if shape:
                        return shape

    if isinstance(args, list):
        for arg in args:
            shape = _tensor_shape(arg)
            if shape:
                return shape
    return None


def _extract_k_values(signatures: list[dict], input_names: tuple[str, ...]) -> set[int]:
    values: set[int] = set()
    for sig in signatures:
        shape = _input_shape(sig, input_names)
        if shape and isinstance(shape[-1], int):
            values.add(shape[-1])
    return values


def _first_observed_scalar(
    signatures: list[dict],
    name: str,
    *,
    default: Any = None,
) -> Any:
    for sig in signatures:
        body = _signature_body(sig)
        kwargs = body.get("kwargs")
        if isinstance(kwargs, dict) and name in kwargs:
            value = kwargs[name]
            if isinstance(value, dict) and value.get("type") == "scalar":
                return value.get("value", default)
            if isinstance(value, (int, float, bool, str)):
                return value

        param_names = sig.get("param_names")
        args = body.get("args")
        if isinstance(param_names, list) and isinstance(args, list) and name in param_names:
            idx = param_names.index(name)
            if idx < len(args):
                value = args[idx]
                if isinstance(value, dict) and value.get("type") == "scalar":
                    return value.get("value", default)
                if isinstance(value, (int, float, bool, str)):
                    return value
    return default


def _variant_info(info: dict) -> tuple[str, str, str] | None:
    for trace_id in info.get("trace_ids", []):
        mapped = _API_VARIANTS.get(trace_id)
        if mapped is not None:
            return mapped
    variant = info.get("variant")
    for op_type, mapped_variant, template_name in _API_VARIANTS.values():
        if mapped_variant == variant:
            return (op_type, mapped_variant, template_name)
    return None


def definition_name(variant: str, params: dict[str, Any]) -> str:
    """Return the canonical quantization definition name."""
    k = params["K"]
    if variant == "fp4":
        return f"fp4_quantize_k{k}"
    if variant == "nvfp4":
        return f"nvfp4_quantize_k{k}"
    if variant == "mxfp4":
        return f"mxfp4_quantize_k{k}"
    if variant == "mxfp8":
        return f"mxfp8_quantize_k{k}"
    if variant == "nvfp4_kv":
        return f"nvfp4_kv_quantize_k{k}"
    raise ValueError(f"unknown quantization variant: {variant}")


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    kernels: list[dict] = []
    for key, info in matched_kernels.items():
        if not any(key.startswith(f"{op_type}:") for op_type in OP_TYPES):
            continue
        variant_info = _variant_info(info)
        if variant_info is None:
            continue
        op_type, variant, template_name = variant_info
        input_names = ("a",) if variant in {"nvfp4", "mxfp4"} else ("input",)
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        emitted = False
        for k in sorted(_extract_k_values(info["signatures"], input_names)):
            params = {
                "K": k,
                "template_name": template_name,
            }
            sf_vec_size = _first_observed_scalar(info["signatures"], "sf_vec_size")
            if isinstance(sf_vec_size, int):
                params["sf_vec_size"] = sf_vec_size
            alignment = _first_observed_scalar(info["signatures"], "alignment")
            if isinstance(alignment, int):
                params["alignment"] = alignment
            emitted = True
            kernels.append({
                "op_type": op_type,
                "variant": variant,
                "fi_api": info["fi_api"],
                "params": params,
                "definition_name": definition_name(variant, params),
                **observation_fields(info, obs_kw, obs_pn),
            })
        if not emitted:
            kernels.append({
                "op_type": op_type,
                "variant": variant,
                "fi_api": info["fi_api"],
                "params": {"template_name": template_name},
                "definition_name": f"{op_type}_{variant}_NEEDS_CONFIG",
                **observation_fields(info, obs_kw, obs_pn),
                "note": "Definition name requires observed quantization input K dimension.",
            })
    return kernels


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    p = kernel["params"]
    k = p["K"]
    variant = kernel["variant"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]

    import torch

    if variant == "fp4":
        kwargs: dict[str, Any] = {
            "input": torch.empty((1, k), dtype=torch.bfloat16),
            "sf_vec_size": int(p.get("sf_vec_size", 16)),
        }
        kwargs["global_scale"] = torch.ones((1,), dtype=torch.float32)
    elif variant == "nvfp4":
        kwargs = {
            "a": torch.empty((1, k), dtype=torch.bfloat16),
            "a_global_sf": torch.ones((1,), dtype=torch.float32),
            "sf_vec_size": int(p.get("sf_vec_size", 16)),
        }
    elif variant == "mxfp4":
        kwargs = {"a": torch.empty((1, k), dtype=torch.bfloat16)}
    elif variant == "mxfp8":
        kwargs = {"input": torch.empty((1, k), dtype=torch.bfloat16)}
    elif variant == "nvfp4_kv":
        kwargs = {
            "input": torch.empty((1, k), dtype=torch.bfloat16),
            "global_scale": torch.ones((1,), dtype=torch.float32),
        }
    else:
        raise ValueError(f"unknown quantization variant: {variant}")

    definition = render_definition(
        "quantize",
        p["template_name"],
        fi_api,
        kwargs,
        name=name,
    )
    return definition


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate a direct FlashInfer quantization baseline solution."""
    if definition.get("op_type") not in OP_TYPES:
        return None
    return direct_function_solution(
        definition,
        description="Baseline solution using the traced FlashInfer quantization API.",
    )
