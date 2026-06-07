"""Adapter for FlashInfer distributed communication kernels."""

from __future__ import annotations

from typing import Any

from adapters.official_templates import render_definition
from ._solution_utils import direct_function_solution
from .extractors import extract_observed_kwargs, extract_observed_param_names
from parse.inventory_helpers import observation_fields

OP_TYPE = "comm"


def pr_reference_source() -> str:
    return "FlashInfer communication unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} communication"


def pr_baseline_description(_def_name: str) -> str:
    return "flashinfer communication wrapper"


_API_VARIANTS = {
    "flashinfer.comm.dcp_alltoall.decode_cp_a2a_alltoall": (
        "decode_cp_a2a_alltoall",
        "decode_cp_a2a_alltoall_trace",
    ),
    "flashinfer.comm.decode_cp_a2a_alltoall": (
        "decode_cp_a2a_alltoall",
        "decode_cp_a2a_alltoall_trace",
    ),
}


def definition_name(variant: str, params: dict[str, Any]) -> str:
    """Return the canonical communication definition name."""
    if variant == "decode_cp_a2a_alltoall":
        return f"decode_cp_a2a_alltoall_d{params['head_dim']}_s{params['stats_dim']}"
    raise ValueError(f"unknown comm variant: {variant}")


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    mapped = _API_VARIANTS.get(trace_id)
    if mapped is None:
        return None
    variant, _template_name = mapped
    return (OP_TYPE, variant)


def _signature_body(sig: dict) -> dict:
    body = sig.get("signature", sig)
    return body if isinstance(body, dict) else {}


def _arg_by_name(sig: dict, name: str) -> Any:
    body = _signature_body(sig)
    kwargs = body.get("kwargs")
    if isinstance(kwargs, dict) and name in kwargs:
        return kwargs[name]

    param_names = sig.get("param_names")
    args = body.get("args")
    if isinstance(param_names, list) and isinstance(args, list) and name in param_names:
        idx = param_names.index(name)
        if idx < len(args):
            return args[idx]
    return None


def _tensor_shape(value: Any) -> list[Any] | None:
    if isinstance(value, dict) and value.get("type") == "tensor":
        shape = value.get("shape")
        if isinstance(shape, list):
            return shape
    return None


def _scalar_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("type") == "scalar":
        return value.get("value")
    if isinstance(value, (int, float, bool, str)):
        return value
    return None


def _extract_decode_cp_params(signatures: list[dict]) -> set[tuple[int, int, int]]:
    params: set[tuple[int, int, int]] = set()
    for sig in signatures:
        partial_shape = _tensor_shape(_arg_by_name(sig, "partial_o"))
        stats_shape = _tensor_shape(_arg_by_name(sig, "softmax_stats"))
        cp_size = _scalar_value(_arg_by_name(sig, "cp_size"))
        if (
            isinstance(partial_shape, list)
            and isinstance(stats_shape, list)
            and len(partial_shape) >= 3
            and len(stats_shape) >= 3
            and partial_shape[:-1] == stats_shape[:-1]
            and isinstance(partial_shape[-2], int)
            and isinstance(partial_shape[-1], int)
            and isinstance(stats_shape[-1], int)
        ):
            observed_cp_size = partial_shape[-2]
            if isinstance(cp_size, int) and cp_size != observed_cp_size:
                continue
            params.add((observed_cp_size, partial_shape[-1], stats_shape[-1]))
    return params


def _variant_info(info: dict) -> tuple[str, str] | None:
    for trace_id in info.get("trace_ids", []):
        mapped = _API_VARIANTS.get(trace_id)
        if mapped is not None:
            return mapped
    variant = info.get("variant")
    for mapped_variant, template_name in _API_VARIANTS.values():
        if mapped_variant == variant:
            return (mapped_variant, template_name)
    return None


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    kernels: list[dict] = []
    for key in [k for k in matched_kernels if k.startswith(f"{OP_TYPE}:")]:
        info = matched_kernels[key]
        variant_info = _variant_info(info)
        if variant_info is None:
            continue
        variant, template_name = variant_info
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        emitted = False
        for cp_size, head_dim, stats_dim in sorted(
            _extract_decode_cp_params(info["signatures"])
        ):
            params = {
                "cp_size": cp_size,
                "head_dim": head_dim,
                "stats_dim": stats_dim,
                "template_name": template_name,
            }
            emitted = True
            kernels.append({
                "op_type": OP_TYPE,
                "variant": variant,
                "fi_api": info["fi_api"],
                "params": params,
                "definition_name": definition_name(variant, params),
                **observation_fields(info, obs_kw, obs_pn),
            })
        if not emitted:
            kernels.append({
                "op_type": OP_TYPE,
                "variant": variant,
                "fi_api": info["fi_api"],
                "params": {"template_name": template_name},
                "definition_name": f"comm_{variant}_NEEDS_CONFIG",
                **observation_fields(info, obs_kw, obs_pn),
                "note": "Definition name requires observed communication tensor shapes.",
            })
    return kernels


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    p = kernel["params"]
    cp_size = p["cp_size"]
    head_dim = p["head_dim"]
    stats_dim = p["stats_dim"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]

    import torch

    definition = render_definition(
        "comm",
        p["template_name"],
        fi_api,
        {
            "partial_o": torch.empty((1, cp_size, head_dim), dtype=torch.bfloat16),
            "softmax_stats": torch.empty((1, cp_size, stats_dim), dtype=torch.float32),
            "workspace": torch.empty((cp_size, 1), dtype=torch.int64),
            "cp_rank": 0,
            "cp_size": cp_size,
        },
        name=name,
    )
    return definition


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate a direct FlashInfer communication baseline solution."""
    if definition.get("op_type") != OP_TYPE:
        return None
    return direct_function_solution(
        definition,
        description=(
            "Baseline solution using the traced FlashInfer communication API. "
            "Distributed runtime setup is still required for multi-rank execution."
        ),
    )
