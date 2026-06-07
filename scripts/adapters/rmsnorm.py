"""Adapter for FlashInfer RMSNorm kernels."""

from __future__ import annotations

import textwrap
import re

from ._param_utils import analysis_allows_family, config_value, first_int
from adapters.official_templates import render_definition
from ._solution_utils import input_names, solution_payload
from .extractors import (
    extract_observed_kwargs,
    extract_observed_param_names,
    extract_rmsnorm_hidden_sizes,
)
from parse.inventory_helpers import observation_fields

OP_TYPE = "rmsnorm"


def pr_reference_source() -> str:
    return "FlashInfer norm unit tests"


def pr_kernel_description(def_name: str, model_label: str) -> str:
    return f"{model_label} {def_name.replace('_', ' ')}"


def pr_baseline_description(def_name: str) -> str:
    if "fused_add" in def_name:
        if "gemma" in def_name:
            return "flashinfer.norm.gemma_fused_add_rmsnorm"
        return "flashinfer.norm.fused_add_rmsnorm"
    if "gemma" in def_name:
        return "flashinfer.norm.gemma_rmsnorm"
    return "flashinfer.norm.rmsnorm"


def _fill_rmsnorm_descriptions(definition: dict, *, fused: bool, gemma: bool) -> None:
    """Fill descriptions used by existing RMSNorm dataset definitions."""
    definition.get("axes", {}).get("batch_size", {}).setdefault(
        "description", "Number of rows to normalize."
    )
    definition.get("axes", {}).get("hidden_size", {}).setdefault(
        "description", "Hidden dimension of each row."
    )

    hidden_description = (
        "Input activations before residual addition."
        if fused
        else "Input activations to normalize."
    )
    for input_name in ("hidden_states", "input"):
        definition.get("inputs", {}).get(input_name, {}).setdefault(
            "description", hidden_description
        )
    definition.get("inputs", {}).get("residual", {}).setdefault(
        "description", "Residual tensor added before RMS normalization."
    )
    definition.get("inputs", {}).get("weight", {}).setdefault(
        "description", "Per-channel RMSNorm weight."
    )

    if fused:
        output_description = "Fused residual-add and RMSNorm output."
    elif gemma:
        output_description = "Gemma-style normalized output activations."
    else:
        output_description = "Normalized output activations."
    for output_name in ("output", "out"):
        definition.get("outputs", {}).get(output_name, {}).setdefault(
            "description", output_description
        )


def _snake_tokens(name: str) -> set[str]:
    return {piece for piece in re.split(r"[_\W]+", name.lower()) if piece}


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    parts = trace_id.split(".")
    if len(parts) < 3 or parts[0] != "flashinfer" or parts[1] != "norm":
        return None
    func = parts[-1]
    tokens = _snake_tokens(func)
    if "gemma" in tokens and {"fused", "add"}.issubset(tokens):
        return (OP_TYPE, "gemma_fused_add_rmsnorm")
    if "gemma" in tokens:
        return (OP_TYPE, "gemma_rmsnorm")
    if {"fused", "add"}.issubset(tokens):
        return (OP_TYPE, "fused_add_rmsnorm")
    if "rmsnorm" in tokens:
        return (OP_TYPE, "rmsnorm")
    return None


def definition_name(variant: str, params: dict) -> str:
    """Return the canonical RMSNorm definition name for resolved params."""
    return f"{variant}_h{params['hidden_size']}"


def fi_api_for_variant(variant: str) -> str | None:
    """Return the FlashInfer API tag used for static candidates."""
    return {
        "rmsnorm": "flashinfer.norm.rmsnorm",
        "fused_add_rmsnorm": "flashinfer.norm.fused_add_rmsnorm",
        "gemma_rmsnorm": "flashinfer.norm.gemma_rmsnorm",
        "gemma_fused_add_rmsnorm": "flashinfer.norm.gemma_fused_add_rmsnorm",
    }.get(variant)


def resolve_rmsnorm_params(
    *,
    raw_params: dict | None = None,
    config: dict | None = None,
) -> dict[str, int] | None:
    """Resolve canonical RMSNorm params owned by this adapter."""
    raw = raw_params or {}
    h = first_int(raw.get("hidden_size"), config_value(config or {}, "hidden_size"))
    return {"hidden_size": h} if h else None


def resolve_config_params(
    kernel: dict,
    config: dict,
    _tp_size: int | None,
) -> dict[str, int] | None:
    """Resolve missing RMSNorm params from HF config."""
    return resolve_rmsnorm_params(raw_params=kernel.get("params"), config=config)


def static_candidates(
    dims: dict,
    analysis: dict,
    *,
    tp: int,
    page_size: int,
) -> list[dict]:
    """Build RMSNorm static candidates from HF config dimensions."""
    params = resolve_rmsnorm_params(raw_params=dims)
    if not params or not analysis_allows_family(analysis, "has_rmsnorm"):
        return []
    return [
        {
            "op_type": OP_TYPE,
            "variant": variant,
            "fi_api": fi_api_for_variant(variant),
            "params": params,
            "definition_name": definition_name(variant, params),
        }
        for variant in ("rmsnorm", "fused_add_rmsnorm")
    ]


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    kernels: list[dict] = []
    for key in [k for k in matched_kernels if k.startswith(f"{OP_TYPE}:")]:
        info = matched_kernels[key]
        hidden_sizes = extract_rmsnorm_hidden_sizes(info["signatures"])
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        emitted = False
        for hidden_size in sorted(hidden_sizes):
            variant = info["variant"]
            params = resolve_rmsnorm_params(raw_params={"hidden_size": hidden_size})
            if not params:
                continue
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
            variant = info["variant"]
            kernels.append({
                "op_type": OP_TYPE,
                "variant": variant,
                "fi_api": info["fi_api"],
                "params": {},
                "definition_name": f"{variant}_NEEDS_CONFIG",
                **observation_fields(info, obs_kw, obs_pn),
                "note": "Definition name requires observed hidden_size or HF config params.",
            })
    return kernels


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    """Generate an RMSNorm definition, dispatching on variant."""
    variant = kernel["variant"]
    if "fused_add" in variant:
        return _gen_fused_add_rmsnorm(kernel, model_tag, tp)
    return _gen_rmsnorm(kernel, model_tag, tp)


def _gen_rmsnorm(kernel: dict, model_tag: str, tp: int) -> dict:
    """Generate an RMSNorm definition (rmsnorm or gemma_rmsnorm)."""
    p = kernel["params"]
    h = p["hidden_size"]
    variant = kernel["variant"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]

    is_gemma = variant.startswith("gemma")
    template_name = "gemma_rmsnorm_trace" if is_gemma else "rmsnorm_trace"
    import torch

    official = render_definition(
        "norm",
        template_name,
        fi_api,
        {
            "input": torch.empty((1, h), dtype=torch.bfloat16),
            "weight": torch.empty((h,), dtype=torch.bfloat16),
        },
        name=name,
    )
    if official is not None:
        _fill_rmsnorm_descriptions(official, fused=False, gemma=is_gemma)
        return official

    return None


def _gen_fused_add_rmsnorm(kernel: dict, model_tag: str, tp: int) -> dict:
    """Generate a fused_add_rmsnorm definition (with or without gemma prefix)."""
    p = kernel["params"]
    h = p["hidden_size"]
    variant = kernel["variant"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]

    is_gemma = variant.startswith("gemma")
    template_name = (
        "gemma_fused_add_rmsnorm_trace" if is_gemma else "fused_add_rmsnorm_trace"
    )
    import torch

    official = render_definition(
        "norm",
        template_name,
        fi_api,
        {
            "input": torch.empty((1, h), dtype=torch.bfloat16),
            "residual": torch.empty((1, h), dtype=torch.bfloat16),
            "weight": torch.empty((h,), dtype=torch.bfloat16),
        },
        name=name,
    )
    if official is not None:
        _fill_rmsnorm_descriptions(official, fused=True, gemma=is_gemma)
        return official

    return None


def _build_rmsnorm_solution_source(api: str, args: list[str], *, fused: bool) -> str:
    hidden_arg = args[0]
    signature = ", ".join(args)
    call_args = ", ".join(args)
    if fused:
        return textwrap.dedent(f"""\
            import importlib

            _FN = getattr(importlib.import_module({api.rpartition('.')[0]!r}), {api.rpartition('.')[2]!r})
            _EPS = 1e-6


            def run({signature}):
                # FlashInfer fused RMSNorm kernels update the activation tensor in-place.
                _FN({call_args}, _EPS)
                return {hidden_arg}
            """) + "\n"

    return textwrap.dedent(f"""\
        import importlib

        _FN = getattr(importlib.import_module({api.rpartition('.')[0]!r}), {api.rpartition('.')[2]!r})
        _EPS = 1e-6


        def run({signature}):
            return _FN({call_args}, eps=_EPS)
        """) + "\n"


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate FlashInfer RMSNorm baseline solution source."""
    if definition.get("op_type") != OP_TYPE:
        return None
    names = input_names(definition)
    if not names:
        return None
    def_name = str(definition.get("name", ""))
    if def_name.startswith("gemma_fused_add_rmsnorm"):
        api = "flashinfer.norm.gemma_fused_add_rmsnorm"
        fused = True
    elif def_name.startswith("fused_add_rmsnorm"):
        api = "flashinfer.norm.fused_add_rmsnorm"
        fused = True
    elif def_name.startswith("gemma_rmsnorm"):
        api = "flashinfer.norm.gemma_rmsnorm"
        fused = False
    elif def_name.startswith("rmsnorm"):
        api = "flashinfer.norm.rmsnorm"
        fused = False
    else:
        return None
    source = _build_rmsnorm_solution_source(api, names, fused=fused)
    return solution_payload(
        definition,
        source,
        description=f"Baseline solution using FlashInfer {api}.",
    )
