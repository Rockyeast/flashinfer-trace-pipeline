"""Adapter for FlashInfer GQA ragged attention kernels."""
from __future__ import annotations

import sys
import textwrap

from ._param_utils import analysis_allows_family
from .gqa_paged import resolve_gqa_params
from adapters.official_templates import render_definition
from ._solution_utils import has_inputs, solution_payload
from .extractors import (
    extract_attention_params,
    extract_observed_kwargs,
    extract_observed_param_names,
)
from parse.inventory_helpers import observation_fields

OP_TYPE = "gqa_ragged"


def pr_reference_source() -> str:
    return "FlashInfer attention unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} GQA ragged attention"


def pr_baseline_description(_def_name: str) -> str:
    return "flashinfer GQA ragged attention wrapper"


_AXIS_DESCRIPTIONS = {
    "num_qo_heads": "Number of query/output attention heads.",
    "num_kv_heads": "Number of key/value attention heads.",
    "head_dim": "Per-head hidden dimension.",
}
_INPUT_DESCRIPTIONS = {
    "q": "Ragged query tensor.",
    "k": "Ragged key tensor.",
    "v": "Ragged value tensor.",
}

_WRAPPER_METHODS = {
    "__call__",
    "forward",
    "forward_decode",
    "forward_extend",
    "forward_return_lse",
    "run",
}

_CLASS_VARIANTS = {
    "flashinfer.BatchPrefillWithRaggedKVCacheWrapper": "prefill",
    "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper": "prefill",
}

_RUNTIME_ONLY_KWARGS = {
    "causal",
    "logits_soft_cap",
}

_WRAPPER_HIDDEN_INPUTS = {
    "kv_indptr",
    "qo_indptr",
}


def ignored_observed_kwargs(_kernel: dict, _definition: dict) -> set[str]:
    """Runtime controls accepted by SGLang wrappers but not modeled here."""
    return set(_RUNTIME_ONLY_KWARGS)


def ignored_definition_inputs_for_observed_params(
    kernel: dict,
    _definition: dict,
) -> set[str]:
    """Official template inputs hidden inside ragged wrapper planning state."""
    observed = set(kernel.get("observed_param_names") or [])
    if "args" not in observed:
        return set()
    return set(_WRAPPER_HIDDEN_INPUTS)


def _fill_gqa_ragged_descriptions(definition: dict) -> None:
    """Fill descriptions used by existing GQA ragged dataset definitions."""
    for name, description in _AXIS_DESCRIPTIONS.items():
        definition.get("axes", {}).get(name, {}).setdefault("description", description)
    for name, description in _INPUT_DESCRIPTIONS.items():
        definition.get("inputs", {}).get(name, {}).setdefault("description", description)
    for name in ("qo_indptr", "kv_indptr"):
        input_spec = definition.get("inputs", {}).get(name)
        if isinstance(input_spec, dict) and input_spec.get("dtype") == "unknown":
            input_spec["dtype"] = "int32"


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    if trace_id in _CLASS_VARIANTS:
        return (OP_TYPE, _CLASS_VARIANTS[trace_id])

    class_path, sep, method = trace_id.rpartition(".")
    if not sep or method not in _WRAPPER_METHODS:
        return None
    variant = _CLASS_VARIANTS.get(class_path)
    if variant is None:
        return None
    return (OP_TYPE, variant)


def definition_name(_variant: str, params: dict) -> str:
    """Return the canonical GQA ragged definition name for resolved params."""
    q = params["num_q_heads"]
    kv = params["num_kv_heads"]
    d = params["head_dim"]
    return f"gqa_ragged_h{q}_kv{kv}_d{d}"


def fi_api_for_variant(variant: str) -> str | None:
    """Return the FlashInfer API tag used for static candidates."""
    return {
        "prefill": "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper",
    }.get(variant)


def resolve_config_params(
    kernel: dict,
    config: dict,
    tp_size: int | None,
) -> dict | None:
    """Resolve missing GQA ragged params from HF config."""
    return resolve_gqa_params(
        raw_params=kernel.get("params"),
        config=config,
        tp=tp_size,
        paged=False,
    )


def static_candidates(
    dims: dict,
    analysis: dict,
    *,
    tp: int,
    page_size: int,
) -> list[dict]:
    """Build GQA ragged static candidates from HF config dimensions."""
    params = resolve_gqa_params(raw_params=dims, paged=False)
    if not params or not analysis_allows_family(analysis, "has_gqa"):
        return []
    variant = "prefill"
    return [{
        "op_type": OP_TYPE,
        "variant": variant,
        "fi_api": fi_api_for_variant(variant),
        "params": params,
        "definition_name": definition_name(variant, params),
    }]


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    kernels: list[dict] = []
    paged_prefill_count = matched_kernels.get("gqa_paged:prefill", {}).get(
        "total_count", 0
    )
    paged_decode_count = matched_kernels.get("gqa_paged:decode", {}).get(
        "total_count", 0
    )
    dominant_paged_count = max(paged_prefill_count, paged_decode_count)

    for key in [k for k in matched_kernels if k.startswith(f"{OP_TYPE}:")]:
        info = matched_kernels[key]
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        ragged_count = info["total_count"]
        if dominant_paged_count >= 10 * ragged_count:
            print(
                f"  ⚠️  gqa_ragged count={ragged_count} is much lower than "
                f"dominant gqa_paged count={dominant_paged_count} "
                f"({dominant_paged_count // max(ragged_count, 1)}x ratio), "
                f"may be warmup artifact. Definition will still be generated.",
                file=sys.stderr,
            )
        kernels.append({
            "op_type": OP_TYPE,
            "variant": "prefill",
            "fi_api": info["fi_api"],
            "params": extract_attention_params(info["signatures"]),
            "definition_name": "gqa_ragged_prefill_NEEDS_CONFIG",
            **observation_fields(info, obs_kw, obs_pn),
            "note": "Definition name requires HF config.json params.",
        })
    return kernels


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    """Generate a GQA ragged attention definition."""
    p = kernel["params"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]

    q_heads = p.get("num_q_heads", p.get("num_heads", "?"))
    kv_heads = p.get("num_kv_heads", "?")
    head_dim = p.get("head_dim", "?")
    fi_api_tag = fi_api if not fi_api or fi_api.endswith(".run") else f"{fi_api}.run"
    if all(isinstance(x, int) for x in (q_heads, kv_heads, head_dim)):
        import torch

        official = render_definition(
            "attention",
            "gqa_ragged_prefill_trace",
            fi_api_tag,
            {
                "q": torch.empty((1, q_heads, head_dim), dtype=torch.bfloat16),
                "k": torch.empty((1, kv_heads, head_dim), dtype=torch.bfloat16),
                "v": torch.empty((1, kv_heads, head_dim), dtype=torch.bfloat16),
            },
            name=name,
        )
        if official is not None:
            _fill_gqa_ragged_descriptions(official)
            return official

    return None


def _build_gqa_ragged_solution_source() -> str:
    """Build FlashInfer wrapper baseline source for ragged prefill."""
    return textwrap.dedent("""\
        import torch
        import flashinfer

        _WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
        _workspace_cache = {}
        _wrapper_cache = {}
        _plan_state = {}


        def _scalar(value):
            if isinstance(value, torch.Tensor):
                return float(value.item())
            return float(value)


        def _get_workspace(device):
            key = str(device)
            buffer = _workspace_cache.get(key)
            if buffer is None or buffer.device != device or buffer.numel() < _WORKSPACE_SIZE_BYTES:
                buffer = torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
                _workspace_cache[key] = buffer
            return buffer


        def _get_wrapper(key, device):
            wrapper = _wrapper_cache.get(key)
            if wrapper is None:
                workspace = _get_workspace(device)
                wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                )
                _wrapper_cache[key] = wrapper
            return wrapper


        def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
            total_q, num_qo_heads, head_dim = q.shape
            total_kv, num_kv_heads, _ = k.shape
            batch_size = qo_indptr.shape[0] - 1
            sm_scale_value = _scalar(sm_scale)

            device = q.device
            wrapper_key = (
                str(device),
                num_qo_heads,
                num_kv_heads,
                head_dim,
                q.dtype,
                k.dtype,
                v.dtype,
            )

            wrapper = _get_wrapper(wrapper_key, device)
            state = _plan_state.get(wrapper_key)

            needs_plan = True
            if state is not None:
                needs_plan = (
                    state.get("total_q") != total_q
                    or state.get("total_kv") != total_kv
                    or state.get("batch_size") != batch_size
                    or state.get("sm_scale") != sm_scale_value
                    or state.get("qo_indptr_ptr") != qo_indptr.data_ptr()
                    or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
                )

            if needs_plan:
                wrapper.plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    causal=True,
                    sm_scale=sm_scale_value,
                    q_data_type=q.dtype,
                    kv_data_type=k.dtype,
                )
                _plan_state[wrapper_key] = {
                    "total_q": total_q,
                    "total_kv": total_kv,
                    "batch_size": batch_size,
                    "sm_scale": sm_scale_value,
                    "qo_indptr_ptr": qo_indptr.data_ptr(),
                    "kv_indptr_ptr": kv_indptr.data_ptr(),
                }

            output, lse = wrapper.run(
                q,
                k,
                v,
                return_lse=True,
            )

            return output, lse
        """) + "\n"


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate a FlashInfer wrapper baseline solution for GQA ragged attention."""
    if definition.get("op_type") != OP_TYPE:
        return None
    if not has_inputs(definition, {"q", "k", "v", "qo_indptr", "kv_indptr", "sm_scale"}):
        return None
    source = _build_gqa_ragged_solution_source()
    return solution_payload(
        definition,
        source,
        description="Baseline solution using FlashInfer BatchPrefillWithRaggedKVCacheWrapper.",
    )
