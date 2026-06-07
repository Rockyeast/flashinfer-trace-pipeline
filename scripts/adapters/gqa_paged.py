"""Adapter for FlashInfer GQA paged attention kernels."""

from __future__ import annotations

import textwrap
from typing import Any

from ._param_utils import analysis_allows_family, extract_hf_dims, first_int
from adapters.official_templates import render_definition
from ._solution_utils import has_inputs, solution_payload
from .extractors import (
    extract_attention_param_sets,
    extract_attention_params,
    extract_observed_kwargs,
    extract_observed_param_names,
)
from parse.inventory_helpers import observation_fields

OP_TYPE = "gqa_paged"


def pr_reference_source() -> str:
    return "FlashInfer attention unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} GQA paged attention"


def pr_baseline_description(_def_name: str) -> str:
    return "flashinfer GQA paged attention wrapper"


_WRAPPER_METHODS = {
    "__call__",
    "forward",
    "forward_decode",
    "forward_extend",
    "forward_return_lse",
    "run",
}

_CLASS_VARIANTS = {
    "flashinfer.BatchDecodeWithPagedKVCacheWrapper": "decode",
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper": "decode",
    "flashinfer.BatchPrefillWithPagedKVCacheWrapper": "prefill",
    "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper": "prefill",
}

_RUNTIME_ONLY_KWARGS = {
    "k_scale",
    "logits_soft_cap",
    "q_scale",
    "v_scale",
}

_WRAPPER_HIDDEN_INPUTS = {
    "k_cache",
    "kv_indices",
    "kv_indptr",
    "v_cache",
}


def ignored_observed_kwargs(_kernel: dict, _definition: dict) -> set[str]:
    """Runtime controls accepted by SGLang wrappers but not modeled here."""
    return set(_RUNTIME_ONLY_KWARGS)


def ignored_definition_inputs_for_observed_params(
    kernel: dict,
    _definition: dict,
) -> set[str]:
    """Official template inputs hidden inside wrapper-owned paged KV state."""
    observed = set(kernel.get("observed_param_names") or [])
    if "paged_kv_cache" not in observed:
        return set()
    return set(_WRAPPER_HIDDEN_INPUTS)


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


def definition_name(variant: str, params: dict[str, Any]) -> str:
    """Return the canonical GQA paged definition name for resolved params."""
    q = params["num_q_heads"]
    kv = params["num_kv_heads"]
    d = params["head_dim"]
    ps = params["page_size"]
    if variant == "decode":
        return f"gqa_paged_decode_h{q}_kv{kv}_d{d}_ps{ps}"
    if variant == "prefill":
        return f"gqa_paged_prefill_h{q}_kv{kv}_d{d}_ps{ps}"
    raise ValueError(f"unknown GQA paged variant: {variant}")


def fi_api_for_variant(variant: str) -> str | None:
    """Return the FlashInfer API tag used for static candidates."""
    return {
        "decode": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper",
        "prefill": "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper",
    }.get(variant)


def resolve_gqa_params(
    *,
    raw_params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    tp: int | None = None,
    page_size: int | None = None,
    paged: bool = True,
) -> dict[str, int] | None:
    """Resolve canonical GQA params owned by this adapter."""
    raw = raw_params or {}
    dims = extract_hf_dims(config or {}, tp)
    q = first_int(
        raw.get("tp_q_head_num"),
        raw.get("num_q_heads"),
        raw.get("num_heads"),
        raw.get("tp_num_attention_heads"),
        dims.get("tp_num_attention_heads"),
    )
    kv = first_int(
        raw.get("tp_k_head_num"),
        raw.get("num_kv_heads"),
        raw.get("tp_num_key_value_heads"),
        dims.get("tp_num_key_value_heads"),
    )
    d = first_int(raw.get("head_dim"), raw.get("head_size"), dims.get("head_dim"))
    if not (q and kv and d):
        return None
    params = {
        "num_q_heads": q,
        "num_kv_heads": kv,
        "head_dim": d,
    }
    if paged:
        ps = first_int(raw.get("page_size"), page_size)
        if not ps:
            return None
        params["page_size"] = ps
    return params


def resolve_config_params(
    kernel: dict,
    config: dict,
    tp_size: int | None,
) -> dict | None:
    """Resolve missing GQA paged params from HF config."""
    return resolve_gqa_params(
        raw_params=kernel.get("params"),
        config=config,
        tp=tp_size,
        page_size=None,
        paged=True,
    )


def static_candidates(
    dims: dict,
    analysis: dict,
    *,
    tp: int,
    page_size: int,
) -> list[dict]:
    """Build GQA paged static candidates from HF config dimensions."""
    params = resolve_gqa_params(raw_params=dims, page_size=page_size, paged=True)
    if not params or not analysis_allows_family(analysis, "has_gqa"):
        return []
    return [
        {
            "op_type": OP_TYPE,
            "variant": variant,
            "fi_api": fi_api_for_variant(variant),
            "params": params,
            "definition_name": definition_name(variant, params),
        }
        for variant in ("decode", "prefill")
    ]


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    kernels: list[dict] = []
    for key in [k for k in matched_kernels if k.startswith(f"{OP_TYPE}:")]:
        info = matched_kernels[key]
        variant = info["variant"]
        param_sets = extract_attention_param_sets(info["signatures"])
        if not param_sets:
            param_sets = [extract_attention_params(info["signatures"])]
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        for attn_params in param_sets:
            page_size = attn_params.get("page_size")
            suffix = f"_ps{page_size}" if isinstance(page_size, int) else ""
            kernels.append({
                "op_type": OP_TYPE,
                "variant": variant,
                "fi_api": info["fi_api"],
                "params": attn_params,
                "definition_name": f"gqa_paged_{variant}{suffix}_NEEDS_CONFIG",
                **observation_fields(info, obs_kw, obs_pn),
                "note": (
                    "Definition name requires HF config.json params "
                    "(num_heads, head_dim, etc.)"
                ),
            })
    return kernels


def _try_official_template(
    *,
    name: str,
    fi_api: str,
    variant: str,
    q_heads: Any,
    kv_heads: Any,
    head_dim: Any,
    page_size: Any,
) -> dict | None:
    """Render the GQA paged definition from FlashInfer's official TraceTemplate."""
    if not all(isinstance(v, int) for v in (q_heads, kv_heads, head_dim, page_size)):
        return None

    import torch

    q = torch.empty((1, q_heads, head_dim), dtype=torch.bfloat16)
    k_cache = torch.empty((1, page_size, kv_heads, head_dim), dtype=torch.bfloat16)
    v_cache = torch.empty((1, page_size, kv_heads, head_dim), dtype=torch.bfloat16)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    kv_indices = torch.tensor([0], dtype=torch.int32)
    kwargs: dict[str, Any] = {
        "q": q,
        "paged_kv_cache": (k_cache, v_cache),
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "sm_scale": float(head_dim) ** -0.5,
    }

    if variant == "decode":
        template_name = "gqa_paged_decode_trace"
    else:
        template_name = "gqa_paged_prefill_trace"
        kwargs["qo_indptr"] = torch.tensor([0, 1], dtype=torch.int32)

    definition = render_definition(
        "attention",
        template_name,
        fi_api,
        kwargs,
        name=name,
    )
    return definition


def _fill_gqa_paged_descriptions(definition: dict) -> None:
    """Fill validation descriptions omitted by the upstream TraceTemplate."""
    axis_descriptions = {
        "num_qo_heads": "Number of query/output attention heads.",
        "num_kv_heads": "Number of key/value attention heads.",
        "head_dim": "Per-head hidden dimension.",
        "num_pages": "Number of KV cache pages available to the kernel.",
        "page_size": "Number of KV tokens stored per page.",
    }
    input_descriptions = {
        "q": "Query tensor.",
        "k_cache": "Paged key cache.",
        "v_cache": "Paged value cache.",
    }
    output_descriptions = {
        "output": "Attention output tensor.",
    }

    for name, description in axis_descriptions.items():
        definition.get("axes", {}).get(name, {}).setdefault("description", description)
    for name, description in input_descriptions.items():
        definition.get("inputs", {}).get(name, {}).setdefault("description", description)
    for name, description in output_descriptions.items():
        definition.get("outputs", {}).get(name, {}).setdefault("description", description)


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    """Generate a GQA paged attention definition (decode or prefill)."""
    p = kernel["params"]
    variant = kernel["variant"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]
    if fi_api and not fi_api.endswith(".run"):
        fi_api = f"{fi_api}.run"

    q_heads = p.get("num_q_heads", p.get("num_heads", "?"))
    kv_heads = p.get("num_kv_heads", "?")
    head_dim = p.get("head_dim", "?")
    page_size = p.get("page_size", 1)

    official = _try_official_template(
        name=name,
        fi_api=fi_api,
        variant=variant,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    if official is not None:
        _fill_gqa_paged_descriptions(official)
        return official
    return None


def _gqa_solution_payload(definition: dict, source: str, variant: str) -> dict:
    """Build a baseline solution JSON for FlashInfer GQA paged wrappers."""
    wrapper = (
        "BatchDecodeWithPagedKVCacheWrapper"
        if variant == "decode"
        else "BatchPrefillWithPagedKVCacheWrapper"
    )
    return solution_payload(
        definition,
        source,
        description=f"Solution using FlashInfer {wrapper}.",
    )


def _build_gqa_decode_solution_source() -> str:
    """Build FlashInfer wrapper baseline source for paged decode."""
    return textwrap.dedent("""\
        import torch
        import flashinfer

        _WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
        _workspace_cache = {}
        _wrapper_cache = {}
        _plan_state = {}


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
                wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                )
                _wrapper_cache[key] = wrapper
            return wrapper


        def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
            batch_size, num_qo_heads, head_dim = q.shape
            _, page_size, num_kv_heads, _ = k_cache.shape
            len_indptr = kv_indptr.shape[0]
            num_kv_indices = kv_indices.shape[0]

            device = q.device
            wrapper_key = (
                str(device),
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                q.dtype,
                k_cache.dtype,
            )

            if isinstance(sm_scale, torch.Tensor):
                sm_scale_value = float(sm_scale.item())
            else:
                sm_scale_value = float(sm_scale)

            wrapper = _get_wrapper(wrapper_key, device)
            state = _plan_state.get(wrapper_key)

            needs_plan = True
            if state is not None:
                needs_plan = (
                    state.get("batch_size") != batch_size
                    or state.get("len_indptr") != len_indptr
                    or state.get("num_kv_indices") != num_kv_indices
                    or state.get("sm_scale") != sm_scale_value
                    or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
                    or state.get("kv_indices_ptr") != kv_indices.data_ptr()
                )

            if needs_plan:
                kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
                wrapper.plan(
                    indptr=kv_indptr,
                    indices=kv_indices,
                    last_page_len=kv_last_page_len,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    pos_encoding_mode="NONE",
                    q_data_type=q.dtype,
                    kv_data_type=k_cache.dtype,
                    sm_scale=sm_scale_value,
                )
                _plan_state[wrapper_key] = {
                    "batch_size": batch_size,
                    "len_indptr": len_indptr,
                    "num_kv_indices": num_kv_indices,
                    "sm_scale": sm_scale_value,
                    "kv_indptr_ptr": kv_indptr.data_ptr(),
                    "kv_indices_ptr": kv_indices.data_ptr(),
                }

            output, lse = wrapper.run(
                q,
                (k_cache, v_cache),
                return_lse=True,
            )

            return output, lse
        """) + "\n"


def _build_gqa_decode_with_last_page_solution_source() -> str:
    """Build FlashInfer wrapper baseline source for paged decode with last-page lengths."""
    return textwrap.dedent("""\
        import torch
        import flashinfer

        _WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
        _workspace_cache = {}
        _wrapper_cache = {}
        _plan_state = {}


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
                wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                )
                _wrapper_cache[key] = wrapper
            return wrapper


        def run(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
            batch_size, num_qo_heads, head_dim = q.shape
            _, page_size, num_kv_heads, _ = k_cache.shape
            len_indptr = kv_indptr.shape[0]
            num_kv_indices = kv_indices.shape[0]

            device = q.device
            wrapper_key = (
                str(device),
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                q.dtype,
                k_cache.dtype,
            )

            if isinstance(sm_scale, torch.Tensor):
                sm_scale_value = float(sm_scale.item())
            else:
                sm_scale_value = float(sm_scale)

            wrapper = _get_wrapper(wrapper_key, device)
            state = _plan_state.get(wrapper_key)

            needs_plan = True
            if state is not None:
                needs_plan = (
                    state.get("batch_size") != batch_size
                    or state.get("len_indptr") != len_indptr
                    or state.get("num_kv_indices") != num_kv_indices
                    or state.get("sm_scale") != sm_scale_value
                    or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
                    or state.get("kv_indices_ptr") != kv_indices.data_ptr()
                    or state.get("kv_last_page_len_ptr") != kv_last_page_len.data_ptr()
                )

            if needs_plan:
                wrapper.plan(
                    indptr=kv_indptr,
                    indices=kv_indices,
                    last_page_len=kv_last_page_len,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    pos_encoding_mode="NONE",
                    q_data_type=q.dtype,
                    kv_data_type=k_cache.dtype,
                    sm_scale=sm_scale_value,
                )
                _plan_state[wrapper_key] = {
                    "batch_size": batch_size,
                    "len_indptr": len_indptr,
                    "num_kv_indices": num_kv_indices,
                    "sm_scale": sm_scale_value,
                    "kv_indptr_ptr": kv_indptr.data_ptr(),
                    "kv_indices_ptr": kv_indices.data_ptr(),
                    "kv_last_page_len_ptr": kv_last_page_len.data_ptr(),
                }

            output, lse = wrapper.run(
                q,
                (k_cache, v_cache),
                return_lse=True,
            )

            return output, lse
        """) + "\n"


def _build_gqa_prefill_solution_source() -> str:
    """Build FlashInfer wrapper baseline source for paged prefill."""
    return textwrap.dedent("""\
        import torch
        import flashinfer

        _WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
        _workspace_cache = {}
        _wrapper_cache = {}
        _plan_state = {}


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
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                )
                _wrapper_cache[key] = wrapper
            return wrapper


        def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
            total_q, num_qo_heads, head_dim = q.shape
            _, page_size, num_kv_heads, _ = k_cache.shape
            batch_size = qo_indptr.shape[0] - 1
            num_kv_indices = kv_indices.shape[0]

            device = q.device
            wrapper_key = (
                str(device),
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                q.dtype,
                k_cache.dtype,
            )

            wrapper = _get_wrapper(wrapper_key, device)
            state = _plan_state.get(wrapper_key)

            if isinstance(sm_scale, torch.Tensor):
                sm_scale_value = float(sm_scale.item())
            else:
                sm_scale_value = float(sm_scale)

            needs_plan = True
            if state is not None:
                needs_plan = (
                    state.get("total_q") != total_q
                    or state.get("batch_size") != batch_size
                    or state.get("num_kv_indices") != num_kv_indices
                    or state.get("sm_scale") != sm_scale_value
                    or state.get("qo_indptr_ptr") != qo_indptr.data_ptr()
                    or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
                    or state.get("kv_indices_ptr") != kv_indices.data_ptr()
                )

            if needs_plan:
                kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
                wrapper.plan(
                    qo_indptr=qo_indptr,
                    paged_kv_indptr=kv_indptr,
                    paged_kv_indices=kv_indices,
                    paged_kv_last_page_len=kv_last_page_len,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    page_size=page_size,
                    causal=True,
                    sm_scale=sm_scale_value,
                    q_data_type=q.dtype,
                    kv_data_type=k_cache.dtype,
                )
                _plan_state[wrapper_key] = {
                    "total_q": total_q,
                    "batch_size": batch_size,
                    "num_kv_indices": num_kv_indices,
                    "sm_scale": sm_scale_value,
                    "qo_indptr_ptr": qo_indptr.data_ptr(),
                    "kv_indptr_ptr": kv_indptr.data_ptr(),
                    "kv_indices_ptr": kv_indices.data_ptr(),
                }

            output, lse = wrapper.run(
                q,
                (k_cache, v_cache),
                return_lse=True,
            )

            return output, lse
        """) + "\n"


def _build_gqa_prefill_with_last_page_solution_source() -> str:
    """Build FlashInfer wrapper baseline source for paged prefill with last-page lengths."""
    return textwrap.dedent("""\
        import torch
        import flashinfer

        _WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
        _workspace_cache = {}
        _wrapper_cache = {}


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
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                )
                _wrapper_cache[key] = wrapper
            return wrapper


        def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
            _total_q, num_qo_heads, head_dim = q.shape
            _, page_size, num_kv_heads, _ = k_cache.shape
            device = q.device
            wrapper_key = (
                str(device),
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                q.dtype,
                k_cache.dtype,
            )
            sm_scale_value = float(sm_scale.item()) if isinstance(sm_scale, torch.Tensor) else float(sm_scale)
            wrapper = _get_wrapper(wrapper_key, device)
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=kv_last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=True,
                sm_scale=sm_scale_value,
                q_data_type=q.dtype,
                kv_data_type=k_cache.dtype,
            )
            output, lse = wrapper.run(
                q,
                (k_cache, v_cache),
                return_lse=True,
            )
            return output, lse
        """) + "\n"


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate a FlashInfer wrapper baseline solution for GQA paged attention."""
    if definition.get("op_type") != OP_TYPE:
        return None
    name = str(definition.get("name", ""))
    tags = definition.get("tags", [])
    tag_text = " ".join(tag for tag in tags if isinstance(tag, str))

    if "decode" in name or "stage:decode" in tag_text:
        if not has_inputs(
            definition,
            {"q", "k_cache", "v_cache", "kv_indptr", "kv_indices", "sm_scale"},
        ):
            return None
        inputs = definition.get("inputs", {})
        if isinstance(inputs, dict) and "kv_last_page_len" in inputs:
            source = _build_gqa_decode_with_last_page_solution_source()
        else:
            source = _build_gqa_decode_solution_source()
        return _gqa_solution_payload(definition, source, "decode")
    if "prefill" in name or "stage:prefill" in tag_text:
        if not has_inputs(
            definition,
            {
                "q",
                "k_cache",
                "v_cache",
                "qo_indptr",
                "kv_indptr",
                "kv_indices",
                "sm_scale",
            },
        ):
            return None
        inputs = definition.get("inputs", {})
        if isinstance(inputs, dict) and "kv_last_page_len" in inputs:
            source = _build_gqa_prefill_with_last_page_solution_source()
        else:
            source = _build_gqa_prefill_solution_source()
        return _gqa_solution_payload(definition, source, "prefill")
    return None
