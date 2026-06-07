"""Adapter for FlashInfer MLA paged attention kernels."""

from __future__ import annotations

import textwrap
from typing import Any

from ._param_utils import analysis_allows_family, extract_hf_dims, first_int
from adapters.official_templates import render_definition
from ._solution_utils import has_inputs, input_names, solution_payload
from .extractors import extract_observed_kwargs, extract_observed_param_names
from parse.inventory_helpers import observation_fields

OP_TYPE = "mla_paged"


def pr_reference_source() -> str:
    return "FlashInfer MLA unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} MLA paged attention"


def pr_baseline_description(_def_name: str) -> str:
    return "flashinfer MLA paged attention wrapper"


_WRAPPER_METHODS = {
    "__call__",
    "forward",
    "forward_decode",
    "forward_extend",
    "forward_return_lse",
    "run",
}

_CLASS_VARIANTS = {
    "flashinfer.decode.BatchDecodeMlaWithPagedKVCacheWrapper": "decode",
    "flashinfer.prefill.BatchPrefillMlaWithPagedKVCacheWrapper": "prefill",
    "flashinfer.mla.BatchMLAPagedDecodeWrapper": "decode",
    "flashinfer.mla.BatchMLAPagedPrefillWrapper": "prefill",
}

_SPECIAL_TRACE_IDS = {
    "flashinfer.mla._core.trtllm_batch_decode_with_kv_cache_mla": (
        "trtllm_decode",
        "trtllm_batch_decode_mla_trace",
    ),
    "flashinfer.mla._core.xqa_batch_decode_with_kv_cache_mla": (
        "xqa_decode",
        "xqa_batch_decode_mla_trace",
    ),
    "flashinfer.cute_dsl.attention.wrappers.batch_mla.BatchMLADecodeCuteDSLWrapper.run": (
        "cute_dsl_decode",
        "cute_dsl_batch_mla_run_trace",
    ),
}

_SPECIAL_VARIANTS = {variant for variant, _template_name in _SPECIAL_TRACE_IDS.values()}


def _all_ints(*values) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def _meta(shape, dtype):
    import torch

    return torch.empty(tuple(int(v) for v in shape), dtype=dtype, device="meta")


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    special = _SPECIAL_TRACE_IDS.get(trace_id)
    if special is not None:
        return (OP_TYPE, special[0])

    if trace_id in _CLASS_VARIANTS:
        return (OP_TYPE, _CLASS_VARIANTS[trace_id])

    class_path, sep, method = trace_id.rpartition(".")
    if not sep or method not in _WRAPPER_METHODS:
        return None
    variant = _CLASS_VARIANTS.get(class_path)
    if variant is None:
        return None
    return (OP_TYPE, variant)


def _shape(value: Any) -> list | None:
    if isinstance(value, dict) and value.get("type") == "tensor":
        shape = value.get("shape")
        if isinstance(shape, list):
            return shape
    return None


def _scalar_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        raw = value.get("value")
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _scalar_int(value)
        if parsed is not None:
            return parsed
    return None


def _bound_signature_args(sig: dict) -> dict:
    body = sig.get("signature", sig)
    if not isinstance(body, dict):
        return {}
    args = body.get("args", [])
    kwargs = body.get("kwargs", {})
    names = sig.get("param_names") or body.get("param_names") or []
    bound: dict[str, Any] = {}
    if isinstance(names, list) and isinstance(args, list):
        for name, arg in zip(names, args):
            if isinstance(name, str):
                bound[name] = arg
    if isinstance(kwargs, dict):
        bound.update(kwargs)
    return bound


def _self_attrs(sig: dict) -> dict:
    body = sig.get("signature", sig)
    if not isinstance(body, dict):
        return {}
    self_info = body.get("self")
    if isinstance(self_info, dict):
        attrs = self_info.get("attrs")
        if isinstance(attrs, dict):
            return attrs
    return {}


def _kv_cache_page_size(shape: list | None) -> int | None:
    if not shape:
        return None
    if len(shape) == 3:
        return _first_int(shape[1])
    if len(shape) == 4:
        return _first_int(shape[2])
    return None


def _special_variant_info(info: dict) -> tuple[str, str, str] | None:
    for trace_id in info.get("trace_ids", []):
        mapped = _SPECIAL_TRACE_IDS.get(trace_id)
        if mapped is not None:
            variant, template_name = mapped
            return variant, template_name, trace_id
    variant = info.get("variant")
    for trace_id, (mapped_variant, template_name) in _SPECIAL_TRACE_IDS.items():
        if mapped_variant == variant:
            return mapped_variant, template_name, trace_id
    return None


def _special_definition_name(variant: str, p: dict[str, int]) -> str:
    h = p["num_heads"]
    d_qk = p["head_dim_qk"]
    page_size = p["page_size"]
    if variant == "trtllm_decode":
        return (
            f"trtllm_batch_decode_mla_h{h}_d_qk{d_qk}_ckv{p['kv_lora_rank']}"
            f"_kpe{p['qk_rope_head_dim']}_nope{p['qk_nope_head_dim']}_ps{page_size}"
        )
    if variant == "xqa_decode":
        return (
            f"xqa_batch_decode_mla_h{h}_d_qk{d_qk}_ckv{p['kv_lora_rank']}"
            f"_kpe{p['qk_rope_head_dim']}_nope{p['qk_nope_head_dim']}_ps{page_size}"
        )
    if variant == "cute_dsl_decode":
        return f"cute_dsl_batch_mla_run_h{h}_d_qk{d_qk}_ps{page_size}"
    raise ValueError(f"unknown MLA special variant: {variant}")


def definition_name(variant: str, params: dict[str, Any]) -> str:
    """Return the canonical MLA paged definition name for resolved params."""
    if variant in _SPECIAL_VARIANTS:
        return _special_definition_name(variant, params)
    ckv = params["ckv_dim"]
    kpe = params["kpe_dim"]
    h = params["num_heads"]
    ps = params["page_size"]
    if variant == "decode":
        return f"mla_paged_decode_h{h}_ckv{ckv}_kpe{kpe}_ps{ps}"
    if variant == "prefill":
        return f"mla_paged_prefill_h{h}_ckv{ckv}_kpe{kpe}_ps{ps}"
    raise ValueError(f"unknown MLA paged variant: {variant}")


def fi_api_for_variant(variant: str) -> str | None:
    """Return the FlashInfer API tag used for static candidates."""
    return {
        "decode": "flashinfer.decode.BatchDecodeMlaWithPagedKVCacheWrapper",
        "prefill": "flashinfer.prefill.BatchPrefillMlaWithPagedKVCacheWrapper",
    }.get(variant)


def resolve_mla_params(
    *,
    raw_params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    tp: int | None = None,
    page_size: int = 1,
) -> dict[str, int] | None:
    """Resolve canonical MLA paged params owned by this adapter."""
    raw = raw_params or {}
    dims = extract_hf_dims(config or {}, tp)
    raw_ckv = None
    raw_kv_lora = first_int(raw.get("kv_lora_rank"))
    raw_kpe = first_int(raw.get("qk_rope_head_dim"))
    if raw_kv_lora and raw_kpe:
        raw_ckv = raw_kv_lora + raw_kpe
    cfg_ckv = None
    if isinstance(dims.get("kv_lora_rank"), int) and isinstance(dims.get("qk_rope_head_dim"), int):
        cfg_ckv = dims["kv_lora_rank"] + dims["qk_rope_head_dim"]
    h = first_int(
        raw.get("num_heads"),
        raw.get("num_qo_heads"),
        raw.get("tp_q_head_num"),
        raw.get("tp_num_attention_heads"),
        dims.get("tp_num_attention_heads"),
        dims.get("num_attention_heads"),
    )
    ckv = first_int(
        raw.get("ckv_dim"),
        raw.get("head_dim_ckv"),
        raw_ckv,
        cfg_ckv,
    )
    kpe = first_int(
        raw.get("kpe_dim"),
        raw.get("head_dim_kpe"),
        raw.get("qk_rope_head_dim"),
        dims.get("qk_rope_head_dim"),
    )
    ps = first_int(raw.get("page_size"), page_size)
    if not (h and ckv and kpe and ps):
        return None
    return {"num_heads": h, "ckv_dim": ckv, "kpe_dim": kpe, "page_size": ps}


def resolve_config_params(
    kernel: dict,
    config: dict,
    tp_size: int | None,
) -> dict | None:
    """Resolve missing MLA paged params from HF config."""
    return resolve_mla_params(
        raw_params=kernel.get("params"),
        config=config,
        tp=tp_size,
        page_size=1,
    )


def static_candidates(
    dims: dict,
    analysis: dict,
    *,
    tp: int,
    page_size: int,
) -> list[dict]:
    """Build MLA paged static candidates from HF config dimensions."""
    params = resolve_mla_params(raw_params=dims, tp=tp, page_size=page_size)
    if not params or not analysis_allows_family(analysis, "has_mla"):
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


def _extract_special_params(signatures: list[dict], variant: str) -> list[dict]:
    params: list[dict] = []
    seen: set[tuple] = set()
    for sig in signatures:
        bound = _bound_signature_args(sig)
        attrs = _self_attrs(sig)

        if variant == "cute_dsl_decode":
            q_shape = _shape(bound.get("q"))
            kv_shape = _shape(bound.get("kv_cache"))
            num_heads = _first_int(
                attrs.get("_num_heads"),
                attrs.get("num_heads"),
                q_shape[-2] if q_shape and len(q_shape) >= 3 else None,
            )
            head_dim_qk = _first_int(q_shape[-1] if q_shape else None)
            page_size = _first_int(
                attrs.get("_page_size"),
                attrs.get("page_size"),
                _kv_cache_page_size(kv_shape),
            )
            p = {
                "num_heads": num_heads,
                "head_dim_qk": head_dim_qk,
                "page_size": page_size,
            }
        else:
            q_shape = _shape(bound.get("query"))
            kv_shape = _shape(bound.get("kv_cache"))
            kv_lora_rank = _first_int(
                bound.get("kv_lora_rank"),
                attrs.get("_kv_lora_rank"),
                attrs.get("kv_lora_rank"),
            )
            qk_rope_head_dim = _first_int(
                bound.get("qk_rope_head_dim"),
                attrs.get("_qk_rope_head_dim"),
                attrs.get("qk_rope_head_dim"),
            )
            qk_nope_head_dim = _first_int(
                bound.get("qk_nope_head_dim"),
                attrs.get("_qk_nope_head_dim"),
                attrs.get("qk_nope_head_dim"),
            )
            num_heads = _first_int(q_shape[-2] if q_shape and len(q_shape) >= 3 else None)
            head_dim_qk = _first_int(
                q_shape[-1] if q_shape else None,
                (
                    kv_lora_rank + qk_rope_head_dim
                    if kv_lora_rank is not None and qk_rope_head_dim is not None
                    else None
                ),
            )
            page_size = _kv_cache_page_size(kv_shape)
            p = {
                "num_heads": num_heads,
                "head_dim_qk": head_dim_qk,
                "kv_lora_rank": kv_lora_rank,
                "qk_rope_head_dim": qk_rope_head_dim,
                "qk_nope_head_dim": qk_nope_head_dim,
                "page_size": page_size,
            }

        p = {key: value for key, value in p.items() if value is not None}
        required = {"num_heads", "head_dim_qk", "page_size"}
        if variant != "cute_dsl_decode":
            required.update({"kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim"})
        if not required.issubset(p):
            continue
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        params.append(p)
    return params


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    kernels: list[dict] = []
    for key in [k for k in matched_kernels if k.startswith(f"{OP_TYPE}:")]:
        info = matched_kernels[key]
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        special = _special_variant_info(info)
        if special is not None:
            variant, template_name, fi_api_tag = special
            for params in _extract_special_params(info["signatures"], variant):
                params = {
                    **params,
                    "template_name": template_name,
                    "fi_api_tag": fi_api_tag,
                }
                kernels.append({
                    "op_type": OP_TYPE,
                    "variant": variant,
                    "fi_api": info["fi_api"],
                    "params": params,
                    "definition_name": definition_name(variant, params),
                    **observation_fields(info, obs_kw, obs_pn),
                })
            continue
        kernels.append({
            "op_type": OP_TYPE,
            "variant": info["variant"],
            "fi_api": info["fi_api"],
            "params": {},
            "definition_name": f"mla_paged_{info['variant']}_NEEDS_CONFIG",
            **observation_fields(info, obs_kw, obs_pn),
            "note": "Definition name requires HF config.json params.",
        })
    return kernels


def _render_special_definition(kernel: dict) -> dict | None:
    p = kernel["params"]
    variant = kernel["variant"]
    template_name = p.get("template_name")
    fi_api_tag = p.get("fi_api_tag") or kernel["fi_api"]
    name = kernel["definition_name"]

    h = p.get("num_heads")
    head_dim_qk = p.get("head_dim_qk")
    page_size = p.get("page_size")
    if not _all_ints(h, head_dim_qk, page_size):
        return None

    import torch

    if variant in {"trtllm_decode", "xqa_decode"}:
        ckv = p.get("kv_lora_rank")
        kpe = p.get("qk_rope_head_dim")
        nope = p.get("qk_nope_head_dim")
        if not _all_ints(ckv, kpe, nope):
            return None
        official_kwargs: dict[str, Any] = {
            "query": _meta((1, 1, h, head_dim_qk), torch.bfloat16),
            "kv_cache": _meta((1, page_size, head_dim_qk), torch.bfloat16),
            "workspace_buffer": _meta((1,), torch.int8),
            "qk_nope_head_dim": nope,
            "kv_lora_rank": ckv,
            "qk_rope_head_dim": kpe,
            "block_tables": _meta((1, 1), torch.int32),
            "seq_lens": _meta((1,), torch.int32),
            "max_seq_len": 1,
            "bmm1_scale": 1.0,
            "bmm2_scale": 1.0,
        }
    elif variant == "cute_dsl_decode":
        official_kwargs = {
            "q": _meta((1, 1, h, head_dim_qk), torch.bfloat16),
            "kv_cache": _meta((1, page_size, head_dim_qk), torch.bfloat16),
            "block_tables": _meta((1, 1), torch.int32),
            "seq_lens": _meta((1,), torch.int32),
            "max_seq_len": 1,
            "softmax_scale": 1.0,
            "output_scale": 1.0,
        }
    else:
        return None

    return render_definition(
        "attention",
        template_name,
        fi_api_tag,
        official_kwargs,
        name=name,
    )


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    """Generate an MLA paged attention definition (decode or prefill)."""
    p = kernel["params"]
    variant = kernel["variant"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]

    if variant in _SPECIAL_VARIANTS:
        official = _render_special_definition(kernel)
        return official

    h = p.get("num_heads", "?")
    ckv = p.get("ckv_dim", "?")
    kpe = p.get("kpe_dim", "?")
    page_size = p.get("page_size", 1)

    is_decode = variant == "decode"
    fi_api_tag = fi_api if not fi_api or fi_api.endswith(".run") else f"{fi_api}.run"
    if all(isinstance(x, int) for x in (h, ckv, kpe, page_size)):
        import torch

        official_kwargs: dict[str, Any] = {
            "q_nope": torch.empty((1, h, ckv), dtype=torch.bfloat16),
            "q_pe": torch.empty((1, h, kpe), dtype=torch.bfloat16),
            "ckv_cache": torch.empty((1, page_size, ckv), dtype=torch.bfloat16),
            "kpe_cache": torch.empty((1, page_size, kpe), dtype=torch.bfloat16),
        }
        if is_decode:
            template_name = "mla_paged_decode_trace"
        else:
            template_name = "mla_paged_prefill_trace"
            official_kwargs = {
                "q_nope": torch.empty((1, h, ckv), dtype=torch.bfloat16),
                "q_pe": torch.empty((1, h, kpe), dtype=torch.bfloat16),
                "ckv_cache": torch.empty((1, page_size, ckv), dtype=torch.bfloat16),
                "kpe_cache": torch.empty((1, page_size, kpe), dtype=torch.bfloat16),
                "qo_indptr": torch.tensor([0, 1], dtype=torch.int32),
                "kv_indptr": torch.tensor([0, 1], dtype=torch.int32),
                "kv_indices": torch.tensor([0], dtype=torch.int32),
                "sm_scale": 1.0,
            }
        official = render_definition(
            "attention",
            template_name,
            fi_api_tag,
            official_kwargs,
            name=name,
        )
        return official

    return None


def _build_mla_solution_source(*, decode: bool, has_kv_last_page_len: bool) -> str:
    """Build FlashInfer MLA paged wrapper baseline source."""
    if decode:
        signature = (
            "q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, "
            "kv_last_page_len, sm_scale"
            if has_kv_last_page_len
            else "q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale"
        )
        kv_len_expr = (
            "((kv_indptr[1:] - kv_indptr[:-1] - 1) * page_size + "
            "kv_last_page_len).to(torch.int32)"
            if has_kv_last_page_len
            else "((kv_indptr[1:] - kv_indptr[:-1]) * page_size).to(torch.int32)"
        )
        extra_state = (
            'or state.get("kv_last_page_len_ptr") != kv_last_page_len.data_ptr()'
            if has_kv_last_page_len
            else ""
        )
        extra_state_record = (
            '"kv_last_page_len_ptr": kv_last_page_len.data_ptr(),'
            if has_kv_last_page_len
            else ""
        )
        return textwrap.dedent(f"""\
            import torch
            import flashinfer

            _WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
            _workspace_cache = {{}}
            _wrapper_cache = {{}}
            _plan_state = {{}}


            def _scalar(value):
                if isinstance(value, torch.Tensor):
                    return float(value.item())
                return float(value)


            def _get_workspace(device):
                key = str(device)
                buffer = _workspace_cache.get(key)
                if buffer is None or buffer.device != device or buffer.numel() < _WORKSPACE_SIZE_BYTES:
                    buffer = torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.int8, device=device)
                    _workspace_cache[key] = buffer
                return buffer


            def _get_wrapper(key, device):
                wrapper = _wrapper_cache.get(key)
                if wrapper is None:
                    workspace = _get_workspace(device)
                    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace)
                    _wrapper_cache[key] = wrapper
                return wrapper


            def run({signature}):
                batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
                head_dim_kpe = q_pe.shape[-1]
                page_size = ckv_cache.shape[1]
                len_indptr = kv_indptr.shape[0]
                num_kv_indices = kv_indices.shape[0]
                sm_scale_value = _scalar(sm_scale)

                device = q_nope.device
                wrapper_key = (
                    str(device),
                    num_qo_heads,
                    head_dim_ckv,
                    head_dim_kpe,
                    page_size,
                    q_nope.dtype,
                    q_pe.dtype,
                    ckv_cache.dtype,
                    kpe_cache.dtype,
                )

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
                        {extra_state}
                    )

                if needs_plan:
                    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
                    kv_len_arr = {kv_len_expr}
                    wrapper.plan(
                        qo_indptr=qo_indptr,
                        kv_indptr=kv_indptr,
                        kv_indices=kv_indices,
                        kv_len_arr=kv_len_arr,
                        num_heads=num_qo_heads,
                        head_dim_ckv=head_dim_ckv,
                        head_dim_kpe=head_dim_kpe,
                        page_size=page_size,
                        causal=False,
                        sm_scale=sm_scale_value,
                        q_data_type=q_nope.dtype,
                        kv_data_type=ckv_cache.dtype,
                    )
                    _plan_state[wrapper_key] = {{
                        "batch_size": batch_size,
                        "len_indptr": len_indptr,
                        "num_kv_indices": num_kv_indices,
                        "sm_scale": sm_scale_value,
                        "kv_indptr_ptr": kv_indptr.data_ptr(),
                        "kv_indices_ptr": kv_indices.data_ptr(),
                        {extra_state_record}
                    }}

                output, lse = wrapper.run(
                    q_nope,
                    q_pe,
                    ckv_cache,
                    kpe_cache,
                    return_lse=True,
                )

                return output, lse
            """) + "\n"

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
                buffer = torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.int8, device=device)
                _workspace_cache[key] = buffer
            return buffer


        def _get_wrapper(key, device):
            wrapper = _wrapper_cache.get(key)
            if wrapper is None:
                workspace = _get_workspace(device)
                wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace)
                _wrapper_cache[key] = wrapper
            return wrapper


        def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
            total_q, num_qo_heads, head_dim_ckv = q_nope.shape
            head_dim_kpe = q_pe.shape[-1]
            page_size = ckv_cache.shape[1]
            len_indptr = kv_indptr.shape[0]
            num_kv_indices = kv_indices.shape[0]
            batch_size = qo_indptr.shape[0] - 1
            sm_scale_value = _scalar(sm_scale)

            device = q_nope.device
            wrapper_key = (
                str(device),
                num_qo_heads,
                head_dim_ckv,
                head_dim_kpe,
                page_size,
                q_nope.dtype,
                q_pe.dtype,
                ckv_cache.dtype,
                kpe_cache.dtype,
            )

            wrapper = _get_wrapper(wrapper_key, device)
            state = _plan_state.get(wrapper_key)

            needs_plan = True
            if state is not None:
                needs_plan = (
                    state.get("total_q") != total_q
                    or state.get("batch_size") != batch_size
                    or state.get("len_indptr") != len_indptr
                    or state.get("num_kv_indices") != num_kv_indices
                    or state.get("sm_scale") != sm_scale_value
                    or state.get("qo_indptr_ptr") != qo_indptr.data_ptr()
                    or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
                    or state.get("kv_indices_ptr") != kv_indices.data_ptr()
                )

            if needs_plan:
                kv_len_arr = ((kv_indptr[1:] - kv_indptr[:-1]) * page_size).to(torch.int32)
                wrapper.plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    kv_indices=kv_indices,
                    kv_len_arr=kv_len_arr,
                    num_heads=num_qo_heads,
                    head_dim_ckv=head_dim_ckv,
                    head_dim_kpe=head_dim_kpe,
                    page_size=page_size,
                    causal=True,
                    sm_scale=sm_scale_value,
                    q_data_type=q_nope.dtype,
                    kv_data_type=ckv_cache.dtype,
                )
                _plan_state[wrapper_key] = {
                    "total_q": total_q,
                    "batch_size": batch_size,
                    "len_indptr": len_indptr,
                    "num_kv_indices": num_kv_indices,
                    "sm_scale": sm_scale_value,
                    "qo_indptr_ptr": qo_indptr.data_ptr(),
                    "kv_indptr_ptr": kv_indptr.data_ptr(),
                    "kv_indices_ptr": kv_indices.data_ptr(),
                }

            output, lse = wrapper.run(
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                return_lse=True,
            )

            return output, lse
        """) + "\n"


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate FlashInfer MLA paged wrapper baseline solution source."""
    if definition.get("op_type") != OP_TYPE:
        return None
    names = set(input_names(definition))
    if {"q_nope", "q_pe", "ckv_cache", "kpe_cache", "kv_indptr", "kv_indices", "sm_scale"}.issubset(names):
        is_decode = "qo_indptr" not in names
        if is_decode:
            source = _build_mla_solution_source(
                decode=True,
                has_kv_last_page_len="kv_last_page_len" in names,
            )
        else:
            if not has_inputs(definition, {"qo_indptr"}):
                return None
            source = _build_mla_solution_source(decode=False, has_kv_last_page_len=False)
        return solution_payload(
            definition,
            source,
            description="Baseline solution using FlashInfer BatchMLAPagedAttentionWrapper.",
        )
    return None
