"""Adapter for FlashInfer MoE kernels."""

from __future__ import annotations

import textwrap

from ._param_utils import analysis_allows_family, extract_hf_dims, first_present, is_unknown
from ._solution_utils import has_inputs, input_names, solution_payload
from .extractors import (
    determine_moe_variant,
    extract_moe_params,
    extract_observed_kwargs,
    extract_observed_param_names,
)
from parse.inventory_helpers import observation_fields

try:
    from adapters.official_templates import render_definition, render_reference
except Exception:  # pragma: no cover - local FlashInfer checkout may be absent.
    render_definition = None
    render_reference = None

OP_TYPE = "moe"


def pr_reference_source() -> str:
    return "FlashInfer MoE unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} MoE"


def pr_baseline_description(def_name: str) -> str:
    if "fp8" in def_name:
        return "flashinfer MoE FP8 wrapper"
    return "torch MoE reference (double loop)"


_SPECIAL_TRACE_IDS = {
    "flashinfer.fused_moe.cute_dsl.fused_moe.cute_dsl_fused_moe_nvfp4": (
        "cute_dsl_nvfp4",
        "cute_dsl_fused_moe_nvfp4_trace",
    ),
    "flashinfer.fused_moe.cute_dsl.fused_moe.CuteDslMoEWrapper.run": (
        "cute_dsl_wrapper",
        "cute_dsl_moe_wrapper_run_trace",
    ),
    "flashinfer.fused_moe.cute_dsl.b12x_moe.b12x_fused_moe": (
        "b12x",
        "b12x_fused_moe_trace",
    ),
    "flashinfer.fused_moe.cute_dsl.b12x_moe.B12xMoEWrapper.run": (
        "b12x_wrapper",
        "b12x_moe_wrapper_run_trace",
    ),
}


def _all_ints(*values) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def _meta(shape, dtype):
    import torch

    return torch.empty(tuple(int(v) for v in shape), dtype=dtype, device="meta")


def _fill_moe_core_descriptions(definition: dict) -> None:
    """Fill descriptions used by existing official MoE dataset definitions."""
    axes = definition.get("axes", {})
    axes.get("top_k", {}).setdefault(
        "description", "Number of experts selected per token."
    )
    axes.get("hidden_size", {}).setdefault("description", "Hidden dimension size.")
    axes.get("intermediate_size", {}).setdefault(
        "description", "MoE expert intermediate size."
    )


def _float8_e4m3fn():
    import torch

    return getattr(torch, "float8_e4m3fn", torch.uint8)


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    special = _SPECIAL_TRACE_IDS.get(trace_id)
    if special is not None:
        return (OP_TYPE, special[0])

    parts = trace_id.split(".")
    if len(parts) < 3 or parts[0] != "flashinfer" or parts[1] != "fused_moe":
        return None
    func = parts[-1].lower()
    if "moe" not in func and "topk" not in func:
        return None
    if "fp8" in func:
        return (OP_TYPE, "fp8")
    if "bf16" in func:
        return (OP_TYPE, "generic_moe")
    return (OP_TYPE, "generic_moe")


def _shape(value) -> list | None:
    if isinstance(value, dict) and value.get("type") == "tensor":
        shape = value.get("shape")
        return shape if isinstance(shape, list) else None
    return None


def _scalar_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        raw = value.get("value")
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
    return None


def _bound_signature_args(sig: dict) -> dict:
    s = sig.get("signature", {})
    args = s.get("args", [])
    kwargs = s.get("kwargs", {})
    names = sig.get("param_names") or s.get("param_names") or []
    bound = {}
    if isinstance(names, list):
        for name, arg in zip(names, args):
            if isinstance(name, str):
                bound[name] = arg
    if isinstance(kwargs, dict):
        bound.update(kwargs)
    return bound


def _self_attrs(sig: dict) -> dict:
    s = sig.get("signature", {})
    self_info = s.get("self")
    if isinstance(self_info, dict):
        attrs = self_info.get("attrs")
        if isinstance(attrs, dict):
            return attrs
    return {}


def _first_int(*values) -> int | None:
    for value in values:
        parsed = _scalar_int(value)
        if parsed is not None:
            return parsed
    return None


def _extract_direct_moe_params(signatures: list[dict]) -> list[dict]:
    params: list[dict] = []
    seen: set[tuple] = set()
    for sig in signatures:
        bound = _bound_signature_args(sig)
        routing_shape = _shape(bound.get("routing_logits"))
        hidden_shape = _shape(bound.get("hidden_states"))
        gemm1_shape = _shape(bound.get("gemm1_weights"))
        gemm2_shape = _shape(bound.get("gemm2_weights"))

        p = {
            "top_k": _scalar_int(bound.get("top_k")),
            "num_experts": _scalar_int(bound.get("num_experts")),
            "n_group": _scalar_int(bound.get("n_group")),
            "topk_group": _scalar_int(bound.get("topk_group")),
        }
        if routing_shape and len(routing_shape) >= 2:
            p["num_experts"] = p["num_experts"] or routing_shape[-1]
        if hidden_shape and len(hidden_shape) >= 2:
            p["hidden_size"] = hidden_shape[-1]
        if gemm1_shape and len(gemm1_shape) >= 3:
            p["num_local_experts"] = gemm1_shape[0]
            p["intermediate_size"] = gemm1_shape[1] // 2
            p["hidden_size"] = p.get("hidden_size") or gemm1_shape[2]
        if gemm2_shape and len(gemm2_shape) >= 3:
            p["num_local_experts"] = p.get("num_local_experts") or gemm2_shape[0]
            p["hidden_size"] = p.get("hidden_size") or gemm2_shape[1]
            p["intermediate_size"] = p.get("intermediate_size") or gemm2_shape[2]
        p = {k: v for k, v in p.items() if v is not None}
        if {"top_k", "hidden_size", "intermediate_size"}.issubset(p):
            key = tuple(sorted(p.items()))
            if key not in seen:
                seen.add(key)
                params.append(p)
    return params


def _extract_special_moe_params(signatures: list[dict], variant: str) -> list[dict]:
    params: list[dict] = []
    seen: set[tuple] = set()
    for sig in signatures:
        bound = _bound_signature_args(sig)
        attrs = _self_attrs(sig)
        x_shape = _shape(bound.get("x"))
        w1_shape = _shape(bound.get("w1_weight"))
        w2_shape = _shape(bound.get("w2_weight"))
        selected_shape = _shape(bound.get("token_selected_experts"))

        if not (
            x_shape
            and w1_shape
            and w2_shape
            and selected_shape
            and len(x_shape) >= 2
            and len(w1_shape) >= 3
            and len(w2_shape) >= 3
            and len(selected_shape) >= 2
        ):
            continue

        top_k = _first_int(
            bound.get("top_k"),
            attrs.get("top_k"),
            selected_shape[-1],
        )
        num_experts = _first_int(
            bound.get("num_experts"),
            attrs.get("num_experts"),
        )
        num_local_experts = _first_int(
            bound.get("num_local_experts"),
            attrs.get("num_local_experts"),
            w1_shape[0],
        )
        hidden_size = _first_int(attrs.get("hidden_size"))
        if hidden_size is None:
            if variant in {"cute_dsl_nvfp4", "cute_dsl_wrapper"}:
                hidden_size = _first_int(w2_shape[1], x_shape[-1] * 2)
            else:
                hidden_size = _first_int(x_shape[-1], w2_shape[1])
        intermediate_size = _first_int(attrs.get("intermediate_size"))
        if intermediate_size is None:
            intermediate_size = _first_int(w2_shape[-1] * 2)

        p = {
            "top_k": top_k,
            "num_experts": num_experts or num_local_experts,
            "num_local_experts": num_local_experts,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        }
        p = {k: v for k, v in p.items() if v is not None}
        if {
            "top_k",
            "num_experts",
            "num_local_experts",
            "hidden_size",
            "intermediate_size",
        }.issubset(p):
            key = tuple(sorted(p.items()))
            if key not in seen:
                seen.add(key)
                params.append(p)
    return params


def _special_template_for_info(info: dict) -> str | None:
    for trace_id in info.get("trace_ids", []):
        special = _SPECIAL_TRACE_IDS.get(trace_id)
        if special is not None:
            return special[1]
    return None


def _special_definition_name(variant: str, params: dict) -> str:
    topk = params["top_k"]
    local_experts = params["num_local_experts"]
    hidden = params["hidden_size"]
    if variant == "cute_dsl_nvfp4":
        return f"cute_dsl_fused_moe_nvfp4_topk{topk}_e{local_experts}_h{hidden}"
    if variant == "cute_dsl_wrapper":
        return f"cute_dsl_moe_wrapper_e{local_experts}_h{hidden}"
    if variant == "b12x":
        return f"b12x_fused_moe_topk{topk}_e{local_experts}_h{hidden}"
    if variant == "b12x_wrapper":
        return f"b12x_moe_wrapper_e{local_experts}_h{hidden}"
    raise ValueError(f"unknown special MoE variant: {variant}")


def definition_name(variant: str, params: dict) -> str:
    """Return the canonical MoE definition name for resolved params."""
    if variant in {"cute_dsl_nvfp4", "cute_dsl_wrapper", "b12x", "b12x_wrapper"}:
        return _special_definition_name(variant, params)

    topk = params.get("top_k", "?")
    hidden = params.get("hidden_size", "?")
    inter = params.get("intermediate_size", "?")
    if variant == "fp8":
        return (
            f"moe_fp8_block_scale_ds_routing_topk{topk}"
            f"_ng{params.get('n_group', '?')}_kg{params.get('topk_group', '?')}"
            f"_e{params.get('num_local_experts', '?')}_h{hidden}_i{inter}"
        )

    local_experts = params.get("num_local_experts", params.get("num_experts", "?"))
    return f"trtllm_bf16_moe_topk{topk}_e{local_experts}_h{hidden}_i{inter}"


def fi_api_for_variant(variant: str) -> str | None:
    """Return the FlashInfer API tag used for static candidates."""
    return {
        "bf16": "flashinfer.fused_moe.trtllm_bf16_moe",
    }.get(variant)


def resolve_moe_params(
    *,
    raw_params: dict | None = None,
    config: dict | None = None,
    tp: int | None = None,
    variant: str | None = None,
) -> dict | None:
    """Resolve canonical MoE params owned by this adapter."""
    raw = raw_params or {}
    dims = extract_hf_dims(config or {}, tp)

    top_k = first_present(
        raw.get("top_k"),
        raw.get("topk"),
        raw.get("num_experts_per_tok"),
        raw.get("num_experts_per_token"),
        dims.get("num_experts_per_tok"),
    )
    hidden = first_present(raw.get("hidden_size"), dims.get("hidden_size"))
    inter = first_present(
        raw.get("intermediate_size_per_partition"),
        raw.get("tp_intermediate_size"),
        dims.get("tp_intermediate_size"),
        raw.get("intermediate_size"),
        dims.get("intermediate_size"),
    )
    num_experts = first_present(
        raw.get("num_experts"),
        raw.get("n_routed_experts"),
        dims.get("num_experts"),
    )
    num_local = first_present(
        raw.get("num_local_experts"),
        raw.get("tp_num_experts"),
        dims.get("num_local_experts"),
        num_experts,
    )

    if not all(not is_unknown(v) for v in (top_k, hidden, inter, num_experts)):
        return None

    has_fp8_evidence = any(
        not is_unknown(raw.get(key))
        for key in ("n_group", "topk_group", "block_size", "weight_block_size")
    )
    if variant == "fp8" or has_fp8_evidence:
        n_group = first_present(raw.get("n_group"), dims.get("n_group"))
        topk_group = first_present(raw.get("topk_group"), dims.get("topk_group"))
        if is_unknown(n_group) or is_unknown(topk_group):
            return None
        return {
            "top_k": top_k,
            "n_group": n_group,
            "topk_group": topk_group,
            "num_experts": num_experts,
            "num_local_experts": num_local,
            "hidden_size": hidden,
            "intermediate_size": inter,
        }

    return {
        "top_k": top_k,
        "num_experts": num_experts,
        "num_local_experts": num_local if not is_unknown(num_local) else num_experts,
        "hidden_size": hidden,
        "intermediate_size": inter,
    }


def resolve_config_params(
    kernel: dict,
    config: dict,
    tp_size: int | None,
) -> dict | None:
    """Resolve missing MoE params from HF config."""
    return resolve_moe_params(
        raw_params=kernel.get("params"),
        config=config,
        tp=tp_size,
        variant=kernel.get("variant"),
    )


def needs_config_resolution(kernel: dict) -> bool:
    """Return whether this MoE kernel still needs HF config enrichment."""
    if "NEEDS_CONFIG" in kernel.get("definition_name", ""):
        return True
    return any(is_unknown(value) for value in kernel.get("params", {}).values())


def static_candidates(
    dims: dict,
    analysis: dict,
    *,
    tp: int,
    page_size: int,
) -> list[dict]:
    """Build MoE static candidates from HF config dimensions."""
    variant = "bf16"
    params = resolve_moe_params(raw_params=dims, tp=tp, variant=variant)
    if not params or not analysis_allows_family(analysis, "has_moe"):
        return []
    return [{
        "op_type": OP_TYPE,
        "variant": variant,
        "fi_api": fi_api_for_variant(variant),
        "params": params,
        "definition_name": definition_name(variant, params),
    }]


def _needs_config_entry(
    info: dict,
    variant: str,
    obs_kw: list[str],
    obs_pn: list[str],
    note: str,
    params: dict | None = None,
) -> dict:
    """Build an explicit unresolved entry instead of silently dropping evidence."""
    return {
        "op_type": OP_TYPE,
        "variant": variant,
        "fi_api": info["fi_api"],
        "params": params or {},
        "definition_name": f"moe_{variant}_NEEDS_CONFIG",
        **observation_fields(info, obs_kw, obs_pn),
        "note": note,
    }


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    kernels: list[dict] = []
    for key in [k for k in matched_kernels if k.startswith(f"{OP_TYPE}:")]:
        info = matched_kernels[key]
        sub_variant = info["variant"]
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        if sub_variant in {
            "cute_dsl_nvfp4",
            "cute_dsl_wrapper",
            "b12x",
            "b12x_wrapper",
        }:
            template_name = _special_template_for_info(info)
            if template_name is None:
                kernels.append(
                    _needs_config_entry(
                        info,
                        sub_variant,
                        obs_kw,
                        obs_pn,
                        "Special MoE trace matched but no official template name was resolved.",
                    )
                )
                continue
            emitted = False
            for params in _extract_special_moe_params(info["signatures"], sub_variant):
                params = dict(params)
                params["template_name"] = template_name
                emitted = True
                kernels.append({
                    "op_type": OP_TYPE,
                    "variant": sub_variant,
                    "fi_api": info["fi_api"],
                    "params": params,
                    "definition_name": definition_name(sub_variant, params),
                    **observation_fields(info, obs_kw, obs_pn),
                })
            if not emitted:
                kernels.append(
                    _needs_config_entry(
                        info,
                        sub_variant,
                        obs_kw,
                        obs_pn,
                        "Special MoE trace matched but runtime signatures did not expose enough params.",
                        params={"template_name": template_name},
                    )
                )
            continue

        moe_params_list = (
            extract_moe_params(info["signatures"])
            + _extract_direct_moe_params(info["signatures"])
        )
        if not moe_params_list:
            kernels.append(
                _needs_config_entry(
                    info,
                    sub_variant,
                    obs_kw,
                    obs_pn,
                    "MoE trace matched but runtime signatures did not expose enough params.",
                )
            )
            continue
        seen: set[tuple] = set()
        for mp in moe_params_list:
            moe_variant = determine_moe_variant(sub_variant, mp)
            params = resolve_moe_params(raw_params=mp, variant=moe_variant) or mp
            key_tuple = tuple(sorted(params.items()))
            if key_tuple in seen:
                continue
            seen.add(key_tuple)
            if moe_variant == "fp8":
                fi_api = info["fi_api"]
            else:
                fi_api = info["fi_api"] or "flashinfer.fused_moe.trtllm_bf16_moe"

            kernels.append({
                "op_type": OP_TYPE,
                "variant": moe_variant,
                "fi_api": fi_api,
                "params": params,
                "definition_name": definition_name(moe_variant, params),
                **observation_fields(info, obs_kw, obs_pn),
            })
    return kernels


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict | None:
    """Generate a MoE definition, dispatching on variant."""
    variant = kernel["variant"]
    if variant in {"cute_dsl_nvfp4", "cute_dsl_wrapper", "b12x", "b12x_wrapper"}:
        return _gen_special_moe(kernel)
    if variant == "fp8":
        return _gen_moe_fp8(kernel, model_tag, tp)
    if variant in {"sum_reduce", "align_block_size"}:
        raise ValueError(
            f"MoE variant {variant!r} is target_api-only. "
            "It is not supported by the serving fi_api adapter path."
        )
    return _gen_moe_bf16(kernel, model_tag, tp)


def _gen_special_moe(kernel: dict) -> dict | None:
    import torch

    p = kernel["params"]
    variant = kernel["variant"]
    topk = p["top_k"]
    experts = p["num_experts"]
    local_experts = p["num_local_experts"]
    hidden = p["hidden_size"]
    inter = p["intermediate_size"]
    template_name = p["template_name"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]

    if not _all_ints(topk, experts, local_experts, hidden, inter):
        raise ValueError(f"incomplete special MoE params for {name}: {p}")

    gemm1_out = 2 * inter
    hidden_packed = hidden // 2
    inter_packed = inter // 2
    hidden_blocks = hidden // 16
    inter_blocks = inter // 16

    common = {
        "token_selected_experts": _meta((1, topk), torch.int32),
        "token_final_scales": _meta((1, topk), torch.float32),
        "w1_weight": _meta((local_experts, gemm1_out, hidden_packed), torch.uint8),
        "w1_weight_sf": _meta(
            (local_experts, gemm1_out, hidden_blocks), _float8_e4m3fn()
        ),
        "w1_alpha": _meta((local_experts,), torch.float32),
        "fc2_input_scale": _meta((1,), torch.float32),
        "w2_weight": _meta((local_experts, hidden, inter_packed), torch.uint8),
        "w2_weight_sf": _meta((local_experts, hidden, inter_blocks), _float8_e4m3fn()),
        "w2_alpha": _meta((local_experts,), torch.float32),
        "num_experts": int(experts),
        "top_k": int(topk),
    }
    if variant in {"cute_dsl_nvfp4", "cute_dsl_wrapper"}:
        kwargs = {
            "x": _meta((1, hidden_packed), torch.uint8),
            "x_sf": _meta((1, hidden_blocks), _float8_e4m3fn()),
            **common,
            "local_expert_offset": 0,
        }
    else:
        kwargs = {
            "x": _meta((1, hidden), torch.bfloat16),
            **common,
            "activation_precision": "fp4",
        }

    official = (
        render_definition(
            "moe",
            template_name,
            fi_api,
            kwargs,
            name=name,
        )
        if render_definition
        else None
    )
    if official is not None:
        return official
    return None


def _gen_moe_bf16(kernel: dict, model_tag: str, tp: int) -> dict | None:
    """Generate a BF16 MoE definition."""
    import torch

    p = kernel["params"]
    topk = p["top_k"]
    experts = p["num_experts"]
    local_experts = p.get("num_local_experts", experts)
    hidden = p["hidden_size"]
    inter = p["intermediate_size"]
    name = kernel["definition_name"]

    gemm1_out = 2 * inter  # gate + up
    if _all_ints(topk, experts, local_experts, hidden, inter):
        official = (
            render_definition(
                "moe",
                "trtllm_bf16_moe_trace",
                kernel.get("fi_api") or "flashinfer.fused_moe.trtllm_bf16_moe",
                {
                    "routing_logits": _meta((1, experts), torch.float32),
                    "routing_bias": _meta((experts,), torch.bfloat16),
                    "hidden_states": _meta((1, hidden), torch.bfloat16),
                    "gemm1_weights": _meta(
                        (local_experts, gemm1_out, hidden), torch.bfloat16
                    ),
                    "gemm2_weights": _meta(
                        (local_experts, hidden, inter), torch.bfloat16
                    ),
                    "num_experts": int(experts),
                    "top_k": int(topk),
                    "local_expert_offset": 0,
                    "routed_scaling_factor": 1.0,
                    "routing_method_type": 0,
                },
                name=name,
            )
            if render_definition
            else None
        )
        if official:
            _fill_moe_core_descriptions(official)
            return official
        return None

    return None


def _gen_moe_fp8(kernel: dict, model_tag: str, tp: int) -> dict | None:
    """Generate an FP8 block-scale MoE definition."""
    import torch

    p = kernel["params"]
    topk = p.get("top_k", "?")
    n_group = p.get("n_group", "?")
    topk_group = p.get("topk_group", "?")
    num_experts = p.get("num_experts", "?")
    num_local_experts = p.get("num_local_experts", "?")
    hidden = p.get("hidden_size", "?")
    inter = p.get("intermediate_size", "?")
    name = kernel["definition_name"]
    block_size = 128

    gemm1_out = 2 * inter
    num_hidden_blocks = hidden // block_size
    num_intermediate_blocks = inter // block_size
    num_gemm1_out_blocks = gemm1_out // block_size
    if _all_ints(
        topk,
        n_group,
        topk_group,
        num_experts,
        num_local_experts,
        hidden,
        inter,
    ):
        official = (
            render_definition(
                "moe",
                "trtllm_fp8_block_scale_moe_ds_routing_trace",
                kernel.get("fi_api")
                or "flashinfer.fused_moe.trtllm_fp8_block_scale_moe",
                {
                    "routing_logits": _meta((1, num_experts), torch.float32),
                    "routing_bias": _meta((num_experts,), torch.bfloat16),
                    "hidden_states": _meta((1, hidden), _float8_e4m3fn()),
                    "hidden_states_scale": _meta(
                        (num_hidden_blocks, 1), torch.float32
                    ),
                    "gemm1_weights": _meta(
                        (num_local_experts, gemm1_out, hidden), _float8_e4m3fn()
                    ),
                    "gemm1_weights_scale": _meta(
                        (num_local_experts, num_gemm1_out_blocks, num_hidden_blocks),
                        torch.float32,
                    ),
                    "gemm2_weights": _meta(
                        (num_local_experts, hidden, inter), _float8_e4m3fn()
                    ),
                    "gemm2_weights_scale": _meta(
                        (num_local_experts, num_hidden_blocks, num_intermediate_blocks),
                        torch.float32,
                    ),
                    "top_k": int(topk),
                    "n_group": int(n_group),
                    "topk_group": int(topk_group),
                    "local_expert_offset": 0,
                    "routed_scaling_factor": 1.0,
                },
                name=name,
            )
            if render_definition
            else None
        )
        if official:
            return official
        return None

    return None


def _axis_const(definition: dict, name: str, default: int) -> int:
    axis = definition.get("axes", {}).get(name, {})
    if isinstance(axis, dict) and isinstance(axis.get("value"), int):
        return int(axis["value"])
    return default


def _scalar_code(name: str, default: int | float, cast: str) -> str:
    return textwrap.indent(textwrap.dedent(f"""\
            {name}_value = {name} if {name!r} in _ARGS else {default!r}
            if isinstance({name}_value, torch.Tensor):
                {name}_value = {name}_value.item()
            {name}_value = {cast}({name}_value)
        """), "    ")


def _build_moe_fp8_solution_source(definition: dict) -> str:
    names = input_names(definition)
    signature = ", ".join(names)
    top_k = _axis_const(definition, "top_k", 8)
    n_group = _axis_const(definition, "n_group", 8)
    topk_group = _axis_const(definition, "topk_group", 4)
    intermediate = _axis_const(definition, "intermediate_size", 2048)
    source = textwrap.dedent(f"""\
        import torch
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

        _ARGS = {set(names)!r}
        _TOP_K = {top_k}
        _N_GROUP = {n_group}
        _TOPK_GROUP = {topk_group}
        _INTERMEDIATE_SIZE = {intermediate}


        def run({signature}):
            local_num_experts = gemm1_weights.shape[0]
            num_experts_global = routing_logits.shape[1]
        """)
    source += _scalar_code("top_k", top_k, "int")
    source += _scalar_code("n_group", n_group, "int")
    source += _scalar_code("topk_group", topk_group, "int")
    source += textwrap.indent(textwrap.dedent("""\
            local_expert_offset_value = local_expert_offset
            if isinstance(local_expert_offset_value, torch.Tensor):
                local_expert_offset_value = local_expert_offset_value.item()
            local_expert_offset_value = int(local_expert_offset_value)

            routed_scaling_value = routed_scaling_factor
            if isinstance(routed_scaling_value, torch.Tensor):
                routed_scaling_value = routed_scaling_value.item()
            routed_scaling_value = float(routed_scaling_value)

            routing_logits_f32 = routing_logits.to(torch.float32).contiguous()
            hidden_states_scale_f32 = hidden_states_scale.to(torch.float32).contiguous()
            gemm1_weights_scale_f32 = gemm1_weights_scale.to(torch.float32).contiguous()
            gemm2_weights_scale_f32 = gemm2_weights_scale.to(torch.float32).contiguous()
            routing_bias_arg = None if routing_bias is None else routing_bias.contiguous()

            return trtllm_fp8_block_scale_moe(
                routing_logits_f32,
                routing_bias_arg,
                hidden_states.contiguous(),
                hidden_states_scale_f32,
                gemm1_weights.contiguous(),
                gemm1_weights_scale_f32,
                gemm2_weights.contiguous(),
                gemm2_weights_scale_f32,
                num_experts_global,
                top_k_value,
                n_group_value,
                topk_group_value,
                _INTERMEDIATE_SIZE,
                local_expert_offset_value,
                local_num_experts,
                routed_scaling_value,
                routing_method_type=2,
                use_shuffled_weight=False,
            )
        """), "    ")
    return source


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate FlashInfer MoE baseline solution source."""
    if definition.get("op_type") != OP_TYPE:
        return None
    name = str(definition.get("name", ""))
    if name.startswith("moe_fp8") and has_inputs(
        definition,
        {
            "routing_logits",
            "routing_bias",
            "hidden_states",
            "hidden_states_scale",
            "gemm1_weights",
            "gemm1_weights_scale",
            "gemm2_weights",
            "gemm2_weights_scale",
            "local_expert_offset",
            "routed_scaling_factor",
        },
    ):
        source = _build_moe_fp8_solution_source(definition)
        return solution_payload(
            definition,
            source,
            description="Baseline solution using FlashInfer FP8 block-scale MoE.",
        )
    return None
