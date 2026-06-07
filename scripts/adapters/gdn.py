"""Adapter for FlashInfer GDN (Gated Delta Network) kernels."""

from __future__ import annotations

import textwrap
import re

from ._param_utils import analysis_allows_family, extract_hf_dims, first_int
from adapters.official_templates import render_definition
from ._solution_utils import has_inputs, solution_payload
from .extractors import extract_observed_kwargs, extract_observed_param_names
from parse.inventory_helpers import observation_fields

OP_TYPE = "gdn"


def pr_reference_source() -> str:
    return "FlashInfer GDN unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} GDN (Gated Delta Network) layer"


def pr_baseline_description(_def_name: str) -> str:
    return "torch reference (delta-rule recurrence)"


def _snake_tokens(name: str) -> set[str]:
    return {piece for piece in re.split(r"[_\W]+", name.lower()) if piece}


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    parts = trace_id.split(".")
    if len(parts) < 2 or parts[0] != "flashinfer" or parts[1] != "gdn":
        return None
    tokens = _snake_tokens(parts[-1])
    if "decode" in tokens:
        return (OP_TYPE, "decode")
    if "prefill" in tokens:
        return (OP_TYPE, "prefill")
    if "chunk" in tokens or "mtp" in tokens:
        return (OP_TYPE, "mtp")
    return None


def definition_name(variant: str, params: dict) -> str:
    """Return the canonical GDN definition name for resolved params."""
    qk = params["q_heads"]
    v = params["v_heads"]
    d = params["head_dim"]
    if variant == "decode":
        return f"gdn_decode_qk{qk}_v{v}_d{d}"
    if variant == "prefill":
        return f"gdn_prefill_qk{qk}_v{v}_d{d}"
    if variant == "mtp":
        return f"gdn_mtp_qk{qk}_v{v}_d{d}"
    raise ValueError(f"unknown GDN variant: {variant}")


def fi_api_for_variant(variant: str) -> str | None:
    """Return the FlashInfer API tag used for static candidates."""
    return {
        "decode": "flashinfer.gdn.gated_delta_rule_decode",
        "prefill": "flashinfer.gdn.chunk_gated_delta_rule",
        "mtp": "flashinfer.gdn.gated_delta_rule_mtp",
    }.get(variant)


def resolve_gdn_params(
    *,
    raw_params: dict | None = None,
    config: dict | None = None,
    tp: int | None = None,
) -> dict | None:
    """Resolve canonical GDN params owned by this adapter."""
    raw = raw_params or {}
    dims = extract_hf_dims(config or {}, tp)
    q = first_int(
        raw.get("q_heads"),
        raw.get("num_q_heads"),
        raw.get("gdn_num_q_heads"),
        dims.get("gdn_num_q_heads"),
    )
    v = first_int(
        raw.get("v_heads"),
        raw.get("num_v_heads"),
        raw.get("gdn_num_v_heads"),
        dims.get("gdn_num_v_heads"),
    )
    d = first_int(
        raw.get("gdn_head_dim"),
        raw.get("head_dim"),
        raw.get("head_size"),
        dims.get("gdn_head_dim"),
    )
    if not (q and d):
        return None
    return {
        "q_heads": q,
        "v_heads": v if v is not None else "?",
        "head_dim": d,
    }


def resolve_config_params(
    kernel: dict,
    config: dict,
    tp_size: int | None,
) -> dict | None:
    """Resolve missing GDN params from HF config."""
    return resolve_gdn_params(
        raw_params=kernel.get("params"),
        config=config,
        tp=tp_size,
    )


def static_candidates(
    dims: dict,
    analysis: dict,
    *,
    tp: int,
    page_size: int,
) -> list[dict]:
    """Build GDN static candidates from HF config dimensions."""
    params = resolve_gdn_params(raw_params=dims)
    if (
        not params
        or params.get("v_heads") == "?"
        or not analysis_allows_family(analysis, "has_gdn")
    ):
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
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        kernels.append({
            "op_type": OP_TYPE,
            "variant": info["variant"],
            "fi_api": info["fi_api"],
            "params": {},
            "definition_name": f"gdn_{info['variant']}_NEEDS_CONFIG",
            **observation_fields(info, obs_kw, obs_pn),
            "note": (
                "Definition name requires HF config.json params "
                "(q_heads, v_heads, head_dim)."
            ),
        })
    return kernels


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    """Generate a GDN (Gated Delta Net) definition (decode, prefill, or mtp)."""
    p = kernel["params"]
    variant = kernel["variant"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]

    qk = p.get("q_heads", "?")
    v = p.get("v_heads", "?")
    d = p.get("head_dim", "?")
    if all(isinstance(x, int) for x in (qk, v, d)):
        import torch

        fi_api_tag = fi_api or "flashinfer.gdn"
        if not fi_api_tag.endswith(".run") and "Wrapper" in fi_api_tag:
            fi_api_tag = f"{fi_api_tag}.run"
        official_kwargs: dict[str, object]
        if variant == "decode":
            official_kwargs = {
                "q": torch.empty((1, 1, qk, d), dtype=torch.bfloat16),
                "k": torch.empty((1, 1, qk, d), dtype=torch.bfloat16),
                "v": torch.empty((1, 1, v, d), dtype=torch.bfloat16),
                "state": torch.empty((1, v, d, d), dtype=torch.float32),
                "A_log": torch.empty((v,), dtype=torch.float32),
                "a": torch.empty((1, 1, v), dtype=torch.bfloat16),
                "dt_bias": torch.empty((v,), dtype=torch.float32),
                "b": torch.empty((1, 1, v), dtype=torch.bfloat16),
                "scale": 1.0,
            }
            template_name = "gated_delta_rule_decode_trace"
        elif variant == "mtp":
            official_kwargs = {
                "q": torch.empty((1, 2, qk, d), dtype=torch.bfloat16),
                "k": torch.empty((1, 2, qk, d), dtype=torch.bfloat16),
                "v": torch.empty((1, 2, v, d), dtype=torch.bfloat16),
                "initial_state": torch.empty((1, v, d, d), dtype=torch.float32),
                "initial_state_indices": torch.tensor([0], dtype=torch.int32),
                "A_log": torch.empty((v,), dtype=torch.float32),
                "a": torch.empty((1, 2, v), dtype=torch.bfloat16),
                "dt_bias": torch.empty((v,), dtype=torch.float32),
                "b": torch.empty((1, 2, v), dtype=torch.bfloat16),
                "scale": 1.0,
                "intermediate_states_buffer": torch.empty((1, 2, v, d, d), dtype=torch.float32),
            }
            template_name = "gdn_mtp_trace"
        else:
            official_kwargs = {
                "q": torch.empty((1, qk, d), dtype=torch.bfloat16),
                "k": torch.empty((1, qk, d), dtype=torch.bfloat16),
                "v": torch.empty((1, v, d), dtype=torch.bfloat16),
                "initial_state": torch.empty((1, v, d, d), dtype=torch.float32),
                "A_log": torch.empty((v,), dtype=torch.float32),
                "g": torch.empty((1, v), dtype=torch.bfloat16),
                "dt_bias": torch.empty((v,), dtype=torch.float32),
                "beta": torch.empty((1, v), dtype=torch.bfloat16),
                "cu_seqlens": torch.tensor([0, 1], dtype=torch.int64),
                "scale": 1.0,
            }
            template_name = "gdn_prefill_trace"
        official = render_definition(
            "gdn",
            template_name,
            fi_api_tag,
            official_kwargs,
            name=name,
        )
        if official is not None:
            return official
        return None

    return None


def _build_gdn_decode_solution_source() -> str:
    return textwrap.dedent("""\
        import math
        import torch
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose


        def run(q, k, v, state, A_log, a, dt_bias, b, scale):
            if isinstance(scale, torch.Tensor):
                scale = float(scale.item())
            else:
                scale = float(scale)
            if scale == 0.0:
                scale = 1.0 / math.sqrt(q.shape[-1])

            batch_size, seq_len, num_v_heads, head_size = v.shape
            output = torch.empty(
                batch_size,
                seq_len,
                num_v_heads,
                head_size,
                dtype=q.dtype,
                device=q.device,
            )

            out, new_state = gated_delta_rule_decode_pretranspose(
                q=q,
                k=k,
                v=v,
                state=state,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                scale=scale,
                output=output,
                use_qk_l2norm=False,
            )
            return out, new_state
        """) + "\n"


def _build_gdn_prefill_solution_source() -> str:
    return textwrap.dedent("""\
        import torch
        import torch.nn.functional as F

        try:
            from flashinfer.gdn import chunk_gated_delta_rule
        except Exception:
            from flashinfer.gdn_decode import chunk_gated_delta_rule


        def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
            x = a.float() + dt_bias.float()
            g = -torch.exp(A_log.float()) * F.softplus(x)
            beta = torch.sigmoid(b.float())

            varlen = cu_seqlens is not None and q.dim() == 3
            if varlen:
                q = q.unsqueeze(0)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)

            output, new_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=None,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )

            if varlen:
                output = output.squeeze(0)

            return output, new_state
        """) + "\n"


def _build_gdn_mtp_solution_source() -> str:
    return textwrap.dedent("""\
        import math
        import torch
        from flashinfer.gdn_decode import gated_delta_rule_mtp


        def run(
            q,
            k,
            v,
            initial_state,
            initial_state_indices,
            A_log,
            a,
            dt_bias,
            b,
            scale,
            intermediate_states_buffer=None,
        ):
            if isinstance(scale, torch.Tensor):
                scale = float(scale.item())
            else:
                scale = float(scale)
            if scale == 0.0:
                scale = 1.0 / math.sqrt(q.shape[-1])

            output, final_state = gated_delta_rule_mtp(
                q=q,
                k=k,
                v=v,
                initial_state=initial_state,
                initial_state_indices=initial_state_indices,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                scale=scale,
                intermediate_states_buffer=intermediate_states_buffer,
                disable_state_update=True,
                use_qk_l2norm=False,
            )

            return output, final_state
        """) + "\n"


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate FlashInfer GDN baseline solution source."""
    if definition.get("op_type") != OP_TYPE:
        return None
    name = str(definition.get("name", ""))
    if "gdn_decode" in name and has_inputs(
        definition, {"q", "k", "v", "state", "A_log", "a", "dt_bias", "b", "scale"}
    ):
        source = _build_gdn_decode_solution_source()
    elif "gdn_prefill" in name and has_inputs(
        definition,
        {"q", "k", "v", "state", "A_log", "a", "dt_bias", "b", "cu_seqlens", "scale"},
    ):
        source = _build_gdn_prefill_solution_source()
    elif "gdn_mtp" in name and has_inputs(
        definition,
        {
            "q",
            "k",
            "v",
            "initial_state",
            "initial_state_indices",
            "A_log",
            "a",
            "dt_bias",
            "b",
            "scale",
        },
    ):
        source = _build_gdn_mtp_solution_source()
    else:
        return None
    return solution_payload(
        definition,
        source,
        description="Baseline solution using FlashInfer GDN kernels.",
    )
