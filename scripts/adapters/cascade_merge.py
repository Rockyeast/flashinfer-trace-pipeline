"""Adapter for FlashInfer cascade_merge kernels."""

from __future__ import annotations

import textwrap

from ._param_utils import extract_hf_dims, first_int
from ._solution_utils import direct_function_solution, has_inputs
from .extractors import (
    extract_cascade_merge_state_params,
    extract_observed_kwargs,
    extract_observed_param_names,
)
from parse.inventory_helpers import observation_fields

OP_TYPE = "cascade_merge"


def pr_reference_source() -> str:
    return "FlashInfer cascade merge unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} cascade merge state"


def pr_baseline_description(_def_name: str) -> str:
    return "flashinfer.cascade.merge_state"


def definition_name(variant: str, params: dict) -> str:
    """Return the canonical cascade merge definition name."""
    if variant != "merge_state":
        raise ValueError(f"Unsupported cascade_merge variant: {variant}")
    return f"merge_state_h{params['num_heads']}_d{params['head_dim']}"


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    if trace_id == "flashinfer.cascade.merge_state":
        return (OP_TYPE, "merge_state")
    return None


def resolve_cascade_params(
    *,
    raw_params: dict | None = None,
    config: dict | None = None,
    tp: int | None = None,
) -> dict[str, int] | None:
    """Resolve canonical cascade merge-state params owned by this adapter."""
    raw = raw_params or {}
    dims = extract_hf_dims(config or {}, tp)
    h = first_int(
        raw.get("num_heads"),
        raw.get("num_q_heads"),
        raw.get("tp_q_head_num"),
        raw.get("tp_num_attention_heads"),
        dims.get("tp_num_attention_heads"),
    )
    d = first_int(raw.get("head_dim"), raw.get("v_head_dim"), dims.get("head_dim"))
    if not (h and d):
        return None
    return {"num_heads": h, "head_dim": d}


def resolve_config_params(
    kernel: dict,
    config: dict,
    tp_size: int | None,
) -> dict[str, int] | None:
    """Resolve missing cascade merge params from HF config."""
    return resolve_cascade_params(
        raw_params=kernel.get("params"),
        config=config,
        tp=tp_size,
    )


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    kernels: list[dict] = []
    for key in [k for k in matched_kernels if k.startswith(f"{OP_TYPE}:")]:
        info = matched_kernels[key]
        variant = info["variant"]
        if variant != "merge_state":
            continue
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        emitted = False
        for num_heads, head_dim in sorted(
            extract_cascade_merge_state_params(info["signatures"])
        ):
            params = resolve_cascade_params(
                raw_params={"num_heads": num_heads, "head_dim": head_dim}
            )
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
            kernels.append({
                "op_type": OP_TYPE,
                "variant": variant,
                "fi_api": info["fi_api"],
                "params": {},
                "definition_name": "cascade_merge_state_NEEDS_CONFIG",
                **observation_fields(info, obs_kw, obs_pn),
                "note": "Definition name requires observed state shape or HF config params.",
            })
    return kernels


def _try_official_template(
    *,
    name: str,
    fi_api: str,
    num_heads: int,
    head_dim: int,
) -> dict | None:
    if not all(isinstance(v, int) for v in (num_heads, head_dim)):
        return None

    import torch
    from adapters.official_templates import render_definition

    return render_definition(
        "cascade",
        "merge_state_trace",
        fi_api,
        {
            "v_a": torch.empty((1, num_heads, head_dim), dtype=torch.bfloat16),
            "s_a": torch.empty((1, num_heads), dtype=torch.float32),
            "v_b": torch.empty((1, num_heads, head_dim), dtype=torch.bfloat16),
            "s_b": torch.empty((1, num_heads), dtype=torch.float32),
        },
        name=name,
    )


def _fill_cascade_merge_descriptions(definition: dict) -> None:
    """Fill validator-required descriptions omitted by the upstream template."""
    axis_descriptions = {
        "num_heads": "Number of attention heads.",
        "head_dim": "Per-head hidden dimension.",
    }
    output_descriptions = {
        "v_merged": "Merged attention output state.",
        "s_merged": "Merged logsumexp state.",
    }
    for name, description in axis_descriptions.items():
        definition.get("axes", {}).get(name, {}).setdefault("description", description)
    for name, description in output_descriptions.items():
        definition.get("outputs", {}).get(name, {}).setdefault("description", description)


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    variant = kernel["variant"]
    if variant != "merge_state":
        raise ValueError(f"Unsupported cascade_merge variant: {variant}")

    p = kernel["params"]
    num_heads = p["num_heads"]
    head_dim = p["head_dim"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]
    official = _try_official_template(
        name=name,
        fi_api=fi_api,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    if official is not None:
        _fill_cascade_merge_descriptions(official)
        return official

    return {
        "name": name,
        "description": "Merge two attention (V, S) states for cascade/speculative attention.",
        "op_type": OP_TYPE,
        "tags": [f"fi_api:{fi_api}", "status:verified"],
        "axes": {
            "seq_len": {
                "type": "var",
                "description": "Number of query tokens.",
            },
            "num_heads": {
                "type": "const",
                "value": num_heads,
                "description": "Number of attention heads.",
            },
            "head_dim": {
                "type": "const",
                "value": head_dim,
                "description": "Per-head hidden dimension.",
            },
        },
        "inputs": {
            "v_a": {
                "shape": ["seq_len", "num_heads", "head_dim"],
                "dtype": "bfloat16",
                "description": "Attention output from KV segment A.",
            },
            "s_a": {
                "shape": ["seq_len", "num_heads"],
                "dtype": "float32",
                "description": "Logsumexp (base-2) from KV segment A.",
            },
            "v_b": {
                "shape": ["seq_len", "num_heads", "head_dim"],
                "dtype": "bfloat16",
                "description": "Attention output from KV segment B.",
            },
            "s_b": {
                "shape": ["seq_len", "num_heads"],
                "dtype": "float32",
                "description": "Logsumexp (base-2) from KV segment B.",
            },
        },
        "outputs": {
            "v_merged": {
                "shape": ["seq_len", "num_heads", "head_dim"],
                "dtype": "bfloat16",
                "description": "Merged attention output state.",
            },
            "s_merged": {
                "shape": ["seq_len", "num_heads"],
                "dtype": "float32",
                "description": "Merged logsumexp state.",
            },
        },
        "reference": textwrap.dedent("""\
            import torch
            import math

            @torch.no_grad()
            def run(v_a, s_a, v_b, s_b):
                \"\"\"Merge two attention (V, S) states via numerically stable log-sum-exp.\"\"\"
                # s_a, s_b are log2-scale logsumexp values; convert to natural scale
                s_a = s_a.to(torch.float32) * math.log(2.0)
                s_b = s_b.to(torch.float32) * math.log(2.0)
                v_a = v_a.to(torch.float32)
                v_b = v_b.to(torch.float32)
                s_max = torch.maximum(s_a, s_b)
                exp_a = torch.exp(s_a - s_max)
                exp_b = torch.exp(s_b - s_max)
                exp_sum = exp_a + exp_b
                v_merged = (
                    v_a * exp_a.unsqueeze(-1) + v_b * exp_b.unsqueeze(-1)
                ) / exp_sum.unsqueeze(-1)
                s_merged = (s_max + torch.log(exp_sum)) / math.log(2.0)
                return v_merged.to(v_a.dtype), s_merged.to(torch.float32)
            """),
    }


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate a direct FlashInfer cascade merge baseline solution."""
    if definition.get("op_type") != OP_TYPE:
        return None
    if not has_inputs(definition, {"v_a", "s_a", "v_b", "s_b"}):
        return None
    return direct_function_solution(
        definition,
        fi_api="flashinfer.cascade.merge_state",
        args=["v_a", "s_a", "v_b", "s_b"],
        description="Baseline solution using FlashInfer cascade.merge_state.",
    )
