"""Adapter for FlashInfer sampling kernels."""

from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from typing import Any

from ._param_utils import analysis_allows_family, first_int, infer_vocab_size
from adapters.official_templates import render_definition
from ._solution_utils import has_inputs, solution_payload
from .extractors import extract_observed_kwargs, extract_observed_param_names
from parse.inventory_helpers import observation_fields

OP_TYPE = "sampling"

_OFFICIAL_CONTROL_KWARGS = {
    "indices",
    "filter_apply_order",
    "deterministic",
    "generator",
    "check_nan",
    "seed",
    "offset",
    "return_valid",
}


def pr_reference_source() -> str:
    return "FlashInfer sampling unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    return f"{model_label} sampling"


def pr_baseline_description(def_name: str) -> str:
    if "top_k_top_p" in def_name:
        return "flashinfer.sampling.top_k_top_p_sampling_from_probs"
    if "top_k" in def_name:
        return "flashinfer.sampling.top_k_sampling_from_probs"
    if "top_p" in def_name:
        return "flashinfer.sampling.top_p_sampling_from_probs"
    return "flashinfer.sampling"


def eval_validation_policy(_definition: dict) -> str:
    return "sampling_support"


def ignored_observed_kwargs(_kernel: dict, _definition: dict) -> set[str]:
    """Runtime controls accepted by FlashInfer but excluded from official templates."""
    return set(_OFFICIAL_CONTROL_KWARGS)


def ignored_definition_inputs_for_observed_params(
    kernel: dict,
    definition: dict,
) -> set[str]:
    """Inputs stored inside SGLang SamplingBatchInfo at runtime."""
    observed = set(kernel.get("observed_param_names") or [])
    if "sampling_info" not in observed:
        return set()
    return set(definition.get("inputs", {})) - {"probs"}


_TEMPLATE_BY_VARIANT = {
    "top_k": "top_k_sampling_trace",
    "top_p": "top_p_sampling_trace",
    "top_k_top_p": "top_k_top_p_sampling_trace",
    "min_p": "min_p_sampling_trace",
    "softmax": "softmax_trace",
    "sampling_from_probs": "sampling_from_probs_trace",
    "sampling_from_logits": "sampling_from_logits_trace",
    "top_p_renorm_probs": "top_p_renorm_probs_trace",
    "top_k_renorm_probs": "top_k_renorm_probs_trace",
    "top_k_mask_logits": "top_k_mask_logits_trace",
    "top_k_top_p_sampling_from_logits": "top_k_top_p_sampling_from_logits_trace",
}

_NAME_PREFIX_BY_VARIANT = {
    "top_k": "top_k_sampling",
    "top_p": "top_p_sampling",
    "top_k_top_p": "top_k_top_p_sampling",
}


def _snake_tokens(name: str) -> set[str]:
    return {piece for piece in re.split(r"[_\W]+", name.lower()) if piece}


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    if trace_id == "sglang.srt.layers.sampler.Sampler._sample_from_probs":
        return (OP_TYPE, "from_probs_dispatch")

    parts = trace_id.split(".")
    if len(parts) < 3:
        return None
    if parts[0] == "flashinfer" and parts[1] == "sampling":
        func = parts[-1]
    elif parts[0] == "sgl_kernel" and parts[1] == "sampling":
        func = parts[-1]
    else:
        return None

    tokens = _snake_tokens(func)
    if {"top", "k", "p"}.issubset(tokens):
        return (OP_TYPE, "top_k_top_p")
    if {"top", "k"}.issubset(tokens):
        return (OP_TYPE, "top_k")
    if {"top", "p"}.issubset(tokens):
        return (OP_TYPE, "top_p")
    if {"min", "p"}.issubset(tokens):
        return (OP_TYPE, "min_p")
    return None


def _tensor_vocab_size(value: Any) -> int | None:
    if isinstance(value, dict) and value.get("type") == "tensor":
        shape = value.get("shape", [])
        if isinstance(shape, list) and len(shape) == 2 and isinstance(shape[1], int):
            return shape[1]
    return None


def _extract_vocab_sizes(signatures: list[dict]) -> set[int]:
    sizes: set[int] = set()
    for sig in signatures:
        s = sig.get("signature", {})
        args = s.get("args", [])
        for arg in args:
            vocab_size = _tensor_vocab_size(arg)
            if vocab_size is not None:
                sizes.add(vocab_size)
                break
    return sizes


def _sampling_variant_from_info_arg(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    attrs = value.get("attrs")
    if not isinstance(attrs, dict):
        return None
    need_top_k = attrs.get("need_top_k_sampling")
    need_top_p = attrs.get("need_top_p_sampling")
    if need_top_k and need_top_p:
        return "top_k_top_p"
    if need_top_k:
        return "top_k"
    if need_top_p:
        return "top_p"
    return None


def _extract_dispatch_variant_vocab_sizes(signatures: list[dict]) -> dict[str, set[int]]:
    sizes: dict[str, set[int]] = defaultdict(set)
    for sig in signatures:
        body = sig.get("signature", {})
        args = body.get("args", []) if isinstance(body, dict) else []
        if len(args) < 2:
            continue
        vocab_size = _tensor_vocab_size(args[0])
        variant = _sampling_variant_from_info_arg(args[1])
        if vocab_size is not None and variant is not None:
            sizes[variant].add(vocab_size)
    return sizes


def definition_name(variant: str, params: dict) -> str:
    """Return the canonical sampling definition name for resolved params."""
    prefix = _NAME_PREFIX_BY_VARIANT.get(variant, f"{variant}_sampling")
    return f"{prefix}_v{params['vocab_size']}"


def fi_api_for_variant(variant: str) -> str | None:
    """Return the FlashInfer API tag used for static candidates."""
    return {
        "top_k_top_p": "flashinfer.sampling.top_k_top_p_sampling_from_probs",
        "top_k": "flashinfer.sampling.top_k_sampling_from_probs",
        "top_p": "flashinfer.sampling.top_p_sampling_from_probs",
    }.get(variant)


def resolve_sampling_params(
    *,
    raw_params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, int] | None:
    """Resolve canonical sampling params owned by this adapter."""
    raw = raw_params or {}
    vocab = first_int(raw.get("vocab_size"), infer_vocab_size(config or {}))
    return {"vocab_size": vocab} if vocab else None


def resolve_config_params(
    kernel: dict,
    config: dict,
    _tp_size: int | None,
) -> dict[str, int] | None:
    """Resolve missing sampling params from HF config."""
    return resolve_sampling_params(raw_params=kernel.get("params"), config=config)


def static_candidates(
    dims: dict,
    analysis: dict,
    *,
    tp: int,
    page_size: int,
) -> list[dict]:
    """Build sampling static candidates from HF config dimensions."""
    params = resolve_sampling_params(raw_params=dims)
    if not params or not analysis_allows_family(analysis, "has_sampling"):
        return []
    return [
        {
            "op_type": OP_TYPE,
            "variant": variant,
            "fi_api": fi_api_for_variant(variant),
            "params": params,
            "definition_name": definition_name(variant, params),
        }
        for variant in ("top_k", "top_p", "top_k_top_p")
    ]


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    kernels: list[dict] = []
    for key in [k for k in matched_kernels if k.startswith(f"{OP_TYPE}:")]:
        info = matched_kernels[key]
        obs_kw = extract_observed_kwargs(info["signatures"])
        obs_pn = extract_observed_param_names(info["signatures"])
        variant_vocab_sizes: dict[str, set[int]]
        if info["variant"] == "from_probs_dispatch":
            variant_vocab_sizes = _extract_dispatch_variant_vocab_sizes(info["signatures"])
        else:
            variant_vocab_sizes = {
                info["variant"]: _extract_vocab_sizes(info["signatures"])
            }

        for variant, vocab_sizes in variant_vocab_sizes.items():
            fi_api = info["fi_api"] or fi_api_for_variant(variant)
            if fi_api is None:
                continue
            emitted = False
            for vocab_size in sorted(vocab_sizes):
                params = resolve_sampling_params(raw_params={"vocab_size": vocab_size})
                if not params:
                    continue
                emitted = True
                kernels.append({
                    "op_type": OP_TYPE,
                    "variant": variant,
                    "fi_api": fi_api,
                    "params": params,
                    "definition_name": definition_name(variant, params),
                    **observation_fields(info, obs_kw, obs_pn),
                })
            if not emitted:
                prefix = _NAME_PREFIX_BY_VARIANT.get(variant, f"{variant}_sampling")
                kernels.append({
                    "op_type": OP_TYPE,
                    "variant": variant,
                    "fi_api": fi_api,
                    "params": {},
                    "definition_name": f"{prefix}_NEEDS_CONFIG",
                    **observation_fields(info, obs_kw, obs_pn),
                    "note": "Definition name requires observed vocab_size or HF config params.",
                })
    return kernels


def _try_official_template(
    *,
    name: str,
    fi_api: str,
    variant: str,
    vocab_size: Any,
) -> dict | None:
    """Render sampling definitions from FlashInfer's official TraceTemplate."""
    if not isinstance(vocab_size, int):
        return None
    template_name = _TEMPLATE_BY_VARIANT.get(variant)
    if template_name is None:
        return None

    import torch

    batch_size = 1
    probs = torch.full((batch_size, vocab_size), 1.0 / vocab_size, dtype=torch.float32)
    logits = torch.empty((batch_size, vocab_size), dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32)
    top_k = torch.tensor([min(50, vocab_size)], dtype=torch.int32)
    top_p = torch.tensor([0.9], dtype=torch.float32)

    if variant == "top_k":
        kwargs: dict[str, Any] = {"probs": probs, "top_k": top_k}
    elif variant == "top_p":
        kwargs = {"probs": probs, "top_p": top_p}
    elif variant == "top_k_top_p":
        kwargs = {"probs": probs, "top_k": top_k, "top_p": top_p}
    elif variant == "min_p":
        kwargs = {"probs": probs, "min_p": 0.1, "indices": indices}
    elif variant == "softmax":
        kwargs = {"logits": logits, "temperature": 1.0}
    elif variant == "sampling_from_probs":
        kwargs = {"probs": probs, "indices": indices}
    elif variant == "sampling_from_logits":
        kwargs = {"logits": logits, "indices": indices}
    elif variant == "top_p_renorm_probs":
        kwargs = {"probs": probs, "top_p": 0.9}
    elif variant in ("top_k_renorm_probs", "top_k_mask_logits"):
        kwargs = {
            "probs" if variant == "top_k_renorm_probs" else "logits": (
                probs if variant == "top_k_renorm_probs" else logits
            ),
            "top_k": min(50, vocab_size),
        }
    elif variant == "top_k_top_p_sampling_from_logits":
        kwargs = {
            "logits": logits,
            "top_k": min(50, vocab_size),
            "top_p": 0.9,
            "indices": indices,
        }
    else:
        return None

    return render_definition(
        "sampling",
        template_name,
        fi_api,
        kwargs,
        name=name,
    )


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    """Generate a sampling definition."""
    p = kernel["params"]
    v = p["vocab_size"]
    variant = kernel["variant"]
    name = kernel["definition_name"]
    fi_api = kernel["fi_api"]
    official = _try_official_template(
        name=name,
        fi_api=fi_api,
        variant=variant,
        vocab_size=v,
    )
    return official


def _build_sampling_solution_source(variant: str) -> str | None:
    """Build FlashInfer sampling baseline source for supported variants."""
    if variant == "top_k_top_p":
        return textwrap.dedent("""\
            import torch
            import flashinfer


            def run(probs, top_k, top_p):
                samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                    probs=probs.to(torch.float32),
                    top_k=top_k,
                    top_p=top_p,
                    indices=None,
                    filter_apply_order="top_k_first",
                    deterministic=False,
                    generator=None,
                    check_nan=False,
                )
                return samples.to(torch.int64)
            """) + "\n"
    if variant == "top_k":
        return textwrap.dedent("""\
            import torch
            import flashinfer


            def run(probs, top_k):
                samples = flashinfer.sampling.top_k_sampling_from_probs(
                    probs=probs.to(torch.float32),
                    top_k=top_k,
                    indices=None,
                    deterministic=False,
                    generator=None,
                    check_nan=False,
                )
                return samples.to(torch.int64)
            """) + "\n"
    if variant == "top_p":
        return textwrap.dedent("""\
            import torch
            import flashinfer


            def run(probs, top_p):
                samples = flashinfer.sampling.top_p_sampling_from_probs(
                    probs=probs.to(torch.float32),
                    top_p=top_p,
                    indices=None,
                    deterministic=False,
                    generator=None,
                    check_nan=False,
                )
                return samples.to(torch.int64)
            """) + "\n"
    return None


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate FlashInfer sampling baseline solution source."""
    if definition.get("op_type") != OP_TYPE:
        return None
    name = str(definition.get("name", ""))
    if name.startswith("top_k_top_p"):
        variant = "top_k_top_p"
        required = {"probs", "top_k", "top_p"}
    elif name.startswith("top_k"):
        variant = "top_k"
        required = {"probs", "top_k"}
    elif name.startswith("top_p"):
        variant = "top_p"
        required = {"probs", "top_p"}
    else:
        return None
    if not has_inputs(definition, required):
        return None
    source = _build_sampling_solution_source(variant)
    if source is None:
        return None
    return solution_payload(
        definition,
        source,
        description=f"Baseline solution using FlashInfer sampling {variant}.",
        validation_policy=eval_validation_policy(definition),
    )
