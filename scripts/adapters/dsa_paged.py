"""Adapter for FlashInfer DSA (Dynamic Sparse Attention) paged kernels."""
from __future__ import annotations

import re
import textwrap

from typing import Any

from ._param_utils import analysis_allows_family, extract_hf_dims, first_present, is_unknown
from .mla_paged import resolve_mla_params
from adapters.official_templates import render_definition
from ._solution_utils import has_inputs, solution_payload
from .extractors import extract_observed_kwargs, extract_observed_param_names
from parse.inventory_helpers import observation_fields

OP_TYPE = "dsa_paged"
_TOPK_INDEXER_VARIANT = "topk_indexer"
_TOPK_INDEXER_NUM_HEADS = 64
_TOPK_INDEXER_HEAD_DIM = 128
_TOPK_INDEXER_PAGE_SIZE = 64
_TOPK_INDEXER_TOPK = 2048


def pr_reference_source() -> str:
    return "FlashInfer DSA unit tests"


def pr_kernel_description(_def_name: str, model_label: str) -> str:
    if "topk_indexer" in _def_name:
        return f"{model_label} DSA top-k indexer"
    return f"{model_label} DSA sparse attention"


def pr_baseline_description(_def_name: str) -> str:
    if "topk_indexer" in _def_name:
        return "DeepGEMM logits + flashinfer top_k_page_table_transform"
    return "flashinfer DSA sparse attention wrapper"


_WRAPPER_METHODS = {
    "__call__",
    "forward",
    "forward_decode",
    "forward_extend",
    "forward_return_lse",
    "run",
}

_CLASS_VARIANTS = {
    "flashinfer.sparse.BlockSparseAttentionWrapper": "decode",
    "flashinfer.sparse.BatchMLASparseDecodeWrapper": "decode",
}


def _class_name_tokens(name: str) -> set[str]:
    pieces = re.findall(
        r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+",
        name.replace("_", " "),
    )
    return {piece.lower() for piece in pieces}


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    if trace_id in _CLASS_VARIANTS:
        return (OP_TYPE, _CLASS_VARIANTS[trace_id])

    class_path, sep, method = trace_id.rpartition(".")
    if not sep or method not in _WRAPPER_METHODS:
        return None
    variant = _CLASS_VARIANTS.get(class_path)
    if variant is not None:
        return (OP_TYPE, variant)

    parts = trace_id.split(".")
    if len(parts) < 4 or parts[0] != "flashinfer" or parts[1] != "sparse":
        return None
    if method != "run":
        return None
    tokens = _class_name_tokens(parts[2])
    if "prefill" in tokens:
        return (OP_TYPE, "prefill")
    if "decode" in tokens:
        return (OP_TYPE, "decode")
    if "sparse" in tokens and ("attention" in tokens or "wrapper" in tokens):
        return (OP_TYPE, "decode")
    return None


def definition_name(_variant: str, params: dict) -> str:
    """Return the canonical DSA paged definition name for resolved params."""
    if _variant == _TOPK_INDEXER_VARIANT:
        h = params["num_index_heads"]
        d = params["index_head_dim"]
        topk = params["topk"]
        ps = params["page_size"]
        return f"dsa_topk_indexer_fp8_h{h}_d{d}_topk{topk}_ps{ps}"
    h = params["num_heads"]
    ckv = params["ckv_dim"]
    kpe = params["kpe_dim"]
    topk = params["topk"]
    ps = params["page_size"]
    return f"dsa_sparse_attention_h{h}_ckv{ckv}_kpe{kpe}_topk{topk}_ps{ps}"


def fi_api_for_variant(variant: str) -> str | None:
    """Return the FlashInfer API tag used for static candidates."""
    return {
        "decode": "flashinfer.sparse.BlockSparseAttentionWrapper",
        "prefill": "flashinfer.sparse.BlockSparseAttentionWrapper",
        _TOPK_INDEXER_VARIANT: "flashinfer.top_k_page_table_transform",
    }.get(variant)


def resolve_dsa_params(
    *,
    raw_params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    tp: int | None = None,
    page_size: int = 1,
) -> dict[str, Any] | None:
    """Resolve canonical DSA paged params owned by this adapter."""
    raw = raw_params or {}
    mla = resolve_mla_params(raw_params=raw, config=config, tp=tp, page_size=page_size)
    if not mla:
        return None
    dims = extract_hf_dims(config or {}, tp)
    topk = first_present(raw.get("topk"), raw.get("top_k"), raw.get("dsa_topk"), dims.get("dsa_topk"))
    ps = first_present(raw.get("page_size"), raw.get("dsa_page_size"), dims.get("dsa_page_size"), mla["page_size"])
    if is_unknown(topk) or is_unknown(ps):
        return None
    return {
        "num_heads": mla["num_heads"],
        "ckv_dim": mla["ckv_dim"],
        "kpe_dim": mla["kpe_dim"],
        "topk": topk,
        "page_size": ps,
    }


def resolve_dsa_topk_indexer_params(
    *,
    raw_params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    tp: int | None = None,
) -> dict[str, int] | None:
    """Resolve the DeepGEMM-backed DSA top-k indexer dimensions."""
    raw = raw_params or {}
    dims = extract_hf_dims(config or {}, tp)
    topk = first_present(
        raw.get("topk"),
        raw.get("top_k"),
        raw.get("dsa_topk"),
        dims.get("dsa_topk"),
        _TOPK_INDEXER_TOPK,
    )
    ps = first_present(
        raw.get("page_size"),
        raw.get("dsa_page_size"),
        dims.get("dsa_page_size"),
        _TOPK_INDEXER_PAGE_SIZE,
    )
    h = first_present(
        raw.get("num_index_heads"),
        raw.get("index_heads"),
        _TOPK_INDEXER_NUM_HEADS,
    )
    d = first_present(
        raw.get("index_head_dim"),
        raw.get("head_dim"),
        _TOPK_INDEXER_HEAD_DIM,
    )
    if any(is_unknown(value) for value in (topk, ps, h, d)):
        return None
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (topk, ps, h, d)
    ):
        return None
    return {
        "num_index_heads": h,
        "index_head_dim": d,
        "topk": topk,
        "page_size": ps,
    }


def resolve_config_params(
    kernel: dict,
    config: dict,
    tp_size: int | None,
) -> dict | None:
    """Resolve missing DSA paged params from HF config."""
    if kernel.get("variant") == _TOPK_INDEXER_VARIANT:
        return resolve_dsa_topk_indexer_params(
            raw_params=kernel.get("params"),
            config=config,
            tp=tp_size,
        )
    return resolve_dsa_params(
        raw_params=kernel.get("params"),
        config=config,
        tp=tp_size,
        page_size=1,
    )


def diagnose_config_resolution(
    kernel: dict,
    config: dict,
    tp_size: int | None,
) -> dict:
    """Return DSA-specific missing canonical params for inventory diagnostics."""
    if kernel.get("variant") == _TOPK_INDEXER_VARIANT:
        params = resolve_dsa_topk_indexer_params(
            raw_params=kernel.get("params"),
            config=config,
            tp=tp_size,
        )
        return {
            "missing": []
            if params
            else ["num_index_heads", "index_head_dim", "topk", "page_size"],
        }

    raw = kernel.get("params") or {}
    mla = resolve_mla_params(raw_params=raw, config=config, tp=tp_size, page_size=1)
    dims = extract_hf_dims(config or {}, tp_size)
    missing = []
    if not mla:
        missing.extend(["num_heads", "ckv_dim", "kpe_dim", "page_size"])
    topk = first_present(
        raw.get("topk"),
        raw.get("top_k"),
        raw.get("dsa_topk"),
        dims.get("dsa_topk"),
    )
    if is_unknown(topk):
        missing.append("topk")
    ps = first_present(
        raw.get("page_size"),
        raw.get("dsa_page_size"),
        dims.get("dsa_page_size"),
        mla["page_size"] if mla else None,
    )
    if is_unknown(ps) and "page_size" not in missing:
        missing.append("page_size")
    return {"missing": missing}


def static_candidates(
    dims: dict,
    analysis: dict,
    *,
    tp: int,
    page_size: int,
) -> list[dict]:
    """Build DSA paged static candidates from HF config dimensions."""
    params = resolve_dsa_params(raw_params=dims, tp=tp, page_size=page_size)
    if not analysis_allows_family(analysis, "has_dsa"):
        return []
    candidates = []
    if params:
        variant = "decode"
        candidates.append({
            "op_type": OP_TYPE,
            "variant": variant,
            "fi_api": fi_api_for_variant(variant),
            "params": params,
            "definition_name": definition_name(variant, params),
        })

    indexer_params = resolve_dsa_topk_indexer_params(raw_params=dims, tp=tp)
    if indexer_params:
        variant = _TOPK_INDEXER_VARIANT
        candidates.append({
            "op_type": OP_TYPE,
            "variant": variant,
            "fi_api": fi_api_for_variant(variant),
            "params": indexer_params,
            "definition_name": definition_name(variant, indexer_params),
            "page_size_source": "deep_gemm_required",
        })
    return candidates


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
            "definition_name": f"dsa_{info['variant']}_NEEDS_CONFIG",
            **observation_fields(info, obs_kw, obs_pn),
            "note": "Definition name requires HF config.json params.",
        })
    return kernels


def _try_official_template(
    *,
    name: str,
    fi_api: str,
    h: Any,
    ckv: Any,
    kpe: Any,
    topk: Any,
    page_size: Any,
) -> dict | None:
    """Render DSA paged attention from FlashInfer's official TraceTemplate."""
    if not all(isinstance(v, int) for v in (h, ckv, kpe, topk, page_size)):
        return None
    import torch

    return render_definition(
        "attention",
        "dsa_paged_trace",
        fi_api,
        {
            "q_nope": torch.empty((1, h, ckv), dtype=torch.bfloat16),
            "q_pe": torch.empty((1, h, kpe), dtype=torch.bfloat16),
            "ckv_cache": torch.empty((1, page_size, ckv), dtype=torch.bfloat16),
            "kpe_cache": torch.empty((1, page_size, kpe), dtype=torch.bfloat16),
            "sparse_indices": torch.zeros((1, topk), dtype=torch.int32),
            "sm_scale": float(ckv + kpe) ** -0.5,
        },
        name=name,
    )


def _build_dsa_topk_indexer_reference(topk: int) -> str:
    """Build a torch reference for the FP8 DSA top-k indexer."""
    return textwrap.dedent(f"""\
        import torch


        def dequant_fp8_kv_cache(k_index_cache_fp8):
            \"\"\"Dequantize deep_gemm FP8 KV cache packed as int8 bytes.\"\"\"
            cache = k_index_cache_fp8.view(torch.uint8)
            num_pages, page_size, _num_heads, head_dim_with_scale = cache.shape
            head_dim = head_dim_with_scale - 4

            flat = cache.reshape(num_pages, page_size * head_dim_with_scale)
            fp8_bytes = flat[:, :page_size * head_dim].contiguous()
            fp8_values = (
                fp8_bytes.reshape(num_pages, page_size, head_dim)
                .view(torch.float8_e4m3fn)
                .to(torch.float32)
            )

            scale_bytes = flat[:, page_size * head_dim:].contiguous()
            scales = scale_bytes.reshape(num_pages, page_size, 4).view(torch.float32)
            return fp8_values * scales


        @torch.no_grad()
        def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
            batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
            _num_pages, page_size, _kv_heads, _head_dim_with_scale = k_index_cache_fp8.shape
            topk = {topk}

            assert num_index_heads == {_TOPK_INDEXER_NUM_HEADS}
            assert index_head_dim == {_TOPK_INDEXER_HEAD_DIM}
            assert page_size == {_TOPK_INDEXER_PAGE_SIZE}

            device = q_index_fp8.device
            q = q_index_fp8.to(torch.float32)
            k_all = dequant_fp8_kv_cache(k_index_cache_fp8)
            topk_indices = torch.full(
                (batch_size, topk), -1, dtype=torch.int32, device=device
            )

            for b in range(batch_size):
                seq_len = int(seq_lens[b].item())
                if seq_len <= 0:
                    continue

                num_pages_for_seq = (seq_len + page_size - 1) // page_size
                page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
                k_tokens = k_all[page_indices].reshape(-1, index_head_dim)[:seq_len]

                scores = torch.relu(q[b] @ k_tokens.T)
                weighted_scores = scores * weights[b].to(torch.float32)[:, None]
                final_scores = weighted_scores.sum(dim=0)

                actual_topk = min(topk, seq_len)
                _values, local_indices = torch.topk(final_scores, actual_topk)
                page_offsets = local_indices // page_size
                token_offsets = local_indices % page_size
                global_pages = page_indices[page_offsets]
                topk_tokens = global_pages * page_size + token_offsets
                topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

            return (topk_indices,)
        """)


def _generate_topk_indexer_definition(name: str, params: dict[str, int], model_tag: str) -> dict:
    h = params["num_index_heads"]
    d = params["index_head_dim"]
    topk = params["topk"]
    page_size = params["page_size"]
    head_dim_with_scale = d + 4

    return {
        "name": name,
        "description": (
            "Native Sparse Attention (DSA) TopK indexer with FP8 quantization. "
            "Computes sparse attention scores using ReLU activation and learned "
            "weights, then selects top-K KV cache indices."
        ),
        "op_type": OP_TYPE,
        "tags": [
            "stage:indexer",
            "status:verified",
            f"model:{model_tag}",
            "sparse:topk",
            "quant:fp8",
        ],
        "axes": {
            "batch_size": {"type": "var"},
            "num_index_heads": {
                "type": "const",
                "value": h,
                "description": "Number of indexer heads required by deep_gemm.",
            },
            "index_head_dim": {
                "type": "const",
                "value": d,
                "description": "Indexer head dimension required by deep_gemm.",
            },
            "page_size": {
                "type": "const",
                "value": page_size,
                "description": "KV cache page size required by deep_gemm.",
            },
            "topk": {
                "type": "const",
                "value": topk,
                "description": "Number of top-K indices to select.",
            },
            "max_num_pages": {
                "type": "var",
                "description": "Maximum number of pages per sequence.",
            },
            "num_pages": {
                "type": "var",
                "description": "Total number of allocated KV cache pages.",
            },
            "kv_cache_num_heads": {
                "type": "const",
                "value": 1,
                "description": "KV cache head count for deep_gemm MQA format.",
            },
            "head_dim_with_scale": {
                "type": "const",
                "value": head_dim_with_scale,
                "description": "Indexer head dimension plus FP8 scale bytes.",
            },
        },
        "constraints": ["topk <= max_num_pages * page_size"],
        "inputs": {
            "q_index_fp8": {
                "shape": ["batch_size", "num_index_heads", "index_head_dim"],
                "dtype": "float8_e4m3fn",
                "description": "FP8 quantized query tensor for indexing.",
            },
            "k_index_cache_fp8": {
                "shape": [
                    "num_pages",
                    "page_size",
                    "kv_cache_num_heads",
                    "head_dim_with_scale",
                ],
                "dtype": "int8",
                "description": (
                    "FP8 key index cache in deep_gemm packed format, represented "
                    "as int8 bytes."
                ),
            },
            "weights": {
                "shape": ["batch_size", "num_index_heads"],
                "dtype": "float32",
                "description": "Learned per-index-head weights.",
            },
            "seq_lens": {
                "shape": ["batch_size"],
                "dtype": "int32",
                "description": "Sequence lengths for each batch element.",
            },
            "block_table": {
                "shape": ["batch_size", "max_num_pages"],
                "dtype": "int32",
                "description": "Page-level block table mapping sequences to KV cache pages.",
            },
        },
        "outputs": {
            "topk_indices": {
                "shape": ["batch_size", "topk"],
                "dtype": "int32",
                "description": "Top-K token indices. Values of -1 indicate padding.",
            },
        },
        "reference": _build_dsa_topk_indexer_reference(topk),
    }


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
    """Generate a DSA (DeepSeek Sparse Attention) paged definition."""
    p = kernel["params"]
    variant = kernel["variant"]
    name = kernel["definition_name"]
    if variant == _TOPK_INDEXER_VARIANT:
        return _generate_topk_indexer_definition(name, p, model_tag)

    fi_api = kernel["fi_api"]

    # DSA params will need HF config enrichment
    h = p.get("num_heads", "?")
    ckv = p.get("ckv_dim", "?")
    kpe = p.get("kpe_dim", "?")
    topk = p.get("topk", "?")
    page_size = p.get("page_size", 1)
    official = _try_official_template(
        name=name,
        fi_api=fi_api,
        h=h,
        ckv=ckv,
        kpe=kpe,
        topk=topk,
        page_size=page_size,
    )
    if official is not None:
        return official

    tags = [
        f"fi_api:{fi_api}",
        "status:verified",
        "sparse:topk",
    ]

    axes: dict[str, Any] = {
        "num_tokens": {
            "type": "var",
            "description": "Number of tokens (batch_size for decode, total_num_tokens for prefill).",
        },
        "num_qo_heads": {
            "type": "const",
            "value": h,
            "description": "Number of query heads after tensor parallel split.",
        },
        "head_dim_ckv": {
            "type": "const",
            "value": ckv,
            "description": "Compressed KV head dimension.",
        },
        "head_dim_kpe": {
            "type": "const",
            "value": kpe,
            "description": "Key positional encoding dimension.",
        },
        "topk": {
            "type": "const",
            "value": topk,
            "description": "Number of top-K KV cache entries selected for sparse attention.",
        },
        "page_size": {
            "type": "const",
            "value": page_size,
            "description": "Page size for KV cache.",
        },
        "num_pages": {
            "type": "var",
            "description": "Total number of allocated pages in the KV cache.",
        },
    }

    constraints = [
        "sparse_indices.shape[0] == num_tokens",
        "sparse_indices.shape[-1] == topk",
        "ckv_cache.shape[1] == page_size",
    ]

    inputs: dict[str, Any] = {
        "q_nope": {
            "shape": ["num_tokens", "num_qo_heads", "head_dim_ckv"],
            "dtype": "bfloat16",
            "description": "Query tensor without positional encoding component.",
        },
        "q_pe": {
            "shape": ["num_tokens", "num_qo_heads", "head_dim_kpe"],
            "dtype": "bfloat16",
            "description": "Query positional encoding component.",
        },
        "ckv_cache": {
            "shape": ["num_pages", "page_size", "head_dim_ckv"],
            "dtype": "bfloat16",
            "description": "Compressed key-value cache.",
        },
        "kpe_cache": {
            "shape": ["num_pages", "page_size", "head_dim_kpe"],
            "dtype": "bfloat16",
            "description": "Key positional encoding cache.",
        },
        "sparse_indices": {
            "shape": ["num_tokens", "topk"],
            "dtype": "int32",
            "description": "Sparse indices selecting top-K KV cache entries per token. -1 = padding.",
        },
        "sm_scale": {
            "shape": None,
            "dtype": "float32",
            "description": "Softmax scale. For MLA pre-absorption: 1/sqrt(head_dim_qk + head_dim_kpe).",
        },
    }

    reference = textwrap.dedent("""\
        import torch
        import math

        @torch.no_grad()
        def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
            \"\"\"
            Batched Native Sparse Attention (DSA) reference implementation.

            Uses sparse_indices to select top-K KV cache entries per token.
            Values of -1 in sparse_indices indicate padding (ignored).
            \"\"\"
            num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
            head_dim_kpe = q_pe.shape[-1]
            device = q_nope.device

            # Squeeze page dimension when page_size=1; otherwise flatten pages.
            Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
            Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

            output = torch.zeros(
                (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
            )
            lse = torch.full(
                (num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
            )

            for t in range(num_tokens):
                indices = sparse_indices[t]
                valid_mask = indices != -1
                valid_indices = indices[valid_mask]
                if valid_indices.numel() == 0:
                    output[t].zero_()
                    continue
                tok_idx = valid_indices.to(torch.long)
                Kc = Kc_all[tok_idx]
                Kp = Kp_all[tok_idx]
                qn = q_nope[t].to(torch.float32)
                qp = q_pe[t].to(torch.float32)
                logits = (qn @ Kc.T) + (qp @ Kp.T)
                logits_scaled = logits * sm_scale
                lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                attn = torch.softmax(logits_scaled, dim=-1)
                output[t] = (attn @ Kc).to(torch.bfloat16)

            return output, lse
        """)

    return {
        "name": name,
        "op_type": OP_TYPE,
        "description": (
            "DSA (Dense Sparse Attention): MLA latent layout + per-query top-K "
            "selection via sparse_indices (-1 = padding). Covers decode and prefill; "
            "no kv_indptr/indices."
        ),
        "tags": tags,
        "axes": axes,
        "constraints": constraints,
        "inputs": inputs,
        "outputs": {
            "output": {
                "shape": ["num_tokens", "num_qo_heads", "head_dim_ckv"],
                "dtype": "bfloat16",
                "description": "Attention output tensor.",
            },
            "lse": {
                "shape": ["num_tokens", "num_qo_heads"],
                "dtype": "float32",
                "description": "The 2-based log-sum-exp of attention logits.",
            },
        },
        "reference": reference,
    }


def _build_dsa_solution_source(*, include_lse: bool) -> str:
    """Build FlashInfer-backed DSA sparse attention baseline source."""
    lse_block = ""
    return_expr = "(output,)"
    if include_lse:
        lse_block = textwrap.dedent("""\

            lse = torch.full(
                (num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
            )
            Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
            Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)
            for t in range(num_tokens):
                indices = sparse_indices[t]
                valid_indices = indices[indices != -1].to(torch.long)
                if valid_indices.numel() == 0:
                    continue
                Kc = Kc_all[valid_indices]
                Kp = Kp_all[valid_indices]
                qn = q_nope[t].to(torch.float32)
                qp = q_pe[t].to(torch.float32)
                logits = (qn @ Kc.T) + (qp @ Kp.T)
                logits = logits * bmm1_scale
                lse[t] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            """)
        return_expr = "output, lse"

    source = textwrap.dedent("""\
        import math
        import torch
        import flashinfer.decode

        _WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
        _workspace_cache = {}


        def _get_workspace(device):
            key = str(device)
            buf = _workspace_cache.get(key)
            if buf is None:
                buf = torch.zeros(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
                _workspace_cache[key] = buf
            return buf


        def _scalar(value):
            if isinstance(value, torch.Tensor):
                return float(value.item())
            return float(value)


        def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
            num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
            head_dim_kpe = q_pe.shape[-1]
            device = q_nope.device
            topk = sparse_indices.shape[-1]
            bmm1_scale = _scalar(sm_scale)

            query = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)
            kv_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)
            block_tables = sparse_indices.unsqueeze(1)
            seq_lens = (sparse_indices != -1).sum(dim=1).to(torch.int32)
            max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() else 0
            workspace = _get_workspace(device)

            output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=query,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                qk_nope_head_dim=head_dim_ckv,
                kv_lora_rank=head_dim_ckv,
                qk_rope_head_dim=head_dim_kpe,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                sparse_mla_top_k=topk,
                bmm1_scale=bmm1_scale,
            )
            output = output.squeeze(1)
        """)
    source += textwrap.indent(lse_block, "    ")
    source += f"\n    return {return_expr}\n"
    return source


def _axis_const(definition: dict, name: str, default: int) -> int:
    axis = definition.get("axes", {}).get(name, {})
    if isinstance(axis, dict) and isinstance(axis.get("value"), int):
        return int(axis["value"])
    return default


def _build_dsa_topk_indexer_solution_source(definition: dict) -> str:
    """Build DeepGEMM + FlashInfer top-k indexer baseline source."""
    topk = _axis_const(definition, "topk", 2048)
    return textwrap.dedent(f"""\
        import torch
        import deep_gemm
        import flashinfer

        _TOPK = {topk}


        @torch.no_grad()
        def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
            batch_size, _num_index_heads, _index_head_dim = q_index_fp8.shape
            _num_pages, page_size, _heads, _head_dim_sf = k_index_cache_fp8.shape
            device = q_index_fp8.device
            max_num_pages = block_table.shape[1]
            max_context_len = max_num_pages * page_size

            q_index_fp8_4d = q_index_fp8.unsqueeze(1)
            k_index_cache_uint8 = k_index_cache_fp8.view(torch.uint8)

            num_sms = torch.cuda.get_device_properties(device).multi_processor_count
            schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
                seq_lens, page_size, num_sms
            )
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_index_fp8_4d,
                k_index_cache_uint8,
                weights,
                seq_lens,
                block_table,
                schedule_meta,
                max_context_len,
                clean_logits=False,
            )

            offsets = torch.arange(page_size, device=device, dtype=torch.int32)
            physical = block_table.unsqueeze(-1) * page_size + offsets
            physical_flat = physical.reshape(batch_size, -1)
            token_indices = torch.arange(max_num_pages * page_size, device=device)
            mask = token_indices.unsqueeze(0) < seq_lens.unsqueeze(1)
            token_page_table = torch.where(
                mask, physical_flat, torch.zeros_like(physical_flat)
            )

            topk_indices = flashinfer.top_k_page_table_transform(
                input=logits.to(torch.float16),
                src_page_table=token_page_table,
                lengths=seq_lens,
                k=_TOPK,
            )
            return (topk_indices,)
        """) + "\n"


def generate_baseline_solution(definition: dict) -> dict | None:
    """Generate a FlashInfer-backed DSA sparse attention baseline solution."""
    if definition.get("op_type") != OP_TYPE:
        return None
    if has_inputs(
        definition,
        {"q_index_fp8", "k_index_cache_fp8", "weights", "seq_lens", "block_table"},
    ):
        source = _build_dsa_topk_indexer_solution_source(definition)
        return solution_payload(
            definition,
            source,
            name_prefix="flashinfer_deepgemm_wrapper",
            dependencies=["flashinfer", "deep_gemm"],
            description=(
                "Baseline solution using DeepGEMM logits and FlashInfer "
                "top_k_page_table_transform."
            ),
        )
    if not has_inputs(
        definition,
        {"q_nope", "q_pe", "ckv_cache", "kpe_cache", "sparse_indices", "sm_scale"},
    ):
        return None
    outputs = definition.get("outputs", {})
    source = _build_dsa_solution_source(
        include_lse=isinstance(outputs, dict) and "lse" in outputs
    )
    return solution_payload(
        definition,
        source,
        description=(
            "Baseline solution using FlashInfer TRT-LLM MLA decode for DSA sparse "
            "attention."
        ),
    )
