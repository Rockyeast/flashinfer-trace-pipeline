"""Kernel parameter extractors used by scripts/adapters/*.py.

中文导览：
    adapters 先把 trace_id 分类成 op_type / variant。
    这个文件负责下一步：从 probe 记录的 signature 里抽具体参数。

    例如：
      - RMSNorm 需要 hidden_size
      - Sampling 需要 vocab_size
      - GEMM 需要 N/K
      - Attention 需要 num_heads/head_dim/page_size
      - MoE 需要 top_k/experts/hidden/intermediate 等

    signature 大概长这样：
        {
          "trace_id": "...",
          "signature": {
            "self": {"attrs": {...}},
            "args": [{"type": "tensor", "shape": [...], "dtype": "..."}],
            "kwargs": {...}
          },
          "param_names": [...]
        }

    extractor 的输出不会直接写文件，而是被本目录下的 adapter 使用，
    最终生成 kernel_inventory.json 里的 kernel entry。
"""

from __future__ import annotations

import json


def _shape(value) -> list | None:
    if isinstance(value, dict):
        shape = value.get("shape")
        if isinstance(shape, list):
            return shape
    return None


def _first_int(*values) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _bind_signature_args(sig: dict) -> dict:
    """Bind positional signature args to param_names when probe recorded them."""
    body = sig.get("signature", sig)
    if not isinstance(body, dict):
        return {}
    args = body.get("args", [])
    kwargs = body.get("kwargs", {})
    names = sig.get("param_names") or body.get("param_names") or []
    bound: dict = {}
    if isinstance(names, list) and isinstance(args, list):
        for name, arg in zip(names, args):
            if isinstance(name, str):
                bound[name] = arg
    if isinstance(kwargs, dict):
        bound.update(kwargs)
    return bound


def _cache_shape_from_value(value) -> list | None:
    """Return a KV-cache tensor shape from serialized probe value metadata."""
    shape = _shape(value)
    if shape:
        return shape
    if isinstance(value, dict) and value.get("type") in {"tuple", "list"}:
        items = value.get("items")
        if isinstance(items, list):
            for item in items:
                shape = _shape(item)
                if shape:
                    return shape
    if isinstance(value, (tuple, list)):
        for item in value:
            shape = _cache_shape_from_value(item)
            if shape:
                return shape
    return None


def _page_size_from_cache_shape(shape: list | None) -> int | None:
    """Infer paged KV-cache page_size from serialized cache tensor shape."""
    if not shape:
        return None
    if len(shape) == 4:
        return _first_int(shape[1])
    if len(shape) == 3:
        # SGLang/FlashInfer often squeezes page_size=1 caches to
        # [num_pages, num_kv_heads, head_dim].
        return 1
    return None


def _heads_from_q_shape(shape: list | None) -> tuple[int | None, int | None]:
    if not shape or len(shape) < 2:
        return None, None
    return _first_int(shape[-2]), _first_int(shape[-1])


def _kv_heads_from_cache_shape(shape: list | None) -> int | None:
    if not shape:
        return None
    if len(shape) == 4:
        return _first_int(shape[2])
    if len(shape) == 3:
        return _first_int(shape[1])
    return None


def _extract_attention_params_from_signature(sig: dict) -> dict:
    """Extract attention params from one signature without guessing from config."""
    body = sig.get("signature", sig)
    if not isinstance(body, dict):
        return {}

    params = {}
    self_info = body.get("self")
    if self_info and isinstance(self_info, dict):
        attrs = self_info.get("attrs", {})
        for key in (
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "page_size",
            "_page_size",
            "tp_q_head_num",
            "tp_k_head_num",
            "head_size",
            "v_head_dim",
        ):
            if key in attrs:
                normalized = "page_size" if key == "_page_size" else key
                params[normalized] = attrs[key]

    bound = _bind_signature_args(sig)
    args = body.get("args", [])
    if isinstance(args, list):
        if "q" not in bound and args:
            bound["q"] = args[0]
        if "paged_kv_cache" not in bound and len(args) > 1:
            bound["paged_kv_cache"] = args[1]

    q_shape = _shape(bound.get("q") or bound.get("query"))
    q_heads, head_dim = _heads_from_q_shape(q_shape)
    q_heads = _first_int(
        bound.get("num_qo_heads"),
        bound.get("num_q_heads"),
        bound.get("num_heads"),
        q_heads,
    )
    head_dim = _first_int(
        bound.get("head_dim_qk"),
        bound.get("head_dim_vo"),
        bound.get("head_dim"),
        bound.get("head_size"),
        head_dim,
    )
    if q_heads is not None:
        params.setdefault("num_q_heads", q_heads)
    if head_dim is not None:
        params.setdefault("head_dim", head_dim)

    cache_shape = None
    for key in (
        "paged_kv_cache",
        "kv_cache",
        "k_cache",
        "v_cache",
        "ckv_cache",
        "kpe_cache",
    ):
        cache_shape = _cache_shape_from_value(bound.get(key))
        if cache_shape:
            break

    kv_heads = _kv_heads_from_cache_shape(cache_shape)
    kv_heads = _first_int(bound.get("num_kv_heads"), bound.get("num_kv_head"), kv_heads)
    if kv_heads is not None:
        params.setdefault("num_kv_heads", kv_heads)

    page_size = _first_int(
        sig.get("probe_page_size"),
        bound.get("page_size"),
        _page_size_from_cache_shape(cache_shape),
    )
    if page_size is not None:
        params.setdefault("page_size", page_size)

    return params


def extract_rmsnorm_hidden_sizes(signatures: list[dict]) -> set[int]:
    """Extract hidden_size from RMSNorm signatures."""
    # RMSNorm 的核心参数是 hidden_size。
    # 优先从 self.attrs.hidden_size 读取；如果没有，就从第一个 tensor shape 推断。
    #
    # 常见输入：
    #   [batch, hidden] -> hidden_size = shape[1]
    #   [hidden]        -> hidden_size = shape[0]
    #
    # 返回 set[int] 是因为同一种 trace 可能出现多个 shape，我们要去重。
    sizes = set()
    for sig in signatures:
        s = sig.get("signature", {})
        self_info = s.get("self", {})
        attrs = self_info.get("attrs", {}) if isinstance(self_info, dict) else {}
        if "hidden_size" in attrs:
            sizes.add(attrs["hidden_size"])
            continue

        args = s.get("args", [])
        for a in args:
            if isinstance(a, dict) and a.get("type") == "tensor":
                shape = a.get("shape", [])
                if len(shape) == 2:
                    sizes.add(shape[1])
                elif len(shape) == 1:
                    sizes.add(shape[0])
                break
    return sizes


def extract_moe_params(signatures: list[dict]) -> list[dict]:
    """Extract MoE parameters from FusedMoE method signatures."""
    # MoE 参数通常不直接从 tensor shape 推出来，而是藏在 layer 对象 attrs 里。
    #
    # 两种常见位置：
    #   1. kwargs["layer"]["attrs"]
    #   2. args[0]["attrs"]
    #
    # attrs 里可能包含：
    #   top_k, num_experts, num_local_experts, hidden_size,
    #   intermediate_size, n_group, topk_group, block_size 等。
    #
    # params_set 用 JSON 字符串做 key，是为了把重复 layer 参数去重。
    # layer_id 会导致每层都不同，但 definition 通常不应该按 layer_id 区分，
    # 所以去重时忽略 layer_id。
    params_set: dict[str, dict] = {}
    for sig in signatures:
        s = sig.get("signature", {})

        attrs = {}
        kwargs = s.get("kwargs", {})
        layer_info = kwargs.get("layer")
        if isinstance(layer_info, dict):
            attrs = layer_info.get("attrs", {})

        if not attrs:
            args = s.get("args", [])
            if args and isinstance(args[0], dict):
                attrs = args[0].get("attrs", {})

        if attrs:
            key = json.dumps(
                {k: attrs[k] for k in sorted(attrs) if k != "layer_id"},
                sort_keys=True,
            )
            if key not in params_set:
                params_set[key] = {
                    k: v for k, v in attrs.items() if k != "layer_id"
                }
    return list(params_set.values())


def determine_moe_variant(sub_variant: str, params: dict) -> str:
    """Determine MoE variant (bf16 or fp8) from the sub_variant and params."""
    # rules.py 可能只能粗分出 generic_moe。
    # 这里结合参数再判断最终 variant：
    #   - 明确 fp8_moe -> fp8
    #   - 明确 unquantized_moe -> bf16
    #   - generic_moe 如果带 block_size/weight_block_size/input_scale 这类量化参数，
    #     就认为是 fp8，否则默认 bf16。
    if sub_variant == "fp8_moe":
        return "fp8"
    if sub_variant == "unquantized_moe":
        return "bf16"
    if sub_variant == "generic_moe":
        if any(k in params for k in ("block_size", "weight_block_size", "input_scale")):
            return "fp8"
        return "bf16"
    return "bf16"


def extract_gemm_dims(signatures: list[dict]) -> list[tuple[int, int]]:
    """Extract (N, K) pairs from linear method signatures."""
    # GEMM/Linear 的 definition 名一般需要：
    #   N = output dimension
    #   K = input dimension
    #
    # 对 SGLang LinearMethod wrapper，第一参数通常是 layer 对象，
    # layer.attrs 里会有 input/output size。
    #
    # 优先使用 per_partition 字段，因为 tensor parallel 下实际本地 GEMM
    # 看到的是分片后的 N/K。
    dims: set[tuple[int, int]] = set()
    for sig in signatures:
        s = sig.get("signature", {})
        args = s.get("args", [])
        if not args:
            continue
        layer = args[0]
        if not isinstance(layer, dict) or layer.get("type") != "object":
            continue
        attrs = layer.get("attrs", {})
        if not attrs:
            continue

        n = attrs.get("output_size_per_partition")
        if n is None:
            n = attrs.get("output_size")
        if n is None:
            continue

        k = attrs.get("input_size_per_partition")
        if k is None:
            k = attrs.get("input_size")
        if k is None:
            continue

        if isinstance(n, int) and isinstance(k, int) and n >= 16 and k >= 16:
            dims.add((n, k))

    return sorted(dims)


def extract_cascade_merge_state_params(signatures: list[dict]) -> set[tuple[int, int]]:
    """Extract (num_heads, head_dim) from flashinfer.cascade.merge_state calls."""
    # flashinfer.cascade.merge_state 的前两个 tensor 通常形如：
    #   v: [batch, num_heads, head_dim]
    #   s: [batch, num_heads]
    #
    # 所以：
    #   num_heads = v_shape[1]
    #   head_dim  = v_shape[2]
    #
    # 同时检查 v/s 的 batch 和 head 维一致，避免误读其它 tensor。
    params: set[tuple[int, int]] = set()
    for sig in signatures:
        s = sig.get("signature", sig)
        args = s.get("args", [])
        if len(args) < 4:
            continue
        v_arg = args[0]
        s_arg = args[1]
        v_shape = v_arg.get("shape") if isinstance(v_arg, dict) else None
        s_shape = s_arg.get("shape") if isinstance(s_arg, dict) else None
        if (
            isinstance(v_shape, list)
            and len(v_shape) == 3
            and isinstance(s_shape, list)
            and len(s_shape) == 2
            and v_shape[0] == s_shape[0]
            and v_shape[1] == s_shape[1]
        ):
            params.add((int(v_shape[1]), int(v_shape[2])))
    return params


def extract_attention_params(signatures: list[dict]) -> dict:
    """Extract attention parameters from decode/prefill wrapper signatures."""
    # Attention wrapper 的关键参数通常存在 self.attrs。
    # 这里收集所有可能有用的字段，后面由 gqa/mla/gdn builder 决定怎么用。
    #
    # 字段含义大概是：
    #   num_heads / tp_q_head_num: query heads
    #   num_kv_heads / tp_k_head_num: kv heads
    #   head_dim / head_size: head dimension
    #   v_head_dim: value head dimension
    #   page_size: paged KV cache page size
    params = {}
    for sig in signatures:
        for key, value in _extract_attention_params_from_signature(sig).items():
            params.setdefault(key, value)
    return params


def extract_attention_param_sets(signatures: list[dict]) -> list[dict]:
    """Extract unique attention parameter sets from observed signatures."""
    sets: list[dict] = []
    seen: set[str] = set()
    for sig in signatures:
        params = _extract_attention_params_from_signature(sig)
        if not params:
            continue
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        sets.append(params)
    return sets


def extract_observed_kwargs(signatures: list[dict]) -> list[str]:
    """Extract unique kwargs key names observed across all signatures."""
    # 这个不是生成 definition 的核心参数，而是诊断信息。
    # 它记录这组 trace 里出现过哪些 kwargs 名字，方便人工 debug。
    keys: set[str] = set()
    for sig in signatures:
        s = sig.get("signature", {})
        kwargs = s.get("kwargs", {})
        if isinstance(kwargs, dict):
            keys.update(kwargs.keys())
    return sorted(keys)


def extract_observed_param_names(signatures: list[dict]) -> list[str]:
    """Extract the function parameter names from probe signatures."""
    # probe 有时能拿到函数签名里的参数名 param_names。
    # 多个 signature 样本可能记录长度不同，这里选最长的一份作为最完整版本。
    # 这个主要用于 inventory 诊断，不是核心分类依据。
    best: list[str] = []
    for sig in signatures:
        pn = sig.get("param_names")
        if isinstance(pn, list) and len(pn) > len(best):
            best = pn
    return best
