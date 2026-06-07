"""Shared parameter utilities for adapters.

This module stays kernel-agnostic. Kernel-specific config resolution belongs in
the adapter that owns that kernel family.
"""

from __future__ import annotations

from typing import Any


def nested_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return nested text_config when present, otherwise the original config."""
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def config_value(config: dict[str, Any], *names: str) -> Any:
    """Read the first matching config key from top-level or text_config."""
    nested = nested_config(config)
    for name in names:
        if name in config:
            return config[name]
        if name in nested:
            return nested[name]
    return None


def first_int(*values: Any) -> int | None:
    """Return the first integer-like value from a candidate list."""
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def is_unknown(value: Any) -> bool:
    """Return whether a value is missing or still an unresolved placeholder."""
    return value is None or value == "?"


def analysis_allows_family(analysis: dict[str, Any], flag: str) -> bool:
    """Return whether static source analysis allows a kernel family candidate.

    Static analysis is only used as an exclusion filter when SGLang source was
    actually scanned. If the source is unavailable, fail open so static
    discovery can still produce review-only candidates from HF config.
    """
    if analysis.get("available") is not True:
        return True
    return bool(analysis.get(flag, False))


def first_present(*values: Any) -> Any:
    """Return the first value that is not None/"?"."""
    for value in values:
        if not is_unknown(value):
            return value
    return None


def split_parallel(value: Any, tp: int | None) -> Any:
    """Return local per-TP value when divisible; otherwise keep original."""
    if not isinstance(value, int) or not isinstance(tp, int) or tp <= 1:
        return value
    if value % tp == 0:
        return value // tp
    return value


def infer_head_dim(config: dict[str, Any]) -> int | None:
    """Infer attention head_dim from explicit config or hidden_size/heads."""
    head_dim = config_value(config, "head_dim", "attention_head_dim")
    if isinstance(head_dim, int):
        return head_dim
    hidden_size = config_value(config, "hidden_size")
    num_heads = config_value(config, "num_attention_heads", "n_head")
    if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads:
        return hidden_size // num_heads
    return None


def infer_vocab_size(config: dict[str, Any]) -> int | None:
    """Return vocab_size only when the config exposes it directly."""
    vocab_size = config_value(config, "vocab_size")
    if isinstance(vocab_size, int) and not isinstance(vocab_size, bool):
        return vocab_size
    return None


def extract_hf_dims(config: dict[str, Any], tp: int | None = None) -> dict[str, Any]:
    """Extract commonly used model dimensions from a HF config dict."""
    hidden_size = config_value(config, "hidden_size")
    vocab_size = infer_vocab_size(config)
    num_heads = config_value(config, "num_attention_heads", "n_head")
    num_kv_heads = config_value(config, "num_key_value_heads", "num_kv_heads")
    if num_kv_heads is None:
        num_kv_heads = num_heads
    head_dim = infer_head_dim(config)

    linear_q_heads = config_value(
        config,
        "linear_num_key_heads",
        "gdn_num_q_heads",
        "num_gdn_heads",
    )
    linear_v_heads = config_value(
        config,
        "linear_num_value_heads",
        "gdn_num_kv_heads",
    )
    linear_head_dim = config_value(config, "linear_key_head_dim", "gdn_head_dim") or head_dim
    moe_intermediate = config_value(
        config,
        "moe_intermediate_size",
        "intermediate_size",
        "ffn_hidden_size",
    )
    num_experts = config_value(config, "num_experts", "n_routed_experts")

    return {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "tp_num_attention_heads": split_parallel(num_heads, tp),
        "tp_num_key_value_heads": split_parallel(num_kv_heads, tp),
        "gdn_num_q_heads": split_parallel(linear_q_heads, tp),
        "gdn_num_v_heads": split_parallel(linear_v_heads, tp),
        "gdn_head_dim": linear_head_dim,
        "intermediate_size": moe_intermediate,
        "tp_intermediate_size": split_parallel(moe_intermediate, tp),
        "num_experts": num_experts,
        # Expert count is not tensor-parallel head count. Keep config value as
        # global unless runtime explicitly reports num_local_experts.
        "num_local_experts": config_value(config, "num_local_experts") or num_experts,
        "num_experts_per_tok": config_value(
            config,
            "num_experts_per_tok",
            "num_experts_per_token",
        ),
        "n_group": config_value(config, "n_group", "num_expert_groups"),
        "topk_group": config_value(config, "topk_group", "top_k_group"),
        "kv_lora_rank": config_value(config, "kv_lora_rank"),
        "qk_rope_head_dim": config_value(config, "qk_rope_head_dim"),
        "dsa_topk": config_value(config, "dsa_topk", "sparse_topk", "topk"),
        "dsa_page_size": config_value(config, "dsa_page_size", "page_size"),
    }
