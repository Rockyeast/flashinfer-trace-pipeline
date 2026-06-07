"""Lightweight static scanner for SGLang model files."""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

_KERNEL_FAMILY_TOKENS: dict[str, tuple[str, ...]] = {
    "rmsnorm": ("RMSNorm",),
    "gqa": ("RadixAttention", "QKVParallelLinear"),
    "gdn": ("RadixLinearAttention", "GatedDelta"),
    "mla": ("MLA", "Mla"),
    "dsa": ("SparseAttention", "BlockSparseAttention", "NativeSparse", "DSA"),
    "moe": ("FusedMoE", "SparseMoe", "Moe"),
    "gemm": (
        "QKVParallelLinear",
        "MergedColumnParallelLinear",
        "RowParallelLinear",
        "ColumnParallelLinear",
    ),
}


def _model_type_file_stem(config: dict) -> str | None:
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        return None
    return model_type.strip().lower().replace("-", "_")


def locate_model_file(sglang_root: Path, config: dict) -> Path | None:
    """Find the SGLang model implementation file from HF config.model_type."""
    stem = _model_type_file_stem(config)
    if stem is None:
        return None
    models_dir = sglang_root / "python" / "sglang" / "srt" / "models"
    if not models_dir.exists():
        return None
    path = models_dir / f"{stem}.py"
    return path if path.exists() else None


def _source_without_comments_and_strings(source: str) -> str:
    """Return source text with comments and string literals removed."""
    tokens: list[tokenize.TokenInfo] = []
    try:
        stream = io.StringIO(source).readline
        for token in tokenize.generate_tokens(stream):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                token = tokenize.TokenInfo(token.type, "", token.start, token.end, token.line)
            tokens.append(token)
        return tokenize.untokenize(tokens)
    except tokenize.TokenError:
        return source


def _matched_tokens(source: str, model_file: Path) -> dict[str, list[str]]:
    """Return keyword tokens matched by kernel family."""
    matched: dict[str, list[str]] = {}
    for family, tokens in _KERNEL_FAMILY_TOKENS.items():
        hits = [token for token in tokens if token in source]
        if family == "mla" and "Mla" in model_file.name and "model_file:Mla" not in hits:
            hits.append("model_file:Mla")
        if hits:
            matched[family] = hits
    return matched


def analyze_sglang_model(sglang_root: Path | None, config: dict) -> dict:
    """Return coarse kernel-family evidence from the SGLang model source."""
    model_type = _model_type_file_stem(config)
    if sglang_root is None:
        return {"model_file": None, "model_type": model_type, "available": False}

    model_file = locate_model_file(sglang_root, config)
    if model_file is None:
        return {"model_file": None, "model_type": model_type, "available": False}

    source = model_file.read_text(encoding="utf-8", errors="ignore")
    matched_tokens = _matched_tokens(_source_without_comments_and_strings(source), model_file)
    return {
        "model_file": str(model_file),
        "model_type": model_type,
        "available": True,
        "matched_tokens": matched_tokens,
        "has_rmsnorm": bool(matched_tokens.get("rmsnorm")),
        "has_gqa": bool(matched_tokens.get("gqa")),
        "has_gdn": bool(matched_tokens.get("gdn")),
        "has_mla": bool(matched_tokens.get("mla")),
        "has_dsa": bool(matched_tokens.get("dsa")),
        "has_moe": bool(matched_tokens.get("moe")),
        "has_gemm": bool(matched_tokens.get("gemm")),
        "has_sampling": True,
    }
