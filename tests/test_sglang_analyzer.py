from pathlib import Path

from static.sglang_analyzer import analyze_sglang_model, locate_model_file


def _write_model(root: Path, stem: str, source: str) -> Path:
    path = root / "python" / "sglang" / "srt" / "models" / f"{stem}.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_locate_model_file_uses_hf_model_type(tmp_path: Path) -> None:
    model_file = _write_model(tmp_path, "qwen3", "class Model: pass\n")

    assert locate_model_file(tmp_path, {"model_type": "qwen3"}) == model_file


def test_analyze_sglang_model_does_not_fallback_to_repo_name(tmp_path: Path) -> None:
    _write_model(tmp_path, "qwen3", "RMSNorm\nRadixAttention\n")

    analysis = analyze_sglang_model(
        tmp_path,
        {"model_type": "unknown_arch"},
    )

    assert analysis == {
        "model_file": None,
        "model_type": "unknown_arch",
        "available": False,
    }


def test_analyze_sglang_model_reports_source_flags(tmp_path: Path) -> None:
    model_file = _write_model(
        tmp_path,
        "deepseek_v4",
        "RMSNorm\nRadixAttention\nFusedMoE\nColumnParallelLinear\n",
    )

    analysis = analyze_sglang_model(tmp_path, {"model_type": "deepseek_v4"})

    assert analysis["model_file"] == str(model_file)
    assert analysis["model_type"] == "deepseek_v4"
    assert analysis["available"] is True
    assert analysis["has_rmsnorm"] is True
    assert analysis["has_gqa"] is True
    assert analysis["has_moe"] is True
    assert analysis["has_gemm"] is True
    assert analysis["matched_tokens"] == {
        "rmsnorm": ["RMSNorm"],
        "gqa": ["RadixAttention"],
        "moe": ["FusedMoE"],
        "gemm": ["ColumnParallelLinear"],
    }


def test_analyze_sglang_model_ignores_comments_and_docstrings(tmp_path: Path) -> None:
    _write_model(
        tmp_path,
        "llama",
        '''
"""RMSNorm and FusedMoE are mentioned in a docstring only."""

# RadixAttention is mentioned in a comment only.
class Model:
    pass
''',
    )

    analysis = analyze_sglang_model(tmp_path, {"model_type": "llama"})

    assert analysis["available"] is True
    assert analysis["matched_tokens"] == {}
    assert analysis["has_rmsnorm"] is False
    assert analysis["has_gqa"] is False
    assert analysis["has_moe"] is False
