import json
from pathlib import Path

import pytest

from pipeline.runtime_defaults import forwarded_runtime_env
from pipeline.runtime_plan import (
    _infer_tp_from_hf_config,
    _normalize_tp_for_hf_config,
    _select_modal_gpu,
)


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_deepseek_v4_flash_model_does_not_match_project_as_pro(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"model_type": "deepseek_v4", "hidden_size": 4096})
    logs: list[str] = []

    tp = _infer_tp_from_hf_config(
        "sgl-project/DeepSeek-V4-Flash-FP8",
        config,
        project_root=tmp_path,
        log=logs.append,
    )

    assert tp == 8
    assert "TP=8" in logs[0]


def test_deepseek_v4_pro_leaf_selects_larger_tp(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"model_type": "deepseek_v4", "hidden_size": 4096})
    logs: list[str] = []

    tp = _infer_tp_from_hf_config(
        "deepseek-ai/DeepSeek-V4-Pro",
        config,
        project_root=tmp_path,
        log=logs.append,
    )

    assert tp == 16
    assert "TP=16" in logs[0]


def test_large_moe_profile_selects_tp_and_gpu(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        {
            "model_type": "custom_moe",
            "hidden_size": 4096,
            "n_routed_experts": 256,
        },
    )
    logs: list[str] = []

    tp = _infer_tp_from_hf_config(
        "vendor/NewMoE",
        config,
        project_root=tmp_path,
        log=logs.append,
    )

    assert tp == 8
    assert _select_modal_gpu("vendor/NewMoE", config, tmp_path) == "A100-80GB"


def test_runtime_env_forwards_sglang_prefix_without_probe_internals() -> None:
    env = {
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_WORKER_TRACE_DIR": "/tmp/internal",
        "PROBE_MEM_FRACTION_STATIC": "0.8",
        "UNRELATED": "1",
    }

    forwarded = forwarded_runtime_env(env)

    assert forwarded == {
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "PROBE_MEM_FRACTION_STATIC": "0.8",
    }


def test_runtime_env_supports_explicit_extra_names_and_prefixes() -> None:
    env = {
        "PROBE_FORWARD_ENV_NAMES": "CUSTOM_FLAG",
        "PROBE_FORWARD_ENV_PREFIXES": "NCCL_,CUDA_",
        "CUSTOM_FLAG": "enabled",
        "NCCL_DEBUG": "INFO",
        "SGLANG_FLAG": "not-forwarded-when-prefixes-overridden",
    }

    forwarded = forwarded_runtime_env(env)

    assert forwarded == {
        "CUSTOM_FLAG": "enabled",
        "NCCL_DEBUG": "INFO",
    }


def test_normalize_tp_falls_back_to_valid_attention_head_divisor(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        {
            "model_type": "unit",
            "num_attention_heads": 20,
        },
    )
    logs: list[str] = []

    tp = _normalize_tp_for_hf_config(
        model_name="unit/model",
        hf_config=config,
        tp=8,
        explicit=False,
        project_root=tmp_path,
        log=logs.append,
    )

    assert tp == 5
    assert "using TP=5" in logs[0]


def test_normalize_tp_rejects_explicit_invalid_attention_head_divisor(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        {
            "model_type": "unit",
            "num_attention_heads": 20,
        },
    )

    with pytest.raises(SystemExit, match="not divisible by TP"):
        _normalize_tp_for_hf_config(
            model_name="unit/model",
            hf_config=config,
            tp=8,
            explicit=True,
            project_root=tmp_path,
            log=lambda _msg: None,
        )
