import json
import os
import subprocess
import sys
import base64
import tarfile
import types
from pathlib import Path

import pytest

from flashinfer_trace.core.capture import CaptureSession
from flashinfer_trace.core.capture import infer_dispatch_value
from flashinfer_trace.core.capture import HF_CONFIG_OVERRIDE_ENV, install_hf_config_override, install_hf_config_override_from_env
from flashinfer_trace.core.definition_audit import audit_and_repair_definitions, prepare_definition_for_output
from flashinfer_trace.core.events import (
    build_event_report,
    build_workload_manifest,
    build_sanitized_workload_entry,
    load_jsonl,
)
from flashinfer_trace.runners.modal_runner import (
    _filter_supported_engine_kwargs,
    _materialize_reviewed_artifacts,
    _probe_passes,
    _prompt_scenarios,
    _supplemental_runs,
    materialize_modal_result,
    run_remote_probe_entrypoint,
)
from flashinfer_trace.cli import _reviewed_definition_artifacts
from flashinfer_trace.core.planning import (
    build_collect_plan_from_probe_plan,
    build_modal_probe_plan,
    build_probe_plan,
    load_definitions,
    load_approved_targets,
)
from flashinfer_trace.validation import (
    export_run_dataset,
    render_run_review_markdown,
    run_official_validate,
    update_run_report,
    validate_run,
)
from tools.proposal_tools import (
    _sglang_config_compat_engine_kwargs,
    check_proposal,
    diagnose_run,
    merge_proposals,
    repair_loop,
    run_agent_loop,
    spawn_agents,
    slug_model_name,
)
from flashinfer_trace.core.schemas import ApprovedTarget, CaptureSpec, CollectPlan, CollectTarget, DefinitionRef, DispatchSpec, ProbePlan, ProbeTarget, WarmupHook


def _capture_json(
    *,
    full_args: list[int] | None = None,
    full_kwargs: list[str] | None = None,
    structural_attr_tokens: list[str] | None = None,
) -> dict:
    tokens = structural_attr_tokens or [
        "indptr",
        "indices",
        "last_page",
        "page_len",
        "seq_len",
        "offset",
        "mask",
        "block_table",
    ]
    return {
        "full_args": sorted(set(full_args or [])),
        "full_kwargs": sorted(set(full_kwargs or [])),
        "structural_attr_tokens": sorted(set(tokens)),
    }


def _capture_spec(
    *,
    full_args: list[int] | None = None,
    full_kwargs: list[str] | None = None,
    structural_attr_tokens: list[str] | None = None,
) -> CaptureSpec:
    return CaptureSpec(**_capture_json(
        full_args=full_args,
        full_kwargs=full_kwargs,
        structural_attr_tokens=structural_attr_tokens,
    ))


def _gqa_paged_hints(definition_name: str = "demo_decode") -> dict:
    return {
        "schema_version": 1,
        "definition_name": definition_name,
        "op_type": "gqa_paged",
        "inputs": {
            "q": [{"source": "arg", "arg_index": 1}],
            "k_cache": [{"source": "arg_tuple", "arg_index": 2, "tuple_index": 0}],
            "v_cache": [{"source": "arg_tuple", "arg_index": 2, "tuple_index": 1}],
            "kv_indptr": [{"source": "attr", "pattern": "paged_kv_indptr"}],
            "kv_indices": [{"source": "attr", "pattern": "paged_kv_indices"}],
            "qo_indptr": [{"source": "attr", "pattern": "qo_indptr"}],
        },
        "shape_overrides": {
            "k_cache": {"squeezed_axes": ["page_size"]},
            "v_cache": {"squeezed_axes": ["page_size"]},
        },
        "axes": {
            "num_kv_indices": {"source": "tensor_last", "input": "kv_indptr"},
            "total_q": {"source": "tensor_last", "input": "qo_indptr"},
            "num_pages": {
                "source": "tensor_max_plus_one",
                "input": "kv_indices",
                "limit_axis": "num_kv_indices",
            },
        },
        "tensor_slices": {
            "kv_indices": {"limit_axis": "num_kv_indices"},
        },
    }


def _gqa_ragged_hints(definition_name: str = "demo_ragged") -> dict:
    return {
        "schema_version": 1,
        "definition_name": definition_name,
        "op_type": "gqa_ragged",
        "inputs": {
            "q": [{"source": "arg", "arg_index": 1}],
            "k": [{"source": "arg", "arg_index": 2}],
            "v": [{"source": "arg", "arg_index": 3}],
        },
    }


def _page_size_dispatch_json() -> dict:
    return {
        "field": "page_size",
        "rules": [
            {"kind": "arg_attr", "arg_index": 0, "attrs": ["page_size", "_page_size"]},
            {"kind": "arg_shape", "arg_index": 2, "tuple_index": 0, "min_rank": 4, "shape_index": 1},
            {"kind": "arg_shape", "arg_index": 2, "tuple_index": 0, "rank": 3, "value": 1},
            {"kind": "arg_shape", "arg_index": 2, "min_rank": 5, "shape_index": 2},
            {"kind": "arg_shape", "arg_index": 2, "rank": 4, "equals_index": 1, "equals": 2, "value": 1},
        ],
    }


def _page_size_dispatch_spec() -> DispatchSpec:
    from flashinfer_trace.core.schemas import dispatch_spec_from_jsonable

    dispatch = dispatch_spec_from_jsonable(_page_size_dispatch_json(), context="test target")
    assert dispatch is not None
    return dispatch


def _write_run_inputs(run_dir: Path, *, config: dict, approved: list[dict]) -> None:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    normalized = []
    for item in approved:
        copied = dict(item)
        if copied.get("role", "target") == "target" and "capture" not in copied:
            copied["capture"] = _capture_json()
        if copied.get("role", "target") == "target" and copied.get("page_size") is not None and "dispatch" not in copied:
            copied["dispatch"] = _page_size_dispatch_json()
        normalized.append(copied)
    (config_dir / "approved_targets.json").write_text(json.dumps(normalized), encoding="utf-8")


def _write_sharegpt_fixture(path: Path, prompts: list[str] | None = None) -> None:
    payload = prompts or [
        "short prompt",
        "medium length prompt for collect testing",
        "a substantially longer prompt used to exercise the collect length bucket planner",
        "another realistic prompt about GPU inference and batch scheduling",
        "write a concise explanation of paged attention",
        "summarize sampling configuration tradeoffs",
        "draft a review request for a pull request",
        "list the main tensor shapes in an attention layer",
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_modal_plan_from_run(run_dir: Path, **kwargs: object) -> dict:
    return build_modal_probe_plan(
        probe_plan=build_probe_plan(load_approved_targets(run_dir / "config" / "approved_targets.json")),
        output_dir=run_dir / ".modal_tmp",
        **kwargs,
    )


def _collect_strategy() -> dict:
    return {
        "batch_sizes": [1],
        "max_new_tokens": 16,
        "supplemental_runs": [_supplemental_run()],
        "max_captures_per_target": 8,
    }


def _supplemental_run(
    *,
    name: str = "sampling_supplemental",
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    allowed_op_types: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "sampling_params": {"temperature": temperature, "top_k": top_k, "top_p": top_p},
        "allowed_op_types": ["sampling"] if allowed_op_types is None else allowed_op_types,
    }


def _write_collect_report(run_dir: Path, *, plan: dict, manifest: dict | None = None) -> None:
    update_run_report(run_dir, collect={"plan": plan, "manifest": manifest or {"summary": {}}})


def test_modal_probe_does_not_generate_supplemental_runs_from_target_type() -> None:
    plan = {
        "sampling": {"max_new_tokens": 3},
        "probe_plan": {
            "targets": [
                {
                    "name": "sampling",
                    "op_type": "sampling",
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="supplemental_runs"):
        _supplemental_runs(plan)


def test_modal_probe_plan_jsonable_rejects_invalid_hook_specs() -> None:
    with pytest.raises(ValueError, match="missing name/target/module/attr"):
        ProbePlan.from_jsonable({
            "targets": [
                {
                    "name": "missing_attr",
                    "target": "flashinfer.norm.gemma_rmsnorm",
                    "module": "flashinfer.norm",
                    "probe_mode": "default",
                }
            ]
        })

    with pytest.raises(ValueError, match="invalid probe_mode"):
        ProbePlan.from_jsonable({
            "targets": [
                {
                    "name": "bad_mode",
                    "target": "flashinfer.norm.gemma_rmsnorm",
                    "module": "flashinfer.norm",
                    "attr": "gemma_rmsnorm",
                    "probe_mode": "guess",
                }
            ]
        })


def test_modal_probe_collect_run_expands_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", [
        "Explain why GPU memory bandwidth matters for transformer inference.",
        "Summarize the tradeoff between latency and throughput in batch serving.",
        "Write a short Python function that checks whether a number is prime.",
        "Give three practical debugging steps for a CUDA kernel launch failure.",
        "Describe how paged KV cache helps long-context language model serving.",
        "Compare greedy decoding with top-p sampling in two concise paragraphs.",
        "Draft a polite email asking a teammate to review a pull request.",
        "List the main components of an attention layer and their tensor shapes.",
    ])
    plan = build_modal_probe_plan(
        probe_plan=ProbePlan(targets=[], skipped=[]),
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir=tmp_path,
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        batch_sizes=[1, 2, 4, 8, 16, 32, 64],
        max_new_tokens=96,
        supplemental_runs=[_supplemental_run()],
        max_captures_per_target=128,
    )

    assert plan["sampling"]["max_new_tokens"] == 96
    assert plan["capture_limits"]["max_captures_per_target"] == 128
    assert "batch_sizes" not in plan
    assert "capture_output" not in plan
    assert [len(scenario["prompts"]) for scenario in _prompt_scenarios(plan)] == [1, 2, 4, 8, 16, 32, 64]
    assert [scenario["max_new_tokens"] for scenario in _prompt_scenarios(plan)] == [96, 64, 32, 32, 16, 8, 8]
    assert all("bucket" not in scenario for scenario in _prompt_scenarios(plan))
    assert plan["sglang"]["disable_cuda_graph"] is True


def test_modal_probe_plan_allows_cuda_graph_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json")
    plan = build_modal_probe_plan(
        probe_plan=ProbePlan(targets=[], skipped=[]),
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir=tmp_path,
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        disable_cuda_graph=False,
        **_collect_strategy(),
    )

    assert plan["sglang"]["disable_cuda_graph"] is False


def test_modal_probe_plan_carries_reviewed_sglang_runtime_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json")
    plan = build_modal_probe_plan(
        probe_plan=ProbePlan(targets=[], skipped=[]),
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir=tmp_path,
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        enable_piecewise_cuda_graph=True,
        force_flashinfer_backends=True,
        mem_fraction_static=0.7,
        cuda_graph_max_bs=64,
        **_collect_strategy(),
    )

    assert plan["sglang"] == {
        "disable_cuda_graph": True,
        "enable_piecewise_cuda_graph": True,
        "force_flashinfer_backends": True,
        "mem_fraction_static": 0.7,
        "cuda_graph_max_bs": 64,
        "engine_kwargs": {},
    }


def test_engine_kwargs_filter_rejects_unsupported_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sglang = types.ModuleType("sglang")
    fake_srt = types.ModuleType("sglang.srt")
    fake_server_args = types.ModuleType("sglang.srt.server_args")

    class ServerArgs:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

    fake_server_args.ServerArgs = ServerArgs
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", fake_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", fake_server_args)

    with pytest.raises(RuntimeError, match="does not support kwargs"):
        _filter_supported_engine_kwargs({"model_path": "m", "page_size": 64})


def test_engine_kwargs_filter_drops_unsupported_optional_runner_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sglang = types.ModuleType("sglang")
    fake_srt = types.ModuleType("sglang.srt")
    fake_server_args = types.ModuleType("sglang.srt.server_args")

    class ServerArgs:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

    fake_server_args.ServerArgs = ServerArgs
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", fake_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", fake_server_args)

    filtered = _filter_supported_engine_kwargs(
        {"model_path": "m", "disable_piecewise_cuda_graph": True},
        optional_kwargs={"disable_piecewise_cuda_graph"},
    )

    assert filtered == {"model_path": "m"}


def test_hf_config_override_replaces_original_config_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    import flashinfer_trace.core.capture as capture_module

    class FakePretrainedConfig:
        seen: dict | None = None

        @classmethod
        def from_dict(cls, config_dict: dict, **kwargs: object) -> dict:
            rope_scaling = config_dict.get("rope_scaling")
            rope_theta = config_dict.get("rope_theta")
            if isinstance(rope_scaling, dict) and rope_theta is not None:
                config_dict = dict(config_dict)
                config_dict["rope_scaling"] = {**rope_scaling, "rope_theta": rope_theta}
            cls.seen = config_dict
            return config_dict

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.PretrainedConfig = FakePretrainedConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(capture_module, "_HF_CONFIG_PATCHED", False)
    monkeypatch.setattr(capture_module, "_HF_CONFIG_OVERRIDE", None)

    install_hf_config_override({
        "model_type": "phi3",
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0],
            "long_factor": [1.0],
        },
    })
    FakePretrainedConfig.from_dict({
        "model_type": "phi3",
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0],
            "long_factor": [1.0],
        },
    })

    assert FakePretrainedConfig.seen is not None
    assert "rope_theta" not in FakePretrainedConfig.seen
    assert "rope_theta" not in FakePretrainedConfig.seen["rope_scaling"]


def test_hf_config_override_installs_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import flashinfer_trace.core.capture as capture_module

    class FakePretrainedConfig:
        seen: dict | None = None

        @classmethod
        def from_dict(cls, config_dict: dict, **kwargs: object) -> dict:
            cls.seen = config_dict
            return config_dict

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.PretrainedConfig = FakePretrainedConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(capture_module, "_HF_CONFIG_PATCHED", False)
    monkeypatch.setattr(capture_module, "_HF_CONFIG_OVERRIDE", None)
    monkeypatch.setenv(HF_CONFIG_OVERRIDE_ENV, json.dumps({"model_type": "phi3"}))

    install_hf_config_override_from_env()
    FakePretrainedConfig.from_dict({"model_type": "other"})

    assert FakePretrainedConfig.seen == {"model_type": "phi3"}


def test_modal_probe_plan_uses_fixed_sharegpt_collect_strategy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", ["alpha", "beta"])
    plan = build_modal_probe_plan(
        probe_plan=ProbePlan(targets=[], skipped=[]),
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir=tmp_path,
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        batch_sizes=[1, 3],
        max_new_tokens=11,
        supplemental_runs=[_supplemental_run(top_k=42, top_p=0.95)],
        max_captures_per_target=17,
    )

    assert "prompts" not in plan
    assert "prompt" not in plan
    assert "batch_sizes" not in plan
    assert [len(scenario["prompts"]) for scenario in _prompt_scenarios(plan)] == [1, 3]
    assert plan["sampling"] == {
        "max_new_tokens": 11,
    }
    assert plan["supplemental_runs"][0]["sampling_params"] == {"temperature": 0.7, "top_k": 42, "top_p": 0.95}
    assert plan["capture_limits"]["max_captures_per_target"] == 17


def test_modal_probe_collect_run_rejects_missing_reviewed_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reviewed batch_sizes"):
        build_modal_probe_plan(
            probe_plan=ProbePlan(targets=[], skipped=[]),
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            output_dir=tmp_path,
            image="lmsysorg/sglang:v0.5.12.post1",
            gpu="L40S",
            tp_size=1,
            max_new_tokens=1,
            supplemental_runs=[_supplemental_run()],
            max_captures_per_target=128,
            )


def test_modal_probe_collect_run_rejects_invalid_reviewed_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json")
    base = {
        "probe_plan": ProbePlan(targets=[], skipped=[]),
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "output_dir": tmp_path,
        "image": "lmsysorg/sglang:v0.5.12.post1",
        "gpu": "L40S",
        "tp_size": 1,
        "batch_sizes": [1],
        "max_new_tokens": 1,
        "supplemental_runs": [_supplemental_run()],
        "max_captures_per_target": 128,
    }

    for override, pattern in [
        ({"max_new_tokens": 0}, "max_new_tokens must be a positive integer"),
        ({"batch_sizes": [1, "8"]}, r"batch_sizes\[1\] must be a positive integer"),
        (
            {"supplemental_runs": [{"name": "bad", "sampling_params": {"top_k": True}}]},
            r"supplemental_runs\[0\].sampling_params.top_k must be an integer",
        ),
        ({"max_captures_per_target": "128"}, "max_captures_per_target must be a positive integer"),
    ]:
        kwargs = dict(base)
        kwargs.update(override)
        with pytest.raises(ValueError, match=pattern):
            build_modal_probe_plan(**kwargs)


def test_modal_probe_collect_run_requires_string_sharegpt_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sharegpt_100.json").write_text(json.dumps(["valid", 123]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"sharegpt_100.json\[1\] must be a string prompt"):
        build_modal_probe_plan(
            probe_plan=ProbePlan(targets=[], skipped=[]),
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            output_dir=tmp_path,
            image="lmsysorg/sglang:v0.5.12.post1",
            gpu="L40S",
            tp_size=1,
            **_collect_strategy(),
        )


def test_modal_probe_plan_requires_reviewed_runtime_config(tmp_path: Path) -> None:
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json")
    with pytest.raises(ValueError, match="image is required"):
        build_modal_probe_plan(
            probe_plan=ProbePlan(targets=[], skipped=[]),
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            output_dir=tmp_path,
            image="",
            gpu="L40S",
            tp_size=1,
            **_collect_strategy(),
        )


def test_modal_probe_uses_reviewed_supplemental_runs() -> None:
    plan = {
        "supplemental_runs": [
            _supplemental_run(name="sampling_a", temperature=0.6, top_k=8, top_p=0.8),
            _supplemental_run(name="sampling_b", temperature=0.9, top_k=-1, top_p=1.0),
        ],
        "probe_plan": {"targets": [{"name": "sampling", "op_type": "sampling"}]},
    }

    assert _supplemental_runs(plan) == [
        {
            "name": "sampling_a",
            "sampling_params": {"temperature": 0.6, "top_k": 8, "top_p": 0.8},
            "allowed_op_types": ["sampling"],
            "use_scenario_tokens": False,
        },
        {
            "name": "sampling_b",
            "sampling_params": {"temperature": 0.9, "top_k": -1, "top_p": 1.0},
            "allowed_op_types": ["sampling"],
            "use_scenario_tokens": False,
        },
    ]


def test_modal_probe_requires_reviewed_supplemental_runs() -> None:
    plan = {
        "sampling": {"max_new_tokens": 2},
        "probe_plan": {"targets": [{"name": "decode", "op_type": "gqa_paged"}]},
    }

    with pytest.raises(ValueError, match="supplemental_runs"):
        _supplemental_runs(plan)


def test_modal_probe_splits_paged_passes_by_page_size() -> None:
    plan = {
        "probe_modes": ["default", "paged"],
        "probe_plan": {
            "targets": [
                {"name": "decode_ps1", "probe_mode": "paged", "page_size": 1},
                {"name": "decode_ps64", "probe_mode": "paged", "page_size": 64},
                {"name": "ragged", "probe_mode": "default"},
            ]
        },
    }

    assert _probe_passes(plan) == [(False, None), (True, 1), (True, 64)]


def test_capture_scope_can_limit_events_to_sampling_targets() -> None:
    sampling = ProbeTarget(
        name="sampling",
        target="flashinfer.sampling.top_k_top_p_sampling_from_probs",
        module="flashinfer.sampling",
        attr="top_k_top_p_sampling_from_probs",
        op_type="sampling",
        capture=_capture_spec(full_args=[0, 2]),
    )
    attention = ProbeTarget(
        name="attention",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        capture=_capture_spec(),
    )

    scope = {"name": "sampling_supplemental", "allowed_op_types": ["sampling"]}

    assert CaptureSession._target_allowed_by_scope(sampling, scope)
    assert not CaptureSession._target_allowed_by_scope(attention, scope)
    assert CaptureSession._target_allowed_by_scope(attention, {"name": "base"})


def test_capture_session_limits_captures_per_target_and_scope(tmp_path: Path) -> None:
    target = ProbeTarget(
        name="sampling",
        target="flashinfer.sampling.top_k_top_p_sampling_from_probs",
        module="flashinfer.sampling",
        attr="top_k_top_p_sampling_from_probs",
        op_type="sampling",
        capture=_capture_spec(full_args=[0, 2]),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[target], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=2,
    )
    session.captures_dir.mkdir(parents=True)

    for _ in range(3):
        session._write_event(target, (), {}, {"name": "base"})
    session._write_event(target, (), {}, {"name": "sampling_supplemental"})

    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    captures = sorted((tmp_path / "captures").glob("*"))

    assert len(events) == 3
    assert len(captures) == 3


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeWrapper:
    def __init__(self, page_size: int | None = None) -> None:
        if page_size is not None:
            self.page_size = page_size


def test_page_size_dispatch_drops_ambiguous_paged_calls(tmp_path: Path) -> None:
    ps1 = ProbeTarget(
        name="paged_ps1",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=1,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    ps64 = ProbeTarget(
        name="paged_ps64",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=64,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[ps1, ps64], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=4,
    )
    written: list[str] = []
    session._write_event = lambda target, args, kwargs, scope=None, traced_definition=None: written.append(target.name)  # type: ignore[method-assign]

    wrapped = session._make_multi_dispatch_wrapper([ps1, ps64], lambda *args, **kwargs: "ok")

    assert wrapped("self", "q", object()) == "ok"
    assert written == []


def test_page_size_dispatch_uses_wrapper_page_size_attr(tmp_path: Path) -> None:
    ps1 = ProbeTarget(
        name="paged_ps1",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=1,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    ps64 = ProbeTarget(
        name="paged_ps64",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=64,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[ps1, ps64], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=4,
    )
    written: list[str] = []
    session._write_event = lambda target, args, kwargs, scope=None, traced_definition=None: written.append(target.name)  # type: ignore[method-assign]

    wrapped = session._make_multi_dispatch_wrapper([ps1, ps64], lambda *args, **kwargs: "ok")

    assert wrapped(_FakeWrapper(page_size=1), "q", ()) == "ok"
    assert written == ["paged_ps1"]


def test_page_size_dispatch_keeps_only_matching_paged_target(tmp_path: Path) -> None:
    ps1 = ProbeTarget(
        name="paged_ps1",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=1,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    ps64 = ProbeTarget(
        name="paged_ps64",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=64,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[ps1, ps64], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=4,
    )
    written: list[str] = []
    session._write_event = lambda target, args, kwargs, scope=None, traced_definition=None: written.append(target.name)  # type: ignore[method-assign]

    wrapped = session._make_multi_dispatch_wrapper([ps1, ps64], lambda *args, **kwargs: "ok")
    kv_cache = (_FakeTensor((16, 64, 8, 128)), _FakeTensor((16, 64, 8, 128)))

    assert wrapped("self", "q", kv_cache) == "ok"
    assert written == ["paged_ps64"]


def test_page_size_inference_treats_3d_cache_as_squeezed_ps1() -> None:
    kv_cache = (_FakeTensor((182823, 8, 128)), _FakeTensor((182823, 8, 128)))

    assert infer_dispatch_value(_page_size_dispatch_spec(), (_FakeWrapper(), "q", kv_cache)) == 1


def test_page_size_inference_ignores_active_probe_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    kv_cache = (_FakeTensor((182823, 8, 128)), _FakeTensor((182823, 8, 128)))
    monkeypatch.setenv("FLASHINFER_TRACE_ACTIVE_PROBE_MODE", "paged_ps64")

    assert infer_dispatch_value(_page_size_dispatch_spec(), (_FakeWrapper(), "q", kv_cache)) == 1


def test_page_size_dispatch_does_not_route_squeezed_ps1_cache_to_ps64(tmp_path: Path) -> None:
    ps1 = ProbeTarget(
        name="paged_ps1",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=1,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    ps64 = ProbeTarget(
        name="paged_ps64",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        op_type="gqa_paged",
        page_size=64,
        capture=_capture_spec(),
        dispatch=_page_size_dispatch_spec(),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[ps1, ps64], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=4,
    )
    written: list[str] = []
    session._write_event = lambda target, args, kwargs, scope=None, traced_definition=None: written.append(target.name)  # type: ignore[method-assign]

    wrapped = session._make_multi_dispatch_wrapper([ps1, ps64], lambda *args, **kwargs: "ok")
    kv_cache = (_FakeTensor((182823, 8, 128)), _FakeTensor((182823, 8, 128)))

    assert wrapped("self", "q", kv_cache) == "ok"
    assert written == ["paged_ps1"]


def test_event_report_omits_dispatch_details_when_values_match() -> None:
    target = ProbeTarget(
        name="paged_ps1",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        page_size=1,
        dispatch=_page_size_dispatch_spec(),
        capture=_capture_spec(),
    )

    report = build_event_report(
        [{"name": "paged_ps1", "is_warmup": False, "page_size": 1}],
        ProbePlan(targets=[target], skipped=[]),
    )

    assert "dispatch_errors" not in report
    assert "dispatch_errors" not in report["summary"]


def test_event_report_reports_dispatch_mismatch_only_on_error() -> None:
    target = ProbeTarget(
        name="paged_ps1",
        target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        module="flashinfer.decode",
        attr="BatchDecodeWithPagedKVCacheWrapper.run",
        page_size=1,
        dispatch=_page_size_dispatch_spec(),
        capture=_capture_spec(),
    )

    report = build_event_report(
        [
            {"name": "paged_ps1", "is_warmup": False, "page_size": 1},
            {"name": "paged_ps1", "is_warmup": False, "page_size": 64},
        ],
        ProbePlan(targets=[target], skipped=[]),
    )

    assert report["summary"]["dispatch_errors"] == 1
    assert report["dispatch_errors"] == [
        {
            "name": "paged_ps1",
            "field": "page_size",
            "reason": "observed dispatch value does not match target",
            "expected": 1,
            "observed": [1, 64],
        }
    ]


def test_capture_event_uses_fitrace_definition_name(tmp_path: Path) -> None:
    target = ProbeTarget(
        name="agent_preview_target",
        target="flashinfer.demo.traced",
        module="flashinfer.demo",
        attr="traced",
        definition_name="agent_preview_name",
        op_type="demo",
        backend="flashinfer",
        collect=True,
        capture=_capture_spec(),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[target], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=4,
    )
    session.captures_dir.mkdir(parents=True)
    session._write_capture_file = lambda capture_path, target, args, kwargs, scope=None, definition_name=None, traced_definition=None: capture_path  # type: ignore[method-assign]

    def original(x: int) -> int:
        return x + 1

    original.fi_trace = lambda **kwargs: {"name": "fitrace_definition_name"}  # type: ignore[attr-defined]
    wrapped = session._make_wrapper(target, original)

    assert wrapped(1) == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["definition_name"] == "fitrace_definition_name"


def test_fitrace_definition_inputs_extend_structural_capture_spec_without_forcing_float(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    target = ProbeTarget(
        name="fitrace_target",
        target="flashinfer.demo.traced",
        module="flashinfer.demo",
        attr="traced",
        op_type="demo",
        backend="flashinfer",
        collect=True,
        capture=_capture_spec(),
    )
    session = CaptureSession(
        probe_plan=ProbePlan(targets=[target], skipped=[]),
        output_dir=tmp_path,
        max_captures_per_target=4,
    )
    session.captures_dir.mkdir(parents=True)
    tensor = torch.zeros((2, 3), dtype=torch.float32)

    session._write_event(
        target,
        (tensor,),
        {},
        traced_definition={
            "name": "fitrace_definition_name",
            "_capture_arg_names": ["probs"],
            "inputs": {"probs": {"shape": ["batch", "vocab"], "dtype": "float32"}},
        },
    )

    capture_files = sorted((tmp_path / "captures").glob("*.pt"))
    assert len(capture_files) == 1
    payload = torch.load(capture_files[0], weights_only=False)
    assert payload["payload"]["args"][0]["saved"] is False
    assert payload["payload"]["args"][0]["summary"]["shape"] == [2, 3]
    assert "value" not in payload["payload"]["args"][0]


def _write_definition(root: Path, name: str, op_type: str = "rmsnorm") -> Path:
    path = root / op_type / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "name": name,
            "op_type": op_type,
            "axes": {},
            "inputs": {},
            "outputs": {},
            "reference": "def run(): pass\n",
        }),
        encoding="utf-8",
    )
    return path


def test_load_definitions_rejects_invalid_metadata(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    bad_name = definitions_dir / "rmsnorm" / "bad_name.json"
    bad_name.parent.mkdir(parents=True)
    bad_name.write_text(json.dumps({"name": "", "op_type": "rmsnorm"}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid name"):
        load_definitions(definitions_dir)

    bad_name.unlink()
    missing_op_type = definitions_dir / "rmsnorm" / "missing_op_type.json"
    missing_op_type.write_text(json.dumps({"name": "missing_op_type"}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid op_type"):
        load_definitions(definitions_dir)


def test_modal_plan_enables_remote_fitrace_collect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", ["alpha", "beta"])
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "batch_sizes": [1, 2],
            "max_new_tokens": 16,
            "supplemental_runs": [_supplemental_run()],
            "max_captures_per_target": 8,
        },
        approved=[
                {
                    "name": "approved_target",
                    "target": "flashinfer.norm.rmsnorm",
                    "module": "flashinfer.norm",
                    "attr": "rmsnorm",
                    "backend": "flashinfer",
                    "collect": True,
                }
        ],
    )

    monkeypatch.chdir(tmp_path)
    modal_plan = _build_modal_plan_from_run(
        run_dir,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        batch_sizes=[1, 2],
        max_new_tokens=16,
        supplemental_runs=[_supplemental_run()],
        max_captures_per_target=8,
    )

    assert modal_plan["capture_limits"]["max_captures_per_target"] == 8
    assert "remote_collect" not in modal_plan
    assert "definitions_dir" not in modal_plan


def test_modal_plan_respects_disable_cuda_graph_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json")
    run_dir = tmp_path / "runs" / "demo"
    definitions_dir = run_dir / "output" / "definitions"
    _write_definition(definitions_dir, "approved_target")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "disable_cuda_graph": False,
            **_collect_strategy(),
        },
        approved=[
            {
                "name": "approved_target",
                "target": "flashinfer.norm.rmsnorm",
                "module": "flashinfer.norm",
                "attr": "rmsnorm",
                "backend": "flashinfer",
                "collect": True,
            }
        ],
    )

    modal_plan = _build_modal_plan_from_run(
        run_dir,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        disable_cuda_graph=False,
        **_collect_strategy(),
    )
    assert modal_plan["sglang"]["disable_cuda_graph"] is False
    assert "remote_collect" not in modal_plan


def test_modal_plan_carries_reviewed_sglang_runtime_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json")
    run_dir = tmp_path / "runs" / "demo"
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "enable_piecewise_cuda_graph": True,
            "force_flashinfer_backends": True,
            "mem_fraction_static": 0.7,
            "cuda_graph_max_bs": 64,
            **_collect_strategy(),
        },
        approved=[
            {
                "name": "approved_target",
                "target": "flashinfer.norm.rmsnorm",
                "module": "flashinfer.norm",
                "attr": "rmsnorm",
                "backend": "flashinfer",
                "collect": True,
            }
        ],
    )

    modal_plan = _build_modal_plan_from_run(
        run_dir,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        enable_piecewise_cuda_graph=True,
        force_flashinfer_backends=True,
        mem_fraction_static=0.7,
        cuda_graph_max_bs=64,
        **_collect_strategy(),
    )
    assert modal_plan["sglang"]["enable_piecewise_cuda_graph"] is True
    assert modal_plan["sglang"]["force_flashinfer_backends"] is True
    assert modal_plan["sglang"]["mem_fraction_static"] == 0.7
    assert modal_plan["sglang"]["cuda_graph_max_bs"] == 64


def test_run_rejects_definitions_dir_in_run_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "definitions_dir": "definitions",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
        },
        approved=[],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashinfer_trace.cli",
            "run",
            "--run",
            "demo",
        ],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )

    assert result.returncode != 0
    assert "definitions_dir" in result.stderr


def test_run_rejects_unknown_run_config_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "old_unused_field": True,
        },
        approved=[],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashinfer_trace.cli",
            "run",
            "--run",
            "demo",
        ],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )

    assert result.returncode != 0
    assert "unsupported run_config.json fields" in result.stderr
    assert "old_unused_field" in result.stderr


def test_run_rejects_prompt_in_run_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "prompt": "do not use me",
        },
        approved=[],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashinfer_trace.cli",
            "run",
            "--run",
            "demo",
        ],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )

    assert result.returncode != 0
    assert "unsupported run_config.json fields" in result.stderr
    assert "prompt" in result.stderr


def test_modal_plan_can_use_model_and_run_id_subpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json")
    run_dir = tmp_path / "runs" / "llama31_8b" / "20260616_collect"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            **_collect_strategy(),
        },
        approved=[
            {
                "name": "approved_target",
                "target": "flashinfer.norm.rmsnorm",
                "module": "flashinfer.norm",
                "attr": "rmsnorm",
                "backend": "flashinfer",
                "collect": True,
            }
        ],
    )

    modal_plan = _build_modal_plan_from_run(
        run_dir,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        **_collect_strategy(),
    )

    assert modal_plan["runtime"]["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert modal_plan["runtime"]["gpu"] == "L40S"


def test_collect_plan_from_probe_plan_uses_fitrace_definitions(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    definition_path = _write_definition(
        definitions_dir,
        "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
        op_type="gqa_paged",
    )
    definitions = {
        "gqa_paged_prefill_causal_h32_kv8_d128_ps1": DefinitionRef(
            name="gqa_paged_prefill_causal_h32_kv8_d128_ps1",
            op_type="gqa_paged",
            path=definition_path,
        )
    }
    plan = ProbePlan(
        targets=[
            ProbeTarget(
                name="llama31_prefill",
                target="flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                module="flashinfer.prefill",
                attr="BatchPrefillWithPagedKVCacheWrapper.run",
                definition_name="gqa_paged_prefill_causal_h32_kv8_d128_ps1",
                op_type="gqa_paged",
                backend="flashinfer",
                collect=True,
                capture=_capture_spec(),
            )
        ],
        skipped=[],
    )

    events = [
        {
            "name": "llama31_prefill",
            "definition_name": "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
            "is_warmup": False,
        }
    ]

    collect_plan = build_collect_plan_from_probe_plan(definitions=definitions, probe_plan=plan, events=events)

    assert collect_plan.skipped == []
    assert len(collect_plan.targets) == 1
    assert collect_plan.targets[0].name == "llama31_prefill"
    assert collect_plan.targets[0].definition_name == "gqa_paged_prefill_causal_h32_kv8_d128_ps1"
    assert collect_plan.targets[0].definition_path == definition_path


def test_collect_plan_uses_event_linked_fitrace_definition(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    definition_path = _write_definition(
        definitions_dir,
        "gqa_paged_prefill_h32_kv128_d128_ps8",
        op_type="gqa_paged",
    )
    definitions = {
        "gqa_paged_prefill_h32_kv128_d128_ps8": DefinitionRef(
            name="gqa_paged_prefill_h32_kv128_d128_ps8",
            op_type="gqa_paged",
            path=definition_path,
            tags=[
                "fi_api:flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                "stage:prefill",
                "status:verified",
            ],
            axes={"page_size": {"type": "const", "value": 8}},
        )
    }
    plan = ProbePlan(
        targets=[
            ProbeTarget(
                name="llama31_prefill_ps8",
                target="flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                module="flashinfer.prefill",
                attr="BatchPrefillWithPagedKVCacheWrapper.run",
                definition_name="agent_preview_ps8",
                op_type="gqa_paged",
                variant="prefill",
                backend="flashinfer",
                collect=True,
                page_size=8,
                capture=_capture_spec(),
                dispatch=_page_size_dispatch_spec(),
            ),
            ProbeTarget(
                name="llama31_prefill_ps64",
                target="flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                module="flashinfer.prefill",
                attr="BatchPrefillWithPagedKVCacheWrapper.run",
                definition_name="agent_preview_ps64",
                op_type="gqa_paged",
                variant="prefill",
                backend="flashinfer",
                collect=True,
                page_size=64,
                capture=_capture_spec(),
                dispatch=_page_size_dispatch_spec(),
            ),
        ],
        skipped=[],
    )

    events = [
        {
            "name": "llama31_prefill_ps8",
            "definition_name": "gqa_paged_prefill_h32_kv128_d128_ps8",
            "is_warmup": False,
        }
    ]

    collect_plan = build_collect_plan_from_probe_plan(definitions=definitions, probe_plan=plan, events=events)

    assert len(collect_plan.targets) == 1
    assert collect_plan.targets[0].definition_name == "gqa_paged_prefill_h32_kv128_d128_ps8"
    assert collect_plan.targets[0].page_size == 8
    assert collect_plan.skipped == [
        {
            "name": "llama31_prefill_ps64",
            "reason": "no event-linked definition",
        }
    ]


def test_collect_plan_uses_repaired_definition_alias(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    repaired_path = _write_definition(
        definitions_dir,
        "gqa_paged_decode_h32_kv8_d128_ps1",
        op_type="gqa_paged",
    )
    definitions = {
        "gqa_paged_decode_h32_kv8_d128_ps1": DefinitionRef(
            name="gqa_paged_decode_h32_kv8_d128_ps1",
            op_type="gqa_paged",
            path=repaired_path,
            axes={"page_size": {"type": "const", "value": 1}},
        )
    }
    plan = ProbePlan(
        targets=[
            ProbeTarget(
                name="llama31_decode_ps1",
                target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                module="flashinfer.decode",
                attr="BatchDecodeWithPagedKVCacheWrapper.run",
                op_type="gqa_paged",
                variant="decode",
                backend="flashinfer",
                collect=True,
                page_size=1,
                capture=_capture_spec(),
                dispatch=_page_size_dispatch_spec(),
            )
        ],
        skipped=[],
    )
    events = [
        {
            "name": "llama31_decode_ps1",
            "definition_name": "gqa_paged_decode_h32_kv128_d128_ps8",
            "is_warmup": False,
        }
    ]

    collect_plan = build_collect_plan_from_probe_plan(
        definitions=definitions,
        probe_plan=plan,
        events=events,
        definition_aliases={
            "gqa_paged_decode_h32_kv128_d128_ps8": "gqa_paged_decode_h32_kv8_d128_ps1"
        },
    )

    assert collect_plan.skipped == []
    assert len(collect_plan.targets) == 1
    assert collect_plan.targets[0].definition_name == "gqa_paged_decode_h32_kv8_d128_ps1"
    assert collect_plan.targets[0].definition_path == repaired_path
    assert collect_plan.targets[0].page_size == 1


def test_collect_plan_accepts_reviewed_non_fitrace_definition(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    definition_path = _write_definition(
        definitions_dir,
        "torch_rotary_embedding",
        op_type="rotary_embedding",
    )
    definitions = {
        "torch_rotary_embedding": DefinitionRef(
            name="torch_rotary_embedding",
            op_type="rotary_embedding",
            path=definition_path,
        )
    }
    plan = ProbePlan(
        targets=[
            ProbeTarget(
                name="rotary_target",
                target="sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward",
                module="sglang.srt.layers.rotary_embedding",
                attr="RotaryEmbedding.forward",
                definition_name="torch_rotary_embedding",
                op_type="rotary_embedding",
                backend="torch",
                collect=True,
                capture=_capture_spec(),
            )
        ],
        skipped=[],
    )
    events = [
        {
            "name": "rotary_target",
            "definition_name": "torch_rotary_embedding",
            "is_warmup": False,
        }
    ]

    collect_plan = build_collect_plan_from_probe_plan(
        definitions=definitions,
        probe_plan=plan,
        events=events,
    )

    assert collect_plan.skipped == []
    assert len(collect_plan.targets) == 1
    assert collect_plan.targets[0].backend == "torch"
    assert collect_plan.targets[0].definition_name == "torch_rotary_embedding"
    assert collect_plan.targets[0].definition_path == definition_path


def test_probe_plan_requires_definition_name_for_non_fitrace_collect() -> None:
    with pytest.raises(ValueError, match="collect requires definition_name"):
        build_probe_plan([
            ApprovedTarget(
                name="manual_collect",
                target="sglang.custom.kernel",
                module="sglang.custom",
                attr="kernel",
                backend="torch",
                collect=True,
                capture=_capture_spec(),
            )
        ])


def test_reviewed_definition_artifacts_materialize_for_remote_collect(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    definition_dir = run_dir / "config" / "definitions" / "rotary_embedding"
    definition_dir.mkdir(parents=True)
    definition = {
        "name": "torch_rotary_embedding",
        "op_type": "rotary_embedding",
        "axes": {"num_tokens": {"type": "var"}},
        "inputs": {"query": {"shape": ["num_tokens"], "dtype": "bfloat16"}},
        "outputs": {"output": {"shape": ["num_tokens"], "dtype": "bfloat16"}},
        "reference": "def run(query):\n    return query\n",
    }
    (definition_dir / "torch_rotary_embedding.json").write_text(json.dumps(definition), encoding="utf-8")
    hints_dir = run_dir / "config" / "definition_hints" / "rotary_embedding"
    hints_dir.mkdir(parents=True)
    hints = {"schema_version": 1, "definition_name": "torch_rotary_embedding", "op_type": "rotary_embedding"}
    (hints_dir / "torch_rotary_embedding.json").write_text(json.dumps(hints), encoding="utf-8")

    artifacts = _reviewed_definition_artifacts(run_dir)
    assert artifacts["definitions"][0]["path"] == "rotary_embedding/torch_rotary_embedding.json"
    assert artifacts["definition_hints"][0]["path"] == "rotary_embedding/torch_rotary_embedding.json"

    output_definitions = tmp_path / "output" / "definitions"
    output_hints = tmp_path / "output" / "definition_hints"
    _materialize_reviewed_artifacts(
        modal_probe_plan={"reviewed_artifacts": artifacts},
        definitions_dir=output_definitions,
        hints_dir=output_hints,
    )

    assert json.loads((output_definitions / "rotary_embedding" / "torch_rotary_embedding.json").read_text()) == definition
    assert json.loads((output_hints / "rotary_embedding" / "torch_rotary_embedding.json").read_text()) == hints


def test_definition_audit_repairs_gqa_paged_3d_cache_shape(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_definitions"
    raw_path = _write_definition(
        raw_dir,
        "gqa_paged_prefill_h32_kv128_d128_ps8",
        op_type="gqa_paged",
    )
    raw_definition = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_definition.update(
        {
            "tags": [
                "fi_api:flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                "stage:prefill",
                "status:verified",
            ],
            "axes": {
                "num_qo_heads": {"type": "const", "value": 32},
                "num_kv_heads": {"type": "const", "value": 128},
                "head_dim": {"type": "const", "value": 128},
                "page_size": {"type": "const", "value": 8},
            },
            "inputs": {
                "q": {"shape": ["total_q", "num_qo_heads", "head_dim"], "dtype": "bfloat16"},
                "k_cache": {"shape": ["num_pages", "page_size", "num_kv_heads", "head_dim"], "dtype": "bfloat16"},
                "v_cache": {"shape": ["num_pages", "page_size", "num_kv_heads", "head_dim"], "dtype": "bfloat16"},
            },
        }
    )
    raw_path.write_text(json.dumps(raw_definition), encoding="utf-8")
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "name": "llama31_8b_gqa_paged_prefill_ps1",
                "target": "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                "variant": "prefill",
                "page_size": 1,
                "args": [
                    {"type": "BatchPrefillWithPagedKVCacheWrapper"},
                    {"type": "Tensor", "shape": [1755, 32, 128], "dtype": "torch.bfloat16"},
                    {
                        "type": "tuple",
                        "elements": [
                            {"type": "Tensor", "shape": [182823, 8, 128], "dtype": "torch.bfloat16"},
                            {"type": "Tensor", "shape": [182823, 8, 128], "dtype": "torch.bfloat16"},
                        ],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_and_repair_definitions(
        raw_definitions_dir=raw_dir,
        output_definitions_dir=tmp_path / "definitions",
        output_hints_dir=tmp_path / "definition_hints",
        events_path=events_path,
        report_dir=tmp_path / "definition_audit",
    )

    repaired_path = tmp_path / "definitions" / "gqa_paged" / "gqa_paged_prefill_causal_h32_kv8_d128_ps1.json"
    repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
    assert report["summary"]["repaired"] == 1
    assert report["summary"]["rejected"] == 0
    assert report["aliases"] == {
        "gqa_paged_prefill_h32_kv128_d128_ps8": "gqa_paged_prefill_causal_h32_kv8_d128_ps1"
    }
    assert repaired["axes"]["num_kv_heads"]["value"] == 8
    assert repaired["axes"]["page_size"]["value"] == 1
    assert "status:repaired" in repaired["tags"]
    hints_path = tmp_path / "definition_hints" / "gqa_paged" / "gqa_paged_prefill_causal_h32_kv8_d128_ps1.json"
    hints = json.loads(hints_path.read_text(encoding="utf-8"))
    assert hints["inputs"]["k_cache"][0]["source"] == "arg_tuple"
    assert hints["axes"]["num_pages"]["source"] == "tensor_max_plus_one"
    assert not (tmp_path / "definition_audit" / "definition_audit_report.json").exists()


def test_probe_plan_json_roundtrip_preserves_collect_fields() -> None:
    plan = ProbePlan(
        targets=[
            ProbeTarget(
                name="demo",
                target="flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                module="flashinfer.prefill",
                attr="BatchPrefillWithPagedKVCacheWrapper.run",
                backend="flashinfer",
                collect=True,
                capture=_capture_spec(),
            )
        ],
        skipped=[],
    )

    roundtripped = ProbePlan.from_jsonable(plan.to_jsonable())

    assert roundtripped.targets[0].backend == "flashinfer"
    assert roundtripped.targets[0].collect is True
    assert roundtripped.targets[0].capture == _capture_spec()


def test_collect_plan_json_roundtrip_preserves_typed_targets(tmp_path: Path) -> None:
    definition_path = tmp_path / "definitions" / "rms.json"
    plan = CollectPlan(
        targets=[
            CollectTarget(
                name="demo",
                definition_name="rms",
                op_type="rmsnorm",
                target="flashinfer.norm.rmsnorm",
                backend="flashinfer",
                collect=True,
                definition_path=definition_path,
            )
        ],
        skipped=[],
    )

    roundtripped = CollectPlan.from_jsonable(plan.to_jsonable())

    assert roundtripped.targets[0].definition_name == "rms"
    assert roundtripped.targets[0].definition_path == definition_path
    assert roundtripped.targets[0].backend == "flashinfer"


def test_modal_plan_uses_reviewed_collect_strategy_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", ["reviewed alpha", "reviewed beta"])
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "batch_sizes": [2],
            "max_new_tokens": 5,
            "supplemental_runs": [_supplemental_run(top_k=16)],
            "max_captures_per_target": 9,
        },
        approved=[
            {
                "name": "approved_target",
                "target": "flashinfer.norm.rmsnorm",
                "module": "flashinfer.norm",
                "attr": "rmsnorm",
                "backend": "flashinfer",
                "collect": True,
            }
        ],
    )

    modal_plan = _build_modal_plan_from_run(
        run_dir,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        image="lmsysorg/sglang:v0.5.12.post1",
        gpu="L40S",
        tp_size=1,
        batch_sizes=[2],
        max_new_tokens=5,
        supplemental_runs=[_supplemental_run(top_k=16)],
        max_captures_per_target=9,
    )
    assert "prompts" not in modal_plan
    assert "prompt" not in modal_plan
    assert "batch_sizes" not in modal_plan
    assert "capture_output" not in modal_plan
    assert modal_plan["supplemental_runs"][0]["sampling_params"] == {"temperature": 0.7, "top_k": 16, "top_p": 0.9}
    assert modal_plan["capture_limits"]["max_captures_per_target"] == 9


def test_run_collect_run_requires_remote_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flashinfer_trace import cli

    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", ["alpha"])
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "batch_sizes": [1],
            "max_new_tokens": 16,
            "supplemental_runs": [_supplemental_run()],
            "max_captures_per_target": 8,
        },
        approved=[
            {
                "name": "approved_target",
                "target": "flashinfer.norm.rmsnorm",
                "module": "flashinfer.norm",
                "attr": "rmsnorm",
                "backend": "flashinfer",
                "collect": True,
            }
        ],
    )

    def fake_run_modal_probe(*, modal_probe_plan, output_dir, timeout, resume_call_id=None):
        return {
            "parse_report": {"summary": {"events": 0, "missing_targets": 1}},
            "summary": {"events_written": 0},
        }

    monkeypatch.setattr(cli, "run_modal_probe", fake_run_modal_probe)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--run", "demo"])

    assert "remote collect was enabled but no workload manifest was returned" in str(exc_info.value)
    assert "definition audit under the run directory" in str(exc_info.value)
    assert not (run_dir / "reports" / "run_report.json").exists()


def test_run_diagnostic_full_scan_sets_modal_plan_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flashinfer_trace import cli

    monkeypatch.chdir(tmp_path)
    _write_sharegpt_fixture(tmp_path / "sharegpt_100.json", ["alpha"])
    run_dir = tmp_path / "runs" / "demo"
    _write_run_inputs(
        run_dir,
        config={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "image": "lmsysorg/sglang:v0.5.12.post1",
            "gpu": "L40S",
            "tp_size": 1,
            "batch_sizes": [1],
            "max_new_tokens": 16,
            "supplemental_runs": [_supplemental_run()],
            "max_captures_per_target": 8,
        },
        approved=[
            {
                "name": "approved_target",
                "target": "flashinfer.norm.rmsnorm",
                "module": "flashinfer.norm",
                "attr": "rmsnorm",
                "backend": "flashinfer",
                "collect": True,
            }
        ],
    )

    def fake_run_modal_probe(*, modal_probe_plan, output_dir, timeout, resume_call_id=None):
        assert modal_probe_plan["diagnostics"] == {"full_scan": True}
        raise RuntimeError("stop after plan assertion")

    monkeypatch.setattr(cli, "run_modal_probe", fake_run_modal_probe)

    with pytest.raises(RuntimeError, match="stop after plan assertion"):
        cli.main(["run", "--run", "demo", "--diagnostic-full-scan"])


def test_remote_diagnostic_full_scan_disables_early_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import flashinfer_trace.runners.modal_runner as modal_runner

    def fake_prepare_fitrace_dump(output_dir: Path) -> Path:
        path = output_dir / "definitions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fake_prepare_worker_injection(**kwargs) -> None:
        return None

    def fake_stage(**kwargs):
        output_dir = kwargs["output_dir"]
        collect_dir = output_dir / "collect"
        definitions_dir = output_dir / "audited_definitions"
        hints_dir = output_dir / "definition_hints"
        collect_dir.mkdir(parents=True, exist_ok=True)
        definitions_dir.mkdir(parents=True, exist_ok=True)
        hints_dir.mkdir(parents=True, exist_ok=True)
        return {
            "collect_dir": collect_dir,
            "audited_definitions_dir": definitions_dir,
            "definition_hints_dir": hints_dir,
            "definition_audit_report": {"summary": {"raw": 1, "passed": 1, "rejected": 0}},
            "collect_plan": {"targets": [], "skipped": []},
            "workload_manifest": {
                "summary": {"workloads": 0, "skipped": 1, "captures": 0, "sanitized": 0},
                "skipped": [{"name": "bad", "reason": "sanitize_failed"}],
            },
        }

    monkeypatch.setattr(modal_runner, "_prepare_fitrace_dump", fake_prepare_fitrace_dump)
    monkeypatch.setattr(modal_runner, "prepare_worker_injection", fake_prepare_worker_injection)
    monkeypatch.setattr(modal_runner, "_build_remote_post_capture_outputs", fake_stage)

    seen_early_checks = []
    modal_probe_plan = {
        "probe_plan": ProbePlan(targets=[], skipped=[], warmup_hooks=[]).to_jsonable(),
        "capture_limits": {"max_captures_per_target": 1},
        "diagnostics": {"full_scan": True},
    }

    result = run_remote_probe_entrypoint(
        modal_probe_plan=modal_probe_plan,
        output_dir=tmp_path / "remote",
        run_model=lambda early_check: seen_early_checks.append(early_check) or True,
    )

    assert seen_early_checks == [None]
    assert "early_stop" not in result
    assert result["summary"]["diagnostic_full_scan"] is True
    assert result["workload_manifest"]["summary"]["skipped"] == 1


def test_materialize_modal_result_extracts_remote_collect_and_redacts_archives(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    collect_dir = source_root / "collect"
    collect_dir.mkdir(parents=True)
    workload_path = collect_dir / "workloads" / "gqa_paged" / "demo.jsonl"
    workload_path.parent.mkdir(parents=True)
    workload_path.write_text("{}\n", encoding="utf-8")
    blob_path = collect_dir / "blob" / "workloads" / "gqa_paged" / "demo" / "x.safetensors"
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(b"blob")
    (collect_dir / "workload_manifest.json").write_text(
        json.dumps({
            "summary": {"workloads": 1},
            "workloads": [
                {
                    "name": "demo",
                    "definition_path": "/tmp/flashinfer-trace-probe/audited_definitions/gqa_paged/demo.json",
                    "capture_paths": ["/tmp/flashinfer-trace-probe/captures/x.pt"],
                    "workload_paths": [
                        "/tmp/flashinfer-trace-probe/collect/workloads/gqa_paged/demo.jsonl"
                    ],
                    "blob_paths": [
                        "/tmp/flashinfer-trace-probe/collect/blob/workloads/gqa_paged/demo/x.safetensors"
                    ],
                }
            ],
        }),
        encoding="utf-8",
    )
    (collect_dir / "collect_plan.json").write_text(
        json.dumps({
            "targets": [
                {
                    "name": "demo",
                    "definition_name": "demo",
                    "definition_path": "/tmp/flashinfer-trace-probe/audited_definitions/gqa_paged/demo.json",
                }
            ]
        }),
        encoding="utf-8",
    )
    archive_path = source_root / "collect.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(collect_dir, arcname="collect")
    definitions_dir = source_root / "definitions" / "gqa_paged"
    definitions_dir.mkdir(parents=True)
    (definitions_dir / "demo.json").write_text('{"name":"demo","op_type":"gqa_paged"}\n', encoding="utf-8")
    definitions_archive_path = source_root / "definitions.tar.gz"
    with tarfile.open(definitions_archive_path, "w:gz") as archive:
        archive.add(source_root / "definitions", arcname="definitions")
    hints_dir = source_root / "definition_hints" / "gqa_paged"
    hints_dir.mkdir(parents=True)
    (hints_dir / "demo.json").write_text(json.dumps(_gqa_paged_hints("demo")) + "\n", encoding="utf-8")
    hints_archive_path = source_root / "definition_hints.tar.gz"
    with tarfile.open(hints_archive_path, "w:gz") as archive:
        archive.add(source_root / "definition_hints", arcname="definition_hints")
    output_dir = tmp_path / "run" / ".modal_tmp"
    materialize_modal_result(
        {
            "parse_report": {"summary": {"events": 1, "missing_targets": 0}},
            "collect_archive_b64": base64.b64encode(archive_path.read_bytes()).decode("ascii"),
            "definitions_archive_b64": base64.b64encode(definitions_archive_path.read_bytes()).decode("ascii"),
            "definition_hints_archive_b64": base64.b64encode(hints_archive_path.read_bytes()).decode("ascii"),
            "definition_audit_report": {"summary": {"raw": 1, "passed": 1, "repaired": 0, "rejected": 0}},
            "summary": {"workloads": 1, "sanitized": 1},
        },
        output_dir,
    )

    assert not (output_dir / "events.jsonl").exists()
    assert not (output_dir / "parse_report.json").exists()
    result = json.loads((output_dir / "modal_result.json").read_text(encoding="utf-8"))
    assert result["parse_report"] == {"summary": {"events": 1, "missing_targets": 0}}
    local_manifest = result["workload_manifest"]
    workload = local_manifest["workloads"][0]
    assert workload["definition_path"] == str(tmp_path / "run" / "output" / "definitions" / "gqa_paged" / "demo.json")
    assert workload["capture_paths"] == ["/tmp/flashinfer-trace-probe/captures/x.pt"]
    assert workload["workload_paths"] == [
        str(tmp_path / "run" / "output" / "workloads" / "gqa_paged" / "demo.jsonl")
    ]
    assert workload["blob_paths"] == [
        str(
            tmp_path
            / "run"
            / "output"
            / "blob"
            / "workloads"
            / "gqa_paged"
            / "demo"
            / "x.safetensors"
        )
    ]
    collect_plan = result["collect_plan"]
    assert collect_plan["targets"][0]["definition_path"] == str(
        tmp_path / "run" / "output" / "definitions" / "gqa_paged" / "demo.json"
    )
    assert (tmp_path / "run" / "output" / "definitions" / "gqa_paged" / "demo.json").exists()
    assert (tmp_path / "run" / "output" / "definition_hints" / "gqa_paged" / "demo.json").exists()
    assert (tmp_path / "run" / "output" / "workloads" / "gqa_paged" / "demo.jsonl").exists()
    assert not (output_dir / "captures").exists()

    assert result["collect_archive_b64"]["redacted"] is True
    assert result["definitions_archive_b64"]["redacted"] is True
    assert result["definition_hints_archive_b64"]["redacted"] is True
    assert not (tmp_path / "run" / "collect.tar.gz").exists()
    assert not (tmp_path / "run" / "definitions.tar.gz").exists()
    assert not (tmp_path / "run" / "definition_hints.tar.gz").exists()


def test_workload_manifest_skips_unsanitized_capture_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    captures_dir = run_dir / "captures"
    captures_dir.mkdir(parents=True)
    capture_path = captures_dir / "000001_rms.pt"
    capture_path.write_bytes(b"capture")
    events_path = run_dir / "events.jsonl"
    events_path.write_text(
        json.dumps({
            "name": "rms",
            "definition_name": "rms",
            "is_warmup": False,
            "capture_path": str(capture_path),
        }) + "\n",
        encoding="utf-8",
    )
    collect_plan_path = tmp_path / "collect_plan.json"
    definition_path = tmp_path / "definitions" / "rms.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(
        json.dumps({
            "name": "rms",
            "op_type": "rmsnorm",
            "inputs": {},
        }),
        encoding="utf-8",
    )
    collect_plan_path.write_text(
        json.dumps({
            "targets": [
                {
                    "name": "rms",
                    "definition_name": "rms",
                    "op_type": "rmsnorm",
                    "target": "flashinfer.norm.rmsnorm",
                    "backend": "flashinfer",
                    "collect": True,
                    "definition_path": str(definition_path),
                }
            ]
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "collect"

    manifest = build_workload_manifest(
        load_jsonl(events_path),
        CollectPlan.from_jsonable(json.loads(collect_plan_path.read_text(encoding="utf-8"))),
        output_dir=output_dir,
    )

    assert manifest["summary"]["workloads"] == 0
    assert manifest["summary"]["captures"] == 0
    assert manifest["summary"]["workload_files"] == 0
    assert manifest["summary"]["sanitized"] == 0
    assert manifest["skipped"][0]["name"] == "rms"
    assert manifest["skipped"][0]["reason"] == "sanitize_failed"
    assert manifest["skipped"][0]["sanitize_reject_reasons"] == {"capture_load_failed:UnpicklingError": 1}
    assert not (output_dir / "workloads" / "rmsnorm" / "rms.jsonl").exists()


def test_workload_manifest_matches_repaired_definition_alias(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.pt"
    capture_path.write_bytes(b"not a torch payload")
    definition_path = tmp_path / "definitions" / "gqa_paged_decode_h32_kv8_d128_ps1.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(
        json.dumps({
            "name": "gqa_paged_decode_h32_kv8_d128_ps1",
            "op_type": "gqa_paged",
            "inputs": {},
        }),
        encoding="utf-8",
    )

    manifest = build_workload_manifest(
        [
            {
                "name": "decode",
                "definition_name": "gqa_paged_decode_h32_kv128_d128_ps8",
                "is_warmup": False,
                "capture_path": str(capture_path),
            }
        ],
        CollectPlan(
            targets=[
                CollectTarget(
                    name="decode",
                    definition_name="gqa_paged_decode_h32_kv8_d128_ps1",
                    op_type="gqa_paged",
                    target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                    backend="flashinfer",
                    collect=True,
                    definition_path=definition_path,
                    page_size=1,
                )
            ],
            skipped=[],
        ),
        output_dir=tmp_path / "collect",
        definition_aliases={
            "gqa_paged_decode_h32_kv128_d128_ps8": "gqa_paged_decode_h32_kv8_d128_ps1",
        },
    )

    assert manifest["summary"]["workloads"] == 0
    assert manifest["skipped"][0]["name"] == "decode"
    assert manifest["skipped"][0]["reason"] == "sanitize_failed"


def test_definition_audit_writes_gdn_hints(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "gdn"
    raw_dir.mkdir(parents=True)
    raw_definition = raw_dir / "gdn_prefill_qk16_v16_d128.json"
    raw_definition.write_text(
        json.dumps({
            "name": "gdn_prefill_qk16_v16_d128",
            "op_type": "gdn",
            "axes": {
                "total_seq_len": {"type": "var"},
                "num_seqs": {"type": "var"},
                "num_q_heads": {"type": "const", "value": 16},
                "num_v_heads": {"type": "const", "value": 16},
                "head_size": {"type": "const", "value": 128},
                "len_cu_seqlens": {"type": "var"},
            },
                "inputs": {
                    "q": {"shape": ["total_seq_len", "num_q_heads", "head_size"], "dtype": "bfloat16"},
                "state": {
                    "shape": ["num_seqs", "num_v_heads", "head_size", "head_size"],
                    "dtype": "float32",
                    "optional": True,
                },
                "A_log": {"shape": ["num_v_heads"], "dtype": "unknown", "optional": True},
                "a": {"shape": ["total_seq_len", "num_v_heads"], "dtype": "float32"},
                "dt_bias": {"shape": ["num_v_heads"], "dtype": "unknown", "optional": True},
                "b": {"shape": ["total_seq_len", "num_v_heads"], "dtype": "float32"},
                    "cu_seqlens": {"shape": ["len_cu_seqlens"], "dtype": "int64"},
                    "scale": {"shape": None, "dtype": "float32", "optional": True},
                },
                "outputs": {
                    "output": {"shape": ["total_seq_len", "num_v_heads", "head_size"], "dtype": "bfloat16"}
                },
                "reference": "def run(q, state, a, b, cu_seqlens, scale=None):\n    return q\n",
            }),
            encoding="utf-8",
        )
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    report = audit_and_repair_definitions(
        raw_definitions_dir=tmp_path / "raw",
        output_definitions_dir=tmp_path / "definitions",
        output_hints_dir=tmp_path / "definition_hints",
        events_path=events_path,
        report_dir=tmp_path / "reports",
    )

    assert report["summary"]["hints"] == 1
    assert report["summary"]["repaired"] == 1
    output_definition = json.loads(
        (tmp_path / "definitions" / "gdn" / "gdn_prefill_qk16_v16_d128.json").read_text(encoding="utf-8")
    )
    assert "A_log" not in output_definition["inputs"]
    assert "dt_bias" not in output_definition["inputs"]
    assert "scale" not in output_definition["inputs"]
    hints_path = tmp_path / "definition_hints" / "gdn" / "gdn_prefill_qk16_v16_d128.json"
    hints = json.loads(hints_path.read_text(encoding="utf-8"))
    assert hints["inputs"]["state"] == [{"source": "kwarg", "name": "initial_state"}]
    assert hints["inputs"]["a"] == [{"source": "kwarg", "name": "g"}]
    assert hints["inputs"]["b"] == [{"source": "kwarg", "name": "beta"}]
    assert "A_log" not in hints["inputs"]
    assert "dt_bias" not in hints["inputs"]
    assert "scale" not in hints["inputs"]
    assert hints["axes"]["num_seqs"] == {"source": "tensor_numel_minus_one", "input": "cu_seqlens"}


def test_prepare_reviewed_rmsnorm_definition_makes_eps_scalar() -> None:
    prepared, fixes, issue = prepare_definition_for_output({
        "name": "rmsnorm_h1024",
        "op_type": "rmsnorm",
        "axes": {
            "num_tokens": {"type": "var"},
            "hidden_size": {"type": "const", "value": 1024},
        },
        "inputs": {
            "input": {"shape": ["num_tokens", "hidden_size"], "dtype": "bfloat16"},
            "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
            "eps": {"shape": [], "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["num_tokens", "hidden_size"], "dtype": "bfloat16"},
        },
        "reference": "def run(input, weight, eps):\n    return input\n",
    })

    assert issue is None
    assert "definition_repair:rmsnorm:rmsnorm_scalar_eps" in fixes
    assert prepared["inputs"]["eps"]["shape"] is None


def test_audit_writes_module_level_rmsnorm_hints(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "rmsnorm"
    raw_dir.mkdir(parents=True)
    (raw_dir / "rmsnorm_h1024.json").write_text(
        json.dumps({
            "name": "rmsnorm_h1024",
            "op_type": "rmsnorm",
            "axes": {
                "num_tokens": {"type": "var"},
                "hidden_size": {"type": "const", "value": 1024},
            },
            "inputs": {
                "input": {"shape": ["num_tokens", "hidden_size"], "dtype": "bfloat16"},
                "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
                "eps": {"shape": None, "dtype": "float32"},
            },
            "outputs": {
                "output": {"shape": ["num_tokens", "hidden_size"], "dtype": "bfloat16"},
            },
            "tags": ["backend:sglang_kernel"],
            "reference": "def run(input, weight, eps):\n    return input\n",
        }),
        encoding="utf-8",
    )
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    report = audit_and_repair_definitions(
        raw_definitions_dir=tmp_path / "raw",
        output_definitions_dir=tmp_path / "definitions",
        output_hints_dir=tmp_path / "definition_hints",
        events_path=events_path,
        report_dir=tmp_path / "reports",
    )

    assert report["summary"]["passed"] == 1
    hints = json.loads((tmp_path / "definition_hints" / "rmsnorm" / "rmsnorm_h1024.json").read_text())
    assert hints["inputs"]["input"] == [{"source": "arg", "arg_index": 0}]
    assert hints["inputs"]["weight"] == [{"source": "arg", "arg_index": 1}]
    assert hints["inputs"]["eps"] == [{"source": "arg", "arg_index": 2}]


def test_audit_writes_gemma_rmsnorm_fitrace_hints(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "rmsnorm"
    raw_dir.mkdir(parents=True)
    (raw_dir / "gemma_fused_add_rmsnorm_h1024.json").write_text(
        json.dumps({
            "name": "gemma_fused_add_rmsnorm_h1024",
            "op_type": "rmsnorm",
            "axes": {
                "batch_size": {"type": "var"},
                "hidden_size": {"type": "const", "value": 1024},
            },
            "inputs": {
                "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
                "residual": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
                "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
            },
            "outputs": {
                "output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
            },
            "tags": ["fi_api:flashinfer.norm.gemma_fused_add_rmsnorm", "model:gemma"],
            "reference": "def run(input, residual, weight):\n    return input\n",
        }),
        encoding="utf-8",
    )
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    report = audit_and_repair_definitions(
        raw_definitions_dir=tmp_path / "raw",
        output_definitions_dir=tmp_path / "definitions",
        output_hints_dir=tmp_path / "definition_hints",
        events_path=events_path,
        report_dir=tmp_path / "reports",
    )

    assert report["summary"]["passed"] == 1
    assert report["summary"]["hints"] == 1
    hints = json.loads(
        (tmp_path / "definition_hints" / "rmsnorm" / "gemma_fused_add_rmsnorm_h1024.json")
        .read_text(encoding="utf-8")
    )
    assert hints["inputs"]["hidden_states"] == [{"source": "arg", "arg_index": 0}]
    assert hints["inputs"]["residual"] == [{"source": "arg", "arg_index": 1}]
    assert hints["inputs"]["weight"] == [{"source": "arg", "arg_index": 2}]


def test_audit_writes_cascade_merge_fitrace_hints(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "cascade_merge"
    raw_dir.mkdir(parents=True)
    (raw_dir / "merge_state_h8_d256.json").write_text(
        json.dumps({
            "name": "merge_state_h8_d256",
            "op_type": "cascade_merge",
            "axes": {
                "seq_len": {"type": "var"},
                "num_heads": {"type": "const", "value": 8},
                "head_dim": {"type": "const", "value": 256},
            },
            "inputs": {
                "v_a": {"shape": ["seq_len", "num_heads", "head_dim"], "dtype": "bfloat16"},
                "s_a": {"shape": ["seq_len", "num_heads"], "dtype": "float32"},
                "v_b": {"shape": ["seq_len", "num_heads", "head_dim"], "dtype": "bfloat16"},
                "s_b": {"shape": ["seq_len", "num_heads"], "dtype": "float32"},
            },
            "outputs": {
                "v_merged": {"shape": ["seq_len", "num_heads", "head_dim"], "dtype": "bfloat16"},
                "s_merged": {"shape": ["seq_len", "num_heads"], "dtype": "float32"},
            },
            "tags": ["fi_api:flashinfer.cascade.merge_state"],
            "reference": "def run(v_a, s_a, v_b, s_b):\n    return v_a, s_a\n",
        }),
        encoding="utf-8",
    )
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    report = audit_and_repair_definitions(
        raw_definitions_dir=tmp_path / "raw",
        output_definitions_dir=tmp_path / "definitions",
        output_hints_dir=tmp_path / "definition_hints",
        events_path=events_path,
        report_dir=tmp_path / "reports",
    )

    assert report["summary"]["passed"] == 1
    assert report["summary"]["hints"] == 1
    hints = json.loads(
        (tmp_path / "definition_hints" / "cascade_merge" / "merge_state_h8_d256.json")
        .read_text(encoding="utf-8")
    )
    assert hints["inputs"] == {
        "v_a": [{"source": "arg", "arg_index": 0}],
        "s_a": [{"source": "arg", "arg_index": 1}],
        "v_b": [{"source": "arg", "arg_index": 2}],
        "s_b": [{"source": "arg", "arg_index": 3}],
    }


def test_audit_writes_silu_and_mul_hints(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "silu_and_mul"
    raw_dir.mkdir(parents=True)
    (raw_dir / "silu_and_mul_i3584.json").write_text(
        json.dumps({
            "name": "silu_and_mul_i3584",
            "op_type": "silu_and_mul",
            "axes": {
                "num_tokens": {"type": "var"},
                "intermediate_size": {"type": "const", "value": 3584},
                "two_intermediate_size": {"type": "const", "value": 7168},
            },
            "inputs": {
                "input": {
                    "shape": ["num_tokens", "two_intermediate_size"],
                    "dtype": "bfloat16",
                },
            },
            "outputs": {
                "output": {
                    "shape": ["num_tokens", "intermediate_size"],
                    "dtype": "bfloat16",
                },
            },
            "tags": ["backend:sglang_kernel"],
            "reference": (
                "def run(input):\n"
                "    import torch\n"
                "    gate, up = input.chunk(2, dim=-1)\n"
                "    return torch.nn.functional.silu(gate) * up\n"
            ),
        }),
        encoding="utf-8",
    )
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    report = audit_and_repair_definitions(
        raw_definitions_dir=tmp_path / "raw",
        output_definitions_dir=tmp_path / "definitions",
        output_hints_dir=tmp_path / "definition_hints",
        events_path=events_path,
        report_dir=tmp_path / "reports",
    )

    assert report["summary"]["passed"] == 1
    assert report["summary"]["hints"] == 1
    hints = json.loads(
        (tmp_path / "definition_hints" / "silu_and_mul" / "silu_and_mul_i3584.json")
        .read_text(encoding="utf-8")
    )
    assert hints["inputs"]["input"] == [{"source": "arg", "arg_index": 0}]


def test_non_fitrace_reviewed_definition_hints_build_workload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    capture_path = tmp_path / "captures" / "000001_nonfi.pt"
    capture_path.parent.mkdir(parents=True)
    torch.save(
        {
            "schema_version": 1,
            "name": "nonfi_rotary",
            "definition_name": "torch_rotary_embedding",
            "target": "sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward",
            "op_type": "rotary_embedding",
            "is_warmup": False,
            "payload": {
                "args": [
                    {
                        "saved": True,
                        "summary": {"type": "Tensor", "shape": [2, 4], "dtype": "torch.float32"},
                        "value": torch.arange(8, dtype=torch.float32).reshape(2, 4),
                    }
                ],
                "kwargs": {},
            },
        },
        capture_path,
    )
    events = [
        {
            "name": "nonfi_rotary",
            "definition_name": "torch_rotary_embedding",
            "is_warmup": False,
            "capture_path": str(capture_path),
        }
    ]
    definitions_dir = tmp_path / "definitions" / "rotary_embedding"
    definitions_dir.mkdir(parents=True)
    definition_path = definitions_dir / "torch_rotary_embedding.json"
    definition_path.write_text(
        json.dumps({
            "name": "torch_rotary_embedding",
            "op_type": "rotary_embedding",
            "axes": {
                "batch_size": {"type": "var"},
                "hidden_size": {"type": "const", "value": 4},
            },
            "inputs": {
                "x": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
            },
        }),
        encoding="utf-8",
    )
    hints_dir = tmp_path / "definition_hints" / "rotary_embedding"
    hints_dir.mkdir(parents=True)
    (hints_dir / "torch_rotary_embedding.json").write_text(
        json.dumps({
            "schema_version": 1,
            "definition_name": "torch_rotary_embedding",
            "op_type": "rotary_embedding",
            "inputs": {"x": [{"source": "arg", "arg_index": 0}]},
            "real_inputs": ["x"],
        }),
        encoding="utf-8",
    )
    collect_plan = CollectPlan(
        targets=[
            CollectTarget(
                name="nonfi_rotary",
                definition_name="torch_rotary_embedding",
                op_type="rotary_embedding",
                target="sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward",
                backend="torch",
                collect=True,
                definition_path=definition_path,
            )
        ],
        skipped=[],
    )

    manifest = build_workload_manifest(
        events,
        collect_plan,
        output_dir=tmp_path / "collect",
        hints_dir=tmp_path / "definition_hints",
    )

    assert manifest["summary"]["workloads"] == 1
    assert manifest["summary"]["sanitized"] == 1
    assert manifest["skipped"] == []
    workload_path = tmp_path / "collect" / "workloads" / "rotary_embedding" / "torch_rotary_embedding.jsonl"
    assert workload_path.exists()
    entry = json.loads(workload_path.read_text(encoding="utf-8").strip())
    assert entry["workload"]["axes"] == {"batch_size": 2}
    input_spec = entry["workload"]["inputs"]["x"]
    assert input_spec["type"] == "safetensors"
    assert (tmp_path / "collect" / input_spec["path"][2:]).exists()


def test_sanitizer_randomizes_float_and_stores_structural_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    definition = {
        "name": "demo_decode",
        "op_type": "gqa_paged",
        "axes": {
            "batch_size": {"type": "var"},
            "num_qo_heads": {"type": "const", "value": 2},
            "head_dim": {"type": "const", "value": 4},
            "len_indptr": {"type": "var"},
        },
        "inputs": {
            "q": {"shape": ["batch_size", "num_qo_heads", "head_dim"], "dtype": "bfloat16"},
            "kv_indptr": {"shape": ["len_indptr"], "dtype": "int32"},
            "sm_scale": {"shape": None, "dtype": "float32"},
        },
    }
    captured = {
        "payload": {
            "args": [
                {
                    "saved": False,
                    "summary": {"type": "Wrapper"},
                    "attrs": {
                        "_paged_kv_indptr_buf": {
                            "saved": True,
                            "summary": {"type": "Tensor", "shape": [4], "dtype": "torch.int32"},
                            "value": torch.tensor([0, 1, 3, 5], dtype=torch.int32),
                        }
                    },
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [3, 2, 4], "dtype": "torch.bfloat16"},
                    "value": torch.zeros((3, 2, 4), dtype=torch.bfloat16),
                },
            ],
            "kwargs": {
                "sm_scale": {"saved": True, "summary": {"type": "float", "value": 0.5}, "value": 0.5}
            },
        }
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints=_gqa_paged_hints(),
        output_dir=tmp_path,
    )
    assert diagnostic == {}
    assert entry is not None

    assert entry["workload"]["axes"] == {"batch_size": 3, "len_indptr": 4}
    assert entry["workload"]["inputs"]["q"] == {"type": "random"}
    assert entry["workload"]["inputs"]["sm_scale"] == {"type": "scalar", "value": 0.5}
    kv_indptr = entry["workload"]["inputs"]["kv_indptr"]
    assert kv_indptr["type"] == "safetensors"
    assert Path(tmp_path / kv_indptr["path"][2:]).exists()


def test_sanitizer_rejects_missing_scalar_instead_of_defaulting(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    definition = {
        "name": "demo_decode",
        "op_type": "gqa_paged",
        "axes": {
            "batch_size": {"type": "var"},
            "num_qo_heads": {"type": "const", "value": 2},
            "head_dim": {"type": "const", "value": 4},
        },
        "inputs": {
            "q": {"shape": ["batch_size", "num_qo_heads", "head_dim"], "dtype": "bfloat16"},
            "sm_scale": {"shape": None, "dtype": "float32"},
        },
    }
    captured = {
        "payload": {
            "args": [
                {
                    "saved": False,
                    "summary": {"type": "Wrapper"},
                    "attrs": {},
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [3, 2, 4], "dtype": "torch.bfloat16"},
                    "value": torch.zeros((3, 2, 4), dtype=torch.bfloat16),
                },
            ],
            "kwargs": {},
        }
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints=_gqa_paged_hints(),
        output_dir=tmp_path,
    )

    assert entry is None
    assert diagnostic == {"reason": "missing_scalar:sm_scale"}


def test_sanitizer_skips_missing_optional_scalar(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    definition = {
        "name": "demo_decode",
        "op_type": "gdn",
        "axes": {
            "batch_size": {"type": "var"},
            "num_q_heads": {"type": "const", "value": 2},
            "head_size": {"type": "const", "value": 4},
        },
        "inputs": {
            "q": {"shape": ["batch_size", "num_q_heads", "head_size"], "dtype": "bfloat16"},
            "scale": {"shape": None, "dtype": "float32", "optional": True},
        },
    }
    captured = {
        "payload": {
            "args": [],
            "kwargs": {
                "q": {
                    "saved": False,
                    "summary": {"type": "Tensor", "shape": [3, 2, 4], "dtype": "torch.bfloat16"},
                    "value": torch.zeros((3, 2, 4), dtype=torch.bfloat16),
                },
                "scale": {"saved": True, "summary": {"type": "NoneType", "value": None}, "value": None},
            },
        }
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints={"inputs": {"q": [{"source": "kwarg", "name": "q"}], "scale": [{"source": "kwarg", "name": "scale"}]}},
        output_dir=tmp_path,
    )

    assert diagnostic == {}
    assert entry is not None
    assert "scale" not in entry["workload"]["inputs"]


def test_sanitizer_builds_gdn_prefill_with_hints(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    definition = {
        "name": "gdn_prefill_qk2_v2_d4",
        "op_type": "gdn",
        "axes": {
            "total_seq_len": {"type": "var"},
            "num_seqs": {"type": "var"},
            "num_q_heads": {"type": "const", "value": 2},
            "num_k_heads": {"type": "const", "value": 2},
            "num_v_heads": {"type": "const", "value": 2},
            "head_size": {"type": "const", "value": 4},
            "len_cu_seqlens": {"type": "var"},
        },
        "inputs": {
            "q": {"shape": ["total_seq_len", "num_q_heads", "head_size"], "dtype": "bfloat16"},
            "k": {"shape": ["total_seq_len", "num_k_heads", "head_size"], "dtype": "bfloat16"},
            "v": {"shape": ["total_seq_len", "num_v_heads", "head_size"], "dtype": "bfloat16"},
            "state": {
                "shape": ["num_seqs", "num_v_heads", "head_size", "head_size"],
                "dtype": "float32",
                "optional": True,
            },
            "a": {"shape": ["total_seq_len", "num_v_heads"], "dtype": "float32"},
            "b": {"shape": ["total_seq_len", "num_v_heads"], "dtype": "float32"},
            "cu_seqlens": {"shape": ["len_cu_seqlens"], "dtype": "int64"},
            "scale": {"shape": None, "dtype": "float32", "optional": True},
        },
    }
    hints = {
        "inputs": {
            "q": [{"source": "kwarg", "name": "q"}],
            "k": [{"source": "kwarg", "name": "k"}],
            "v": [{"source": "kwarg", "name": "v"}],
            "state": [{"source": "kwarg", "name": "initial_state"}],
            "a": [{"source": "kwarg", "name": "g"}],
            "b": [{"source": "kwarg", "name": "beta"}],
            "cu_seqlens": [{"source": "kwarg", "name": "cu_seqlens"}],
            "scale": [{"source": "kwarg", "name": "scale"}],
        },
        "axes": {
            "total_seq_len": {"source": "tensor_last", "input": "cu_seqlens"},
            "num_seqs": {"source": "tensor_numel_minus_one", "input": "cu_seqlens"},
        },
    }
    captured = {
        "payload": {
            "args": [],
            "kwargs": {
                "q": {"saved": False, "summary": {"type": "Tensor", "shape": [3, 2, 4], "dtype": "torch.bfloat16"}},
                "k": {"saved": False, "summary": {"type": "Tensor", "shape": [3, 2, 4], "dtype": "torch.bfloat16"}},
                "v": {"saved": False, "summary": {"type": "Tensor", "shape": [3, 2, 4], "dtype": "torch.bfloat16"}},
                "g": {"saved": False, "summary": {"type": "Tensor", "shape": [3, 2], "dtype": "torch.float32"}},
                "beta": {"saved": False, "summary": {"type": "Tensor", "shape": [3, 2], "dtype": "torch.float32"}},
                "initial_state": {"saved": True, "summary": {"type": "NoneType", "value": None}, "value": None},
                "scale": {"saved": True, "summary": {"type": "NoneType", "value": None}, "value": None},
                "cu_seqlens": {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [3], "dtype": "torch.int64"},
                    "value": torch.tensor([0, 1, 3], dtype=torch.int64),
                },
            },
        }
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints=hints,
        output_dir=tmp_path,
    )

    assert diagnostic == {}
    assert entry is not None
    assert entry["workload"]["axes"] == {"total_seq_len": 3, "num_seqs": 2, "len_cu_seqlens": 3}
    assert entry["workload"]["inputs"]["a"] == {"type": "random"}
    assert entry["workload"]["inputs"]["b"] == {"type": "random"}
    assert entry["workload"]["inputs"]["cu_seqlens"]["type"] == "safetensors"
    assert "state" not in entry["workload"]["inputs"]
    assert "scale" not in entry["workload"]["inputs"]


def test_sanitizer_rejects_missing_random_tensor_summary(tmp_path: Path) -> None:
    definition = {
        "name": "demo_decode",
        "op_type": "gqa_paged",
        "axes": {
            "batch_size": {"type": "var"},
            "num_qo_heads": {"type": "const", "value": 2},
            "head_dim": {"type": "const", "value": 4},
        },
        "inputs": {
            "q": {"shape": ["batch_size", "num_qo_heads", "head_dim"], "dtype": "bfloat16"},
        },
    }
    captured = {
        "payload": {
            "args": [
                {
                    "saved": False,
                    "summary": {"type": "Wrapper"},
                    "attrs": {},
                }
            ],
            "kwargs": {},
        }
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints=_gqa_ragged_hints(),
        output_dir=tmp_path,
    )

    assert entry is None
    assert diagnostic["reason"] == "missing_tensor_summary:q"
    assert "payload_debug" in diagnostic


def test_sanitizer_reads_kv_cache_tuple_summary_for_random_inputs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    definition = {
        "name": "demo_decode",
        "op_type": "gqa_paged",
        "axes": {
            "batch_size": {"type": "var"},
            "num_qo_heads": {"type": "const", "value": 2},
            "num_kv_heads": {"type": "const", "value": 1},
            "head_dim": {"type": "const", "value": 4},
            "num_pages": {"type": "var"},
            "page_size": {"type": "const", "value": 1},
            "len_indptr": {"type": "var"},
            "num_kv_indices": {"type": "var"},
        },
        "inputs": {
            "q": {"shape": ["batch_size", "num_qo_heads", "head_dim"], "dtype": "bfloat16"},
            "k_cache": {"shape": ["num_pages", "page_size", "num_kv_heads", "head_dim"], "dtype": "bfloat16"},
            "v_cache": {"shape": ["num_pages", "page_size", "num_kv_heads", "head_dim"], "dtype": "bfloat16"},
            "kv_indptr": {"shape": ["len_indptr"], "dtype": "int32"},
            "kv_indices": {"shape": ["num_kv_indices"], "dtype": "int32"},
            "sm_scale": {"shape": None, "dtype": "float32"},
        },
    }
    captured = {
        "payload": {
            "args": [
                {
                    "saved": False,
                    "summary": {"type": "Wrapper"},
                    "attrs": {
                        "_paged_kv_indptr_buf": {
                            "saved": True,
                            "summary": {"type": "Tensor", "shape": [2], "dtype": "torch.int32"},
                            "value": torch.tensor([0, 3], dtype=torch.int32),
                        },
                        "_paged_kv_indices_buf": {
                            "saved": True,
                            "summary": {"type": "Tensor", "shape": [3], "dtype": "torch.int32"},
                            "value": torch.tensor([0, 1, 2], dtype=torch.int32),
                        },
                    },
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [1, 2, 4], "dtype": "torch.bfloat16"},
                    "value": torch.zeros((1, 2, 4), dtype=torch.bfloat16),
                },
                {
                    "saved": False,
                    "summary": {
                        "type": "tuple",
                        "elements": [
                            {"type": "Tensor", "shape": [3, 1, 4], "dtype": "torch.bfloat16"},
                            {"type": "Tensor", "shape": [3, 1, 4], "dtype": "torch.bfloat16"},
                        ],
                    },
                },
            ],
            "kwargs": {
                "sm_scale": {"saved": True, "summary": {"type": "float", "value": 0.5}, "value": 0.5}
            },
        }
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints=_gqa_paged_hints(),
        output_dir=tmp_path,
    )

    assert diagnostic == {}
    assert entry is not None
    assert entry["workload"]["inputs"]["k_cache"] == {"type": "random"}
    assert entry["workload"]["inputs"]["v_cache"] == {"type": "random"}


def test_sanitizer_randomizes_ragged_float_tensor_inputs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    definition = {
        "name": "demo_ragged",
        "op_type": "gqa_ragged",
        "axes": {
            "total_q": {"type": "var"},
            "num_qo_heads": {"type": "const", "value": 2},
            "num_kv_heads": {"type": "const", "value": 1},
            "head_dim": {"type": "const", "value": 4},
        },
        "inputs": {
            "q": {"shape": ["total_q", "num_qo_heads", "head_dim"], "dtype": "bfloat16"},
            "k": {"shape": ["total_q", "num_kv_heads", "head_dim"], "dtype": "bfloat16"},
            "v": {"shape": ["total_q", "num_kv_heads", "head_dim"], "dtype": "bfloat16"},
        },
    }
    captured = {
        "payload": {
            "args": [
                {
                    "saved": False,
                    "summary": {"type": "BatchPrefillWithRaggedKVCacheWrapper"},
                    "attrs": {},
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [3, 2, 4], "dtype": "torch.bfloat16"},
                    "value": torch.zeros((3, 2, 4), dtype=torch.bfloat16),
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [3, 1, 4], "dtype": "torch.bfloat16"},
                    "value": torch.zeros((3, 1, 4), dtype=torch.bfloat16),
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [3, 1, 4], "dtype": "torch.bfloat16"},
                    "value": torch.zeros((3, 1, 4), dtype=torch.bfloat16),
                },
            ],
            "kwargs": {
                "q": {
                    "saved": False,
                    "summary": {"type": "Tensor", "shape": [3, 2, 4], "dtype": "torch.bfloat16"},
                },
                "k": {
                    "saved": False,
                    "summary": {"type": "Tensor", "shape": [3, 1, 4], "dtype": "torch.bfloat16"},
                },
                "v": {
                    "saved": False,
                    "summary": {"type": "Tensor", "shape": [3, 1, 4], "dtype": "torch.bfloat16"},
                },
            },
        }
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints=_gqa_ragged_hints(),
        output_dir=tmp_path,
    )
    assert diagnostic == {}
    assert entry is not None

    assert entry["workload"]["axes"] == {"total_q": 3}
    for input_name in ("q", "k", "v"):
        assert entry["workload"]["inputs"][input_name] == {"type": "random"}
    assert not (tmp_path / "blob").exists()


def test_sanitizer_preserves_real_inputs_from_definition_hints(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    definition = {
        "name": "top_k_top_p_sampling_v8",
        "op_type": "sampling",
        "axes": {
            "batch_size": {"type": "var"},
            "vocab_size": {"type": "const", "value": 8},
        },
        "inputs": {
            "probs": {"shape": ["batch_size", "vocab_size"], "dtype": "float32"},
            "top_k": {"shape": ["batch_size"], "dtype": "int32"},
            "top_p": {"shape": ["batch_size"], "dtype": "float32"},
        },
    }
    captured = {
        "payload": {
            "args": [
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [2, 8], "dtype": "torch.float32"},
                    "value": torch.ones((2, 8), dtype=torch.float32),
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [2], "dtype": "torch.int32"},
                    "value": torch.tensor([4, 4], dtype=torch.int32),
                },
                {
                    "saved": True,
                    "summary": {"type": "Tensor", "shape": [2], "dtype": "torch.float32"},
                    "value": torch.tensor([0.9, 0.95], dtype=torch.float32),
                },
            ],
            "kwargs": {},
        }
    }
    hints = {
        "inputs": {
            "probs": [{"source": "arg", "arg_index": 0}],
            "top_k": [{"source": "arg", "arg_index": 1}],
            "top_p": [{"source": "arg", "arg_index": 2}],
        },
        "real_inputs": ["probs", "top_k", "top_p"],
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints=hints,
        output_dir=tmp_path,
    )

    assert diagnostic == {}
    assert entry is not None
    for input_name in ("probs", "top_k", "top_p"):
        input_payload = entry["workload"]["inputs"][input_name]
        assert input_payload["type"] == "safetensors"
        assert Path(tmp_path / input_payload["path"][2:]).exists()


def test_sanitizer_rejects_missing_real_input_tensor(tmp_path: Path) -> None:
    definition = {
        "name": "top_k_top_p_sampling_v8",
        "op_type": "sampling",
        "axes": {"batch_size": {"type": "var"}, "vocab_size": {"type": "const", "value": 8}},
        "inputs": {
            "probs": {"shape": ["batch_size", "vocab_size"], "dtype": "float32"},
        },
    }
    captured = {
        "payload": {
            "args": [
                {
                    "saved": False,
                    "summary": {"type": "Tensor", "shape": [2, 8], "dtype": "torch.float32"},
                }
            ],
            "kwargs": {},
        }
    }

    entry, diagnostic = build_sanitized_workload_entry(
        captured=captured,
        definition=definition,
        hints={"inputs": {"probs": [{"source": "arg", "arg_index": 0}]}, "real_inputs": ["probs"]},
        output_dir=tmp_path,
    )

    assert entry is None
    assert diagnostic == {"reason": "missing_real_tensor:probs"}


def test_probe_plan_rejects_unknown_fields_in_runtime_config(tmp_path: Path) -> None:
    approved_targets = tmp_path / "approved_targets.json"
    approved_targets.write_text(
        json.dumps([
            {
                "name": "approved_fi",
                "target": "flashinfer.norm.rmsnorm",
                "module": "flashinfer.norm",
                "attr": "rmsnorm",
                "op_type": "rmsnorm",
                "variant": "rmsnorm",
                "backend": "flashinfer",
                "probe_mode": "paged",
                "collect": True,
                "capture": _capture_json(),
            },
            {
                "name": "candidate_fi",
                "status": "candidate",
                "target": "flashinfer.norm.gemma_rmsnorm",
                "collect": False,
                "capture": _capture_json(),
            },
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected fields"):
        build_probe_plan(load_approved_targets(approved_targets))


def test_probe_plan_uses_reviewed_dotted_targets(tmp_path: Path) -> None:
    approved_targets = tmp_path / "approved_targets.json"
    approved_targets.write_text(
        json.dumps([
            {
                "name": "approved_fi",
                "target": "flashinfer.norm.rmsnorm",
                "module": "flashinfer.norm",
                "attr": "rmsnorm",
                "op_type": "rmsnorm",
                "variant": "rmsnorm",
                "backend": "flashinfer",
                "probe_mode": "paged",
                "collect": True,
                "capture": _capture_json(),
            },
            {
                "name": "approved_non_fi",
                "target": "sglang.srt.layers.attention.RadixAttention.forward",
                "module": "sglang.srt.layers.attention",
                "attr": "RadixAttention.forward",
                "collect": False,
                "capture": _capture_json(),
            },
        ]),
        encoding="utf-8",
    )

    plan = build_probe_plan(load_approved_targets(approved_targets))
    assert [target.name for target in plan.targets] == ["approved_fi", "approved_non_fi"]
    assert plan.targets[0].target == "flashinfer.norm.rmsnorm"
    assert plan.targets[0].probe_mode == "paged"
    assert plan.skipped == []


def test_probe_plan_rejects_duplicate_approved_targets() -> None:
    with pytest.raises(ValueError, match="duplicate reviewed target"):
        build_probe_plan([
            ApprovedTarget(
                name="dup_target",
                target="flashinfer.norm.rmsnorm",
                module="flashinfer.norm",
                attr="rmsnorm",
                capture=_capture_spec(),
            ),
            ApprovedTarget(
                name="dup_target",
                target="flashinfer.norm.gemma_rmsnorm",
                module="flashinfer.norm",
                attr="gemma_rmsnorm",
                capture=_capture_spec(),
            ),
        ])


def test_probe_plan_rejects_missing_reviewed_hook_specs(tmp_path: Path) -> None:
    approved_targets = tmp_path / "approved_targets.json"
    approved_targets.write_text(
        json.dumps([
            {
                "name": "bad_target",
                "target": "flashinfer.norm.gemma_rmsnorm",
                "collect": True,
                "capture": _capture_json(),
            }
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no explicit hook module/attr"):
        build_probe_plan(load_approved_targets(approved_targets))


def test_check_proposal_rejects_agent_guessed_collect_definition(tmp_path: Path) -> None:
    targets = tmp_path / "candidate_targets.json"
    targets.write_text(
        json.dumps([
            {
                "name": "bad_fi",
                "target": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                "module": "flashinfer.decode",
                "attr": "BatchDecodeWithPagedKVCacheWrapper.run",
                "backend": "flashinfer",
                "collect": True,
                "definition_source": "agent",
            },
            {
                "name": "manual_non_fi",
                "status": "candidate",
                "target": "sglang.srt.layers.attention.RadixAttention.forward",
                "backend": "sglang_kernel",
                "collect": False,
            },
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    report = check_proposal(candidates_path=targets, hf_config_path=hf_config)

    assert report["summary"]["ok"] is False
    assert any(
        item["reason"] == "collectable FlashInfer target must not use agent-guessed final definition"
        for item in report["findings"]
    )


def test_check_proposal_requires_companion_attrs_for_attention_wrappers(tmp_path: Path) -> None:
    targets = tmp_path / "candidate_targets.json"
    targets.write_text(
        json.dumps([
            {
                "name": "missing_companion",
                "status": "candidate",
                "target": "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                "module": "flashinfer.prefill",
                "attr": "BatchPrefillWithPagedKVCacheWrapper.run",
                "backend": "flashinfer",
                "collect": True,
                "definition_source": "fitrace",
            },
            {
                "name": "with_companion",
                "status": "candidate",
                "target": "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
                "module": "flashinfer.prefill",
                "attr": "BatchPrefillWithPagedKVCacheWrapper.run",
                "backend": "flashinfer",
                "collect": True,
                "definition_source": "fitrace",
                "companion_attrs": ["forward"],
            },
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    report = check_proposal(candidates_path=targets, hf_config_path=hf_config)

    companion_findings = [
        item for item in report["findings"]
        if "companion_attrs" in item["reason"]
    ]
    assert companion_findings == [
        {
            "check": "candidate_fields",
            "severity": "error",
            "name": "missing_companion",
            "reason": (
                "FlashInfer attention wrapper collect target must declare reviewed "
                "companion_attrs so scalars like sm_scale are captured from sibling calls"
            ),
        }
    ]


def test_check_proposal_allows_reviewed_non_fitrace_collect_candidate(tmp_path: Path) -> None:
    targets = tmp_path / "candidate_targets.json"
    targets.write_text(
        json.dumps([
            {
                "name": "rotary_collect",
                "status": "candidate",
                "target": "sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward",
                "module": "sglang.srt.layers.rotary_embedding",
                "attr": "RotaryEmbedding.forward",
                "backend": "torch",
                "collect": True,
                "definition_source": "agent",
                "definition_name": "torch_rotary_embedding",
                "op_type": "rotary_embedding",
                "capture": _capture_json(full_args=[1, 2]),
                "evidence": [{"kind": "source_location", "value": "sglang/srt/layers/rotary_embedding.py"}],
            }
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")
    definition_dir = tmp_path / "definitions" / "rotary_embedding"
    definition_dir.mkdir(parents=True)
    (definition_dir / "torch_rotary_embedding.json").write_text(
        json.dumps({
            "name": "torch_rotary_embedding",
            "op_type": "rotary_embedding",
            "axes": {"num_tokens": {"type": "var"}, "head_dim": {"type": "const", "value": 128}},
            "inputs": {"query": {"shape": ["num_tokens", "head_dim"], "dtype": "bfloat16"}},
            "outputs": {"output": {"shape": ["num_tokens", "head_dim"], "dtype": "bfloat16"}},
            "reference": "def run(query):\n    return query\n",
        }),
        encoding="utf-8",
    )
    hints_dir = tmp_path / "definition_hints" / "rotary_embedding"
    hints_dir.mkdir(parents=True)
    (hints_dir / "torch_rotary_embedding.json").write_text(
        json.dumps({
            "schema_version": 1,
            "definition_name": "torch_rotary_embedding",
            "op_type": "rotary_embedding",
            "inputs": {"query": [{"source": "arg", "arg_index": 1}]},
        }),
        encoding="utf-8",
    )

    report = check_proposal(candidates_path=targets, hf_config_path=hf_config)

    assert report["summary"]["ok"] is True
    assert report["summary"]["fitrace_targets"] == 0
    assert report["findings"] == []


def test_check_proposal_rejects_non_fitrace_collect_without_definition_name(tmp_path: Path) -> None:
    targets = tmp_path / "candidate_targets.json"
    targets.write_text(
        json.dumps([
            {
                "name": "bad_non_fi",
                "status": "candidate",
                "target": "sglang.custom.kernel",
                "module": "sglang.custom",
                "attr": "kernel",
                "backend": "torch",
                "collect": True,
                "definition_source": "manual",
                "capture": _capture_json(),
            }
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    report = check_proposal(candidates_path=targets, hf_config_path=hf_config)

    assert report["summary"]["ok"] is False
    assert any(
        item["reason"] == "non-FlashInfer collect target must declare reviewed definition_name"
        for item in report["findings"]
    )


def test_check_proposal_rejects_known_non_fi_manual_skip(tmp_path: Path) -> None:
    targets = tmp_path / "candidate_targets.json"
    targets.write_text(
        json.dumps([
            {
                "name": "silu_and_mul_context_only",
                "status": "candidate",
                "target": "sglang.srt.layers.activation.SiluAndMul.forward",
                "module": "sglang.srt.layers.activation",
                "attr": "SiluAndMul.forward",
                "backend": "sglang_kernel",
                "collect": False,
                "definition_source": "manual",
                "definition_name": "silu_and_mul_i3584",
                "op_type": "silu_and_mul",
                "capture": _capture_json(full_args=[1]),
                "evidence": [{"kind": "source_location", "value": "sglang/srt/layers/activation.py"}],
            }
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    report = check_proposal(candidates_path=targets, hf_config_path=hf_config)

    assert report["summary"]["ok"] is False
    reasons = {item["reason"] for item in report["findings"]}
    assert "known non-FlashInfer op silu_and_mul must be proposed with collect=true" in reasons
    assert "known non-FlashInfer op silu_and_mul must use definition_source=agent" in reasons


def test_check_proposal_checks_target_fi_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_dir = tmp_path / "mods"
    package = module_dir / "fakefi"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "api.py").write_text(
        "\n".join([
            "def traced():",
            "    pass",
            "def _fi_trace(**kwargs):",
            "    return {'name': 'from_fitrace'}",
            "traced.fi_trace = _fi_trace",
            "def untraced():",
            "    pass",
        ]),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_dir))
    hf_config = tmp_path / "config.json"
    hf_config.write_text(
        json.dumps({
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "vocab_size": 128256,
        }),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidate_targets.json"
    candidates.write_text(
        json.dumps([
            {
                "name": "good_fitrace",
                "definition_name": "agent_preview_name",
                "target": "fakefi.api.traced",
                "backend": "flashinfer",
                "definition_source": "fitrace",
                "collect": True,
            },
            {
                "name": "bad_untraced",
                "definition_name": "bad_untraced",
                "target": "fakefi.api.untraced",
                "backend": "flashinfer",
                "definition_source": "fitrace",
                "collect": True,
            },
            {
                "name": "bad_missing",
                "definition_name": "bad_missing",
                "target": "fakefi.api.missing",
                "backend": "flashinfer",
                "definition_source": "fitrace",
                "collect": True,
            },
        ]),
        encoding="utf-8",
    )

    report = check_proposal(
        candidates_path=candidates,
        hf_config_path=hf_config,
    )

    assert report["summary"]["ok"] is False
    assert report["summary"]["collect_candidates"] == 3
    assert report["summary"]["importable_targets"] == 2
    assert report["summary"]["fitrace_targets"] == 1
    preview_reasons = [
        item for item in report["findings"]
        if item["reason"] == "definition_name is only a preview for fitrace-backed targets; final name/schema must come from fitrace dump"
    ]
    assert {item["name"] for item in preview_reasons} >= {"good_fitrace", "bad_untraced"}
    assert any(
        item["name"] == "bad_untraced"
        and item["reason"] == "target is importable but has no callable .fi_trace; final definition cannot come from fitrace"
        for item in report["findings"]
    )
    assert any(
        item["name"] == "bad_missing"
        and item["reason"].startswith("target cannot be imported and source check failed")
        for item in report["findings"]
    )


def test_check_proposal_rejects_op_type_mismatch_with_trace_template(tmp_path: Path) -> None:
    flashinfer_root = tmp_path / "flashinfer"
    templates_dir = flashinfer_root / "trace" / "templates"
    templates_dir.mkdir(parents=True)
    (flashinfer_root / "gdn_prefill.py").write_text(
        "\n".join([
            "from .trace.templates.gdn import gdn_prefill_trace",
            "def flashinfer_api(func=None, *, trace=None):",
            "    def deco(f): return f",
            "    return deco(func) if func is not None else deco",
            "@flashinfer_api(trace=gdn_prefill_trace)",
            "def chunk_gated_delta_rule():",
            "    pass",
        ]),
        encoding="utf-8",
    )
    (templates_dir / "gdn.py").write_text(
        "gdn_prefill_trace = TraceTemplate(op_type='gdn', name_prefix='gdn_prefill')\n",
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text(json.dumps({"hidden_size": 1024}), encoding="utf-8")
    candidates = tmp_path / "candidate_targets.json"
    base_candidate = {
        "name": "gdn_prefill",
        "role": "target",
        "target": "flashinfer.gdn_prefill.chunk_gated_delta_rule",
        "module": "flashinfer.gdn_prefill",
        "attr": "chunk_gated_delta_rule",
        "backend": "flashinfer",
        "variant": "prefill",
        "definition_source": "fitrace",
        "collect": True,
        "capture": _capture_json(),
    }

    candidates.write_text(json.dumps([dict(base_candidate, op_type="gdn_prefill")]), encoding="utf-8")
    failed = check_proposal(
        candidates_path=candidates,
        hf_config_path=hf_config,
        flashinfer_root=flashinfer_root,
    )

    assert failed["summary"]["ok"] is False
    assert any("does not match FlashInfer trace template op_type 'gdn'" in item["reason"] for item in failed["findings"])

    candidates.write_text(json.dumps([dict(base_candidate, op_type="gdn")]), encoding="utf-8")
    passed = check_proposal(
        candidates_path=candidates,
        hf_config_path=hf_config,
        flashinfer_root=flashinfer_root,
    )

    assert passed["summary"]["ok"] is True
    assert passed["fitrace_eval"]["targets"][0]["trace_op_type"] == "gdn"


def test_check_proposal_cli_writes_report(tmp_path: Path) -> None:
    module_dir = tmp_path / "mods"
    package = module_dir / "fakefi"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "api.py").write_text(
        "def traced():\n    pass\ndef _fi_trace(**kwargs):\n    return {}\ntraced.fi_trace = _fi_trace\n",
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text(
        json.dumps({
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidate_targets.json"
    candidates.write_text(
        json.dumps([
            {
                "name": "traced",
                "target": "fakefi.api.traced",
                "backend": "flashinfer",
                "definition_source": "fitrace",
                "collect": True,
                "capture": _capture_json(),
            }
        ]),
        encoding="utf-8",
    )
    output = tmp_path / "fitrace_eval.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.proposal_tools",
            "check",
            "--candidates",
            str(candidates),
            "--hf-config",
            str(hf_config),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[1]), str(module_dir)]),
        },
    )

    assert result.returncode == 0
    assert "collect candidates: 1" in result.stdout
    assert "fitrace targets: 1" in result.stdout
    assert "errors: 0" in result.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["fitrace_targets"] == 1


def test_check_proposal_uses_flashinfer_source_check_for_overloaded_methods(tmp_path: Path) -> None:
    flashinfer_root = tmp_path / "flashinfer"
    flashinfer_root.mkdir()
    (flashinfer_root / "decode.py").write_text(
        "\n".join([
            "def flashinfer_api(func=None, *, trace=None):",
            "    def deco(f): return f",
            "    return deco(func) if func is not None else deco",
            "def overload(f): return f",
            "class BatchDecodeWithPagedKVCacheWrapper:",
            "    @overload",
            "    def run(self, q): ...",
            "    @flashinfer_api(trace='gqa')",
            "    def run(self, q):",
            "        return q",
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")
    candidates = tmp_path / "candidate_targets.json"
    candidates.write_text(
        json.dumps([
            {
                "name": "decode_run",
                "target": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                "backend": "flashinfer",
                "definition_source": "fitrace",
                "collect": True,
                "companion_attrs": ["forward"],
                "capture": _capture_json(),
            }
        ]),
        encoding="utf-8",
    )

    report = check_proposal(
        candidates_path=candidates,
        hf_config_path=hf_config,
        flashinfer_root=flashinfer_root,
    )

    assert report["summary"]["ok"] is True
    assert report["summary"]["fitrace_targets"] == 1
    assert report["fitrace_eval"]["targets"][0]["source_fitrace_ok"] is True


def test_check_proposal_suggests_decorated_run_for_wrapper_forward(tmp_path: Path) -> None:
    flashinfer_root = tmp_path / "flashinfer"
    flashinfer_root.mkdir()
    (flashinfer_root / "decode.py").write_text(
        "\n".join([
            "def flashinfer_api(func=None, *, trace=None):",
            "    def deco(f): return f",
            "    return deco(func) if func is not None else deco",
            "class BatchDecodeWithPagedKVCacheWrapper:",
            "    def forward(self, q):",
            "        return self.run(q)",
            "    @flashinfer_api(trace='gqa')",
            "    def run(self, q):",
            "        return q",
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")
    candidates = tmp_path / "candidate_targets.json"
    candidates.write_text(
        json.dumps([
            {
                "name": "decode_forward",
                "target": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.forward",
                "backend": "flashinfer",
                "definition_source": "fitrace",
                "collect": True,
            }
        ]),
        encoding="utf-8",
    )

    report = check_proposal(
        candidates_path=candidates,
        hf_config_path=hf_config,
        flashinfer_root=flashinfer_root,
    )

    assert report["summary"]["ok"] is False
    assert report["fitrace_eval"]["targets"][0]["suggested_target"] == (
        "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run"
    )
    assert any("suggested fitrace target" in item["reason"] for item in report["findings"])


def test_check_proposal_accepts_review_only_warmup_candidate(tmp_path: Path) -> None:
    candidates = tmp_path / "candidate_targets.json"
    candidates.write_text(
        json.dumps([
            {
                "name": "cuda_graph_capture_warmup",
                "role": "warmup",
                "status": "candidate",
                "target": None,
                "module": "sglang.srt.model_executor.cuda_graph_runner",
                "attr": "CudaGraphRunner.capture",
                "backend": "sglang_kernel",
                "definition_source": "manual",
                "collect": False,
                "op_type": "warmup",
            }
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    report = check_proposal(candidates_path=candidates, hf_config_path=hf_config)

    assert report["summary"]["ok"] is True
    assert report["summary"]["collect_candidates"] == 0
    assert report["summary"]["errors"] == 0


def test_agent_loop_writes_feedback_for_failed_candidate(tmp_path: Path) -> None:
    proposal_dir = tmp_path / "proposal"
    proposal_dir.mkdir()
    candidates = proposal_dir / "candidate_targets.json"
    candidates.write_text(
        json.dumps([
            {
                "name": "bad_collect",
                "target": "fake.missing.target",
                "backend": "flashinfer",
                "definition_source": "fitrace",
                "collect": True,
                "capture": _capture_json(),
            }
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    result = run_agent_loop(
        proposal_dir=proposal_dir,
        hf_config_path=hf_config,
    )

    assert result["summary"]["ok"] is False
    assert (proposal_dir / "proposal_check.json").exists()
    assert (proposal_dir / "agent_loop.json").exists()
    feedback = (proposal_dir / "agent_feedback.md").read_text(encoding="utf-8")
    assert "status: FIX_REQUIRED" in feedback
    assert "Do not edit config/approved_targets.json" in feedback
    assert "bad_collect" in feedback


def test_agent_loop_cli_passes_for_review_only_warmup_candidate(tmp_path: Path) -> None:
    proposal_dir = tmp_path / "proposal"
    proposal_dir.mkdir()
    (proposal_dir / "candidate_targets.json").write_text(
        json.dumps([
            {
                "name": "cuda_graph_capture_warmup",
                "role": "warmup",
                "status": "candidate",
                "module": "sglang.srt.model_executor.cuda_graph_runner",
                "attr": "CudaGraphRunner.capture",
                "backend": "sglang_kernel",
                "definition_source": "manual",
                "collect": False,
            }
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.proposal_tools",
            "agent-loop",
            "--proposal-dir",
            str(proposal_dir),
            "--hf-config",
            str(hf_config),
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )

    assert result.returncode == 0
    assert "ready for human review: True" in result.stdout
    loop_report = json.loads((proposal_dir / "agent_loop.json").read_text(encoding="utf-8"))
    assert loop_report["summary"]["ready_for_human_review"] is True
    assert "status: PASS" in (proposal_dir / "agent_feedback.md").read_text(encoding="utf-8")


def test_check_proposal_cli_passes_for_review_only_warmup_candidate(tmp_path: Path) -> None:
    proposal_dir = tmp_path / "proposal"
    proposal_dir.mkdir()
    (proposal_dir / "candidate_targets.json").write_text(
        json.dumps([
            {
                "name": "cuda_graph_capture_warmup",
                "role": "warmup",
                "status": "candidate",
                "module": "sglang.srt.model_executor.cuda_graph_runner",
                "attr": "CudaGraphRunner.capture",
                "backend": "sglang_kernel",
                "definition_source": "manual",
                "collect": False,
            }
        ]),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.proposal_tools",
            "check-proposal",
            "--proposal-dir",
            str(proposal_dir),
            "--hf-config",
            str(hf_config),
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )

    assert result.returncode == 0
    assert "proposal check ok: True" in result.stdout
    assert "ready for human review: True" in result.stdout
    assert (proposal_dir / "agent_loop.json").exists()
    assert "status: PASS" in (proposal_dir / "agent_feedback.md").read_text(encoding="utf-8")


def test_merge_proposals_unions_candidates_and_evidence(tmp_path: Path) -> None:
    proposal_a = tmp_path / "agent_a" / "proposal"
    proposal_b = tmp_path / "agent_b" / "proposal"
    proposal_a.mkdir(parents=True)
    proposal_b.mkdir(parents=True)
    base = {
        "name": "decode_a",
        "status": "candidate",
        "target": "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
        "module": "flashinfer.decode",
        "attr": "BatchDecodeWithPagedKVCacheWrapper.run",
        "backend": "flashinfer",
        "definition_source": "fitrace",
        "op_type": "gqa_paged",
        "variant": "decode",
        "collect": True,
        "companion_attrs": ["forward"],
        "capture": _capture_json(),
        "evidence": [{"kind": "source_location", "value": "a.py:1"}],
        "review_note": "agent a note",
    }
    same_from_b = dict(
        base,
        name="decode_b",
        evidence=[{"kind": "source_location", "value": "b.py:2"}],
        review_note="agent b note",
    )
    unique = {
        "name": "sampling",
        "status": "candidate",
        "target": "flashinfer.sampling.top_k_top_p_sampling_from_probs",
        "module": "flashinfer.sampling",
        "attr": "top_k_top_p_sampling_from_probs",
        "backend": "flashinfer",
        "definition_source": "fitrace",
        "op_type": "sampling",
        "variant": None,
        "collect": True,
        "capture": _capture_json(full_args=[0]),
    }
    (proposal_a / "candidate_targets.json").write_text(json.dumps([base]), encoding="utf-8")
    (proposal_b / "candidate_targets.json").write_text(json.dumps([same_from_b, unique]), encoding="utf-8")

    output = tmp_path / "merged" / "proposal"
    report = merge_proposals(proposal_dirs=[proposal_a, proposal_b], output_dir=output)

    assert report["summary"]["ok"] is True
    assert report["summary"]["input_candidates"] == 3
    assert report["summary"]["merged_candidates"] == 2
    merged = json.loads((output / "candidate_targets.json").read_text(encoding="utf-8"))
    decode = next(item for item in merged if item["op_type"] == "gqa_paged")
    assert decode["name"] == "decode_a"
    assert decode["evidence"] == [
        {"kind": "source_location", "value": "a.py:1"},
        {"kind": "source_location", "value": "b.py:2"},
    ]
    assert "agent a note" in decode["review_note"]
    assert "agent b note" in decode["review_note"]
    assert (output / "merge_review.md").exists()


def test_merge_proposals_reports_candidate_conflicts(tmp_path: Path) -> None:
    proposal_a = tmp_path / "agent_a" / "proposal"
    proposal_b = tmp_path / "agent_b" / "proposal"
    proposal_a.mkdir(parents=True)
    proposal_b.mkdir(parents=True)
    candidate = {
        "name": "rmsnorm",
        "status": "candidate",
        "target": "sglang.srt.layers.layernorm.RMSNorm.forward",
        "module": "sglang.srt.layers.layernorm",
        "attr": "RMSNorm.forward",
        "backend": "sglang_kernel",
        "definition_source": "agent",
        "definition_name": "rmsnorm_h2560",
        "op_type": "rmsnorm",
        "variant": None,
        "collect": True,
        "capture": _capture_json(full_args=[1]),
    }
    conflicting = dict(candidate, collect=False)
    (proposal_a / "candidate_targets.json").write_text(json.dumps([candidate]), encoding="utf-8")
    (proposal_b / "candidate_targets.json").write_text(json.dumps([conflicting]), encoding="utf-8")

    output = tmp_path / "merged" / "proposal"
    report = merge_proposals(proposal_dirs=[proposal_a, proposal_b], output_dir=output)

    assert report["summary"]["ok"] is False
    assert report["summary"]["candidate_conflicts"] == 1
    assert report["summary"]["merged_candidates"] == 0
    assert report["conflicts"][0]["fields"] == ["collect"]
    merged = json.loads((output / "candidate_targets.json").read_text(encoding="utf-8"))
    assert merged == []
    review = (output / "merge_review.md").read_text(encoding="utf-8")
    assert "conflicting candidate fields" in review


def test_merge_proposals_downgrades_description_only_draft_differences(tmp_path: Path) -> None:
    proposal_a = tmp_path / "agent_a" / "proposal"
    proposal_b = tmp_path / "agent_b" / "proposal"
    proposal_a.mkdir(parents=True)
    proposal_b.mkdir(parents=True)
    (proposal_a / "candidate_targets.json").write_text("[]", encoding="utf-8")
    (proposal_b / "candidate_targets.json").write_text("[]", encoding="utf-8")
    for proposal_dir, description in [(proposal_a, "agent a wording"), (proposal_b, "agent b wording")]:
        definition_dir = proposal_dir / "definitions" / "rmsnorm"
        definition_dir.mkdir(parents=True)
        (definition_dir / "rmsnorm_h2048.json").write_text(
            json.dumps({
                "name": "rmsnorm_h2048",
                "op_type": "rmsnorm",
                "description": description,
                "inputs": {
                    "input": {"shape": ["batch", 2048]},
                    "weight": {"shape": [2048]},
                    "eps": {"shape": None},
                },
            }),
            encoding="utf-8",
        )

    output = tmp_path / "merged" / "proposal"
    report = merge_proposals(proposal_dirs=[proposal_a, proposal_b], output_dir=output)

    assert report["summary"]["ok"] is True
    assert report["summary"]["draft_file_conflicts"] == 0
    assert report["summary"]["warnings"] == 1
    assert report["warnings"][0]["reason"] == "draft file descriptions differ"
    assert (output / "definitions" / "rmsnorm" / "rmsnorm_h2048.json").exists()
    review = (output / "merge_review.md").read_text(encoding="utf-8")
    assert "## Warnings" in review


def test_agent_loop_reports_unresolved_merge_conflicts(tmp_path: Path) -> None:
    proposal_dir = tmp_path / "proposal"
    proposal_dir.mkdir()
    (proposal_dir / "candidate_targets.json").write_text("[]", encoding="utf-8")
    (proposal_dir / "merge_report.json").write_text(
        json.dumps({
            "summary": {"conflicts": 1},
            "conflicts": [
                {
                    "kind": "candidate",
                    "key": ["target", "x"],
                    "reason": "conflicting candidate fields",
                    "fields": ["collect"],
                }
            ],
        }),
        encoding="utf-8",
    )
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")

    result = run_agent_loop(proposal_dir=proposal_dir, hf_config_path=hf_config)

    assert result["summary"]["ok"] is False
    feedback = (proposal_dir / "agent_feedback.md").read_text(encoding="utf-8")
    assert "merge conflicts: 1" in feedback
    assert "conflicting candidate fields; fields=['collect']" in feedback


def test_diagnose_run_requires_action_for_uncollected_definitions(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "model" / "run"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "run_report.json").write_text(
        json.dumps({
            "summary": {
                "accepted": True,
                "errors": 0,
                "warnings": 0,
            },
            "diagnostics": {
                "uncollected_definitions": [
                    {
                        "name": "extra_definition",
                        "reason": "audited definition was not selected by collect_plan",
                    }
                ]
            },
        }),
        encoding="utf-8",
    )

    result = diagnose_run(run=run_dir)

    assert result["summary"]["ok"] is False
    assert result["summary"]["action_required"] == 1
    feedback = (run_dir / "proposal" / "agent_feedback.md").read_text(encoding="utf-8")
    assert "status: FIX_REQUIRED" in feedback
    assert "action_required: 1" in feedback
    assert "extra_definition" in feedback


def test_repair_loop_preserves_run_diagnostics_feedback_without_agent(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "model" / "run"
    reports_dir = run_dir / "reports"
    proposal_dir = run_dir / "proposal"
    reports_dir.mkdir(parents=True)
    proposal_dir.mkdir()
    (proposal_dir / "candidate_targets.json").write_text("[]", encoding="utf-8")
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")
    (reports_dir / "run_report.json").write_text(
        json.dumps({
            "diagnostics": {
                "uncollected_definitions": [
                    {
                        "name": "extra_definition",
                        "reason": "audited definition was not selected by collect_plan",
                    }
                ]
            }
        }),
        encoding="utf-8",
    )

    result = repair_loop(run=run_dir, hf_config_path=hf_config)

    assert result["summary"]["diagnostics_ok"] is False
    assert result["summary"]["diagnostics_action_required"] == 1
    assert result["summary"]["agent_ran"] is False
    assert result["outputs"]["agent_loop"] is None
    feedback = (proposal_dir / "agent_feedback.md").read_text(encoding="utf-8")
    assert "status: FIX_REQUIRED" in feedback
    assert "extra_definition" in feedback
    repair_prompt = (proposal_dir / "repair_prompt.md").read_text(encoding="utf-8")
    assert "repair-pass mode" in repair_prompt
    assert "Do not restart first-pass onboarding" in repair_prompt


def test_repair_loop_runs_agent_loop_when_diagnostics_pass(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "model" / "run"
    reports_dir = run_dir / "reports"
    proposal_dir = run_dir / "proposal"
    reports_dir.mkdir(parents=True)
    proposal_dir.mkdir()
    (proposal_dir / "candidate_targets.json").write_text("[]", encoding="utf-8")
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")
    (reports_dir / "run_report.json").write_text(
        json.dumps({
            "summary": {
                "accepted": True,
                "errors": 0,
                "warnings": 0,
            }
        }),
        encoding="utf-8",
    )

    result = repair_loop(run=run_dir, hf_config_path=hf_config)

    assert result["summary"]["diagnostics_ok"] is True
    assert result["summary"]["ready_for_human_review"] is True
    assert result["summary"]["agent_ran"] is False
    assert result["outputs"]["agent_loop"] == str(proposal_dir / "agent_loop.json")
    assert (proposal_dir / "agent_loop.json").exists()


def test_repair_loop_passes_prompt_to_agent_stdin_and_rechecks(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "model" / "run"
    reports_dir = run_dir / "reports"
    proposal_dir = run_dir / "proposal"
    reports_dir.mkdir(parents=True)
    proposal_dir.mkdir()
    (proposal_dir / "candidate_targets.json").write_text("[]", encoding="utf-8")
    hf_config = tmp_path / "config.json"
    hf_config.write_text("{}", encoding="utf-8")
    (reports_dir / "run_report.json").write_text(
        json.dumps({
            "diagnostics": {
                "uncollected_definitions": [
                    {
                        "name": "extra_definition",
                        "reason": "audited definition was not selected by collect_plan",
                    }
                ]
            }
        }),
        encoding="utf-8",
    )
    agent_script = tmp_path / "fake_agent.py"
    agent_script.write_text(
        "\n".join([
            "import json",
            "import sys",
            "from pathlib import Path",
            "proposal_dir = Path(sys.argv[1])",
            "(proposal_dir / 'seen_prompt.md').write_text(sys.stdin.read(), encoding='utf-8')",
            "(proposal_dir / 'ignored_definitions.json').write_text(json.dumps([",
            "    {'name': 'extra_definition', 'reason': 'not part of reviewed workload surface'}",
            "]), encoding='utf-8')",
        ]),
        encoding="utf-8",
    )

    result = repair_loop(
        run=run_dir,
        hf_config_path=hf_config,
        agent_command=[sys.executable, str(agent_script), str(proposal_dir)],
        max_rounds=2,
    )

    assert result["summary"]["agent_ran"] is True
    assert result["summary"]["rounds"] == 2
    assert result["summary"]["diagnostics_ok"] is True
    assert result["summary"]["ready_for_human_review"] is True
    assert result["agent_rounds"][0]["returncode"] == 0
    seen_prompt = (proposal_dir / "seen_prompt.md").read_text(encoding="utf-8")
    assert "FIX_REQUIRED" in seen_prompt
    assert "extra_definition" in seen_prompt
    assert (proposal_dir / "agent_loop.json").exists()


def test_spawn_agents_generates_isolated_first_pass_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = spawn_agents(
        model_name="microsoft/Phi-4-mini-instruct",
        run_prefix=Path("phi_4_mini_instruct/first_pass"),
        hf_config_path=Path("agent_inputs/config/phi_4_mini_instruct.json"),
        sglang_root=Path("agent_inputs/sglang/python/sglang"),
        flashinfer_root=Path("agent_inputs/flashinfer/flashinfer"),
        cookbook_root=Path("agent_inputs/sgl-cookbook"),
        sglang_model_hints=["srt/models/phi.py"],
        count=3,
    )

    assert result["summary"]["count"] == 3
    assert result["summary"]["agents_started"] == 0
    run_dirs = [Path(item["run_dir"]) for item in result["prompts"]]
    assert run_dirs == [
        Path("runs/phi_4_mini_instruct/first_pass_agent_a"),
        Path("runs/phi_4_mini_instruct/first_pass_agent_b"),
        Path("runs/phi_4_mini_instruct/first_pass_agent_c"),
    ]
    for item in result["prompts"]:
        prompt = Path(item["prompt"])
        text = prompt.read_text(encoding="utf-8")
        assert "microsoft/Phi-4-mini-instruct" in text
        assert item["run_dir"] in text
        assert "Do not write `config/approved_targets.json`" in text


def test_first_pass_prompt_includes_phi_rope_startup_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "agent_inputs" / "config"
    config_dir.mkdir(parents=True)
    hf_config = {
        "model_type": "phi3",
        "architectures": ["Phi3ForCausalLM"],
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0],
            "long_factor": [1.0],
        },
    }
    (config_dir / "phi.json").write_text(json.dumps(hf_config), encoding="utf-8")

    result = spawn_agents(
        model_name="microsoft/Phi-4-mini-instruct",
        run_prefix=Path("phi/run"),
        hf_config_path=Path("agent_inputs/config/phi.json"),
        sglang_root=Path("agent_inputs/sglang/python/sglang"),
        flashinfer_root=Path("agent_inputs/flashinfer/flashinfer"),
        cookbook_root=Path("agent_inputs/sgl-cookbook"),
        sglang_model_hints=[],
    )

    prompt = Path(result["prompts"][0]["prompt"]).read_text(encoding="utf-8")
    assert "Runtime config compatibility requirement" in prompt
    assert "decrypted_config_json" in prompt
    assert "short_factor" in prompt
    assert "rope_theta" not in prompt


def test_agent_loop_requires_phi_rope_startup_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phi" / "run"
    proposal_dir = run_dir / "proposal"
    config_dir = run_dir / "config"
    proposal_dir.mkdir(parents=True)
    config_dir.mkdir()
    (proposal_dir / "candidate_targets.json").write_text("[]", encoding="utf-8")
    hf_config = {
        "model_type": "phi3",
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0],
            "long_factor": [1.0],
        },
    }
    hf_config_path = tmp_path / "phi_config.json"
    hf_config_path.write_text(json.dumps(hf_config), encoding="utf-8")

    missing = run_agent_loop(proposal_dir=proposal_dir, hf_config_path=hf_config_path)

    assert missing["summary"]["ready_for_human_review"] is False
    feedback = (proposal_dir / "agent_feedback.md").read_text(encoding="utf-8")
    assert "decrypted_config_json" in feedback

    required = _sglang_config_compat_engine_kwargs(hf_config)
    (config_dir / "run_config.json").write_text(
        json.dumps({"engine_kwargs": required}),
        encoding="utf-8",
    )

    ok = run_agent_loop(proposal_dir=proposal_dir, hf_config_path=hf_config_path)

    assert ok["summary"]["ready_for_human_review"] is True


def test_spawn_agents_cli_uses_model_defaults_without_manual_paths(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.proposal_tools",
            "spawn-agents",
            "--model",
            "microsoft/Phi-4-mini-instruct",
            "--count",
            "2",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )

    assert result.returncode == 0
    assert "prompts: 2" in result.stdout
    assert "agents started: 0" in result.stdout
    assert "merge report:" not in result.stdout
    slug = slug_model_name("microsoft/Phi-4-mini-instruct")
    assert (tmp_path / "runs" / slug).exists()
    assert list((tmp_path / "runs" / slug).glob("*_firstpass_agent_a/proposal/first_pass_prompt.md"))


def test_spawn_agents_runs_external_agents_with_prompt_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent_script = tmp_path / "fake_first_pass_agent.py"
    agent_script.write_text(
        "\n".join([
            "import json",
            "import sys",
            "from pathlib import Path",
            "prompt = sys.stdin.read()",
            "run_dir = None",
            "for line in prompt.splitlines():",
            "    if line.startswith('run dir: '):",
            "        run_dir = Path(line.split(': ', 1)[1])",
            "        break",
            "assert run_dir is not None",
            "proposal = run_dir / 'proposal'",
            "proposal.mkdir(parents=True, exist_ok=True)",
            "(proposal / 'seen_prompt.md').write_text(prompt, encoding='utf-8')",
            "(proposal / 'candidate_targets.json').write_text(json.dumps([]), encoding='utf-8')",
            "(proposal / 'architecture.md').write_text('# Architecture\\n', encoding='utf-8')",
            "(proposal / 'review_checklist.md').write_text('# Review\\n', encoding='utf-8')",
            "(run_dir / 'config').mkdir(parents=True, exist_ok=True)",
            "(run_dir / 'config' / 'run_config.json').write_text('{}', encoding='utf-8')",
        ]),
        encoding="utf-8",
    )

    result = spawn_agents(
        model_name="demo/model",
        run_prefix=Path("demo/parallel"),
        hf_config_path=Path("agent_inputs/config/demo.json"),
        sglang_root=Path("agent_inputs/sglang/python/sglang"),
        flashinfer_root=Path("agent_inputs/flashinfer/flashinfer"),
        cookbook_root=Path("agent_inputs/sgl-cookbook"),
        sglang_model_hints=[],
        count=2,
        agent_command=[sys.executable, str(agent_script)],
    )

    assert result["summary"]["agents_started"] == 2
    assert result["summary"]["agent_failures"] == 0
    for item in result["agent_results"]:
        proposal = Path(item["proposal_dir"])
        assert item["returncode"] == 0
        assert (proposal / "seen_prompt.md").exists()
        assert (proposal / "agent_stdout.log").exists()
        assert (proposal / "agent_stderr.log").exists()


def test_diagnose_run_accepts_reviewed_ignored_uncollected_definitions(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "model" / "run"
    reports_dir = run_dir / "reports"
    proposal_dir = run_dir / "proposal"
    reports_dir.mkdir(parents=True)
    proposal_dir.mkdir()
    (reports_dir / "run_report.json").write_text(
        json.dumps({
            "diagnostics": {
                "uncollected_definitions": [
                    {
                        "name": "merge_state_h8_d256",
                        "reason": "audited definition was not selected by collect_plan",
                    }
                ]
            },
        }),
        encoding="utf-8",
    )
    (proposal_dir / "ignored_definitions.json").write_text(
        json.dumps([
            {
                "name": "merge_state_h8_d256",
                "reason": "cascade merge helper, not part of reviewed workload surface",
            }
        ]),
        encoding="utf-8",
    )

    result = diagnose_run(run=run_dir)

    assert result["summary"]["ok"] is True
    assert result["summary"]["action_required"] == 0
    diagnostics = json.loads((proposal_dir / "run_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["ignored_uncollected_definitions"] == [
        {
            "name": "merge_state_h8_d256",
            "reason": "cascade merge helper, not part of reviewed workload surface",
        }
    ]
    feedback = (proposal_dir / "agent_feedback.md").read_text(encoding="utf-8")
    assert "status: PASS" in feedback
    assert "Ignored Uncollected Definitions" in feedback
    assert "merge_state_h8_d256" in feedback


def test_diagnose_run_accepts_proposed_uncollected_definitions(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "model" / "run"
    reports_dir = run_dir / "reports"
    proposal_dir = run_dir / "proposal"
    reports_dir.mkdir(parents=True)
    proposal_dir.mkdir()
    (reports_dir / "run_report.json").write_text(
        json.dumps({
            "diagnostics": {
                "uncollected_definitions": [
                    {
                        "name": "gemma_rmsnorm_h1024",
                        "reason": "audited definition was not selected by collect_plan",
                    }
                ]
            },
        }),
        encoding="utf-8",
    )
    (proposal_dir / "candidate_targets.json").write_text(
        json.dumps([
            {
                "name": "qwen35_gemma_rmsnorm_h1024",
                "status": "candidate",
                "definition_name": "gemma_rmsnorm_h1024",
                "collect": True,
            }
        ]),
        encoding="utf-8",
    )

    result = diagnose_run(run=run_dir)

    assert result["summary"]["ok"] is True
    assert result["summary"]["action_required"] == 0
    diagnostics = json.loads((proposal_dir / "run_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["proposed_uncollected_definitions"] == [
        {
            "name": "gemma_rmsnorm_h1024",
            "candidate": "qwen35_gemma_rmsnorm_h1024",
        }
    ]
    feedback = (proposal_dir / "agent_feedback.md").read_text(encoding="utf-8")
    assert "status: PASS" in feedback
    assert "Proposed Uncollected Definitions" in feedback
    assert "gemma_rmsnorm_h1024" in feedback


def test_validate_run_writes_reviewable_summary(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    _write_definition(definitions_dir, "approved_target")
    approved_targets = tmp_path / "approved_targets.json"
    approved_targets.write_text(
        json.dumps([
                {
                    "name": "approved_target",
                    "target": "flashinfer.norm.rmsnorm",
                    "module": "flashinfer.norm",
                    "attr": "rmsnorm",
                    "collect": True,
                    "capture": _capture_json(),
                    }
        ]),
        encoding="utf-8",
    )
    parse_report = tmp_path / "parse_report.json"
    parse_report.write_text(
        json.dumps({
            "summary": {"events": 1, "non_warmup_events": 1, "missing_targets": 0},
            "observed_targets": [{"name": "approved_target"}],
            "missing_targets": [],
        }),
        encoding="utf-8",
    )
    manifest = tmp_path / "workload_manifest.json"
    workload_path = tmp_path / "workloads" / "rmsnorm" / "approved_target.jsonl"
    blob_path = tmp_path / "blob" / "workloads" / "rmsnorm" / "approved_target" / "000001.pt"
    workload_path.parent.mkdir(parents=True)
    blob_path.parent.mkdir(parents=True)
    workload_path.write_text("{}\n", encoding="utf-8")
    blob_path.write_bytes(b"capture")
    manifest.write_text(
        json.dumps({
            "summary": {"workloads": 1, "captures": 1, "workload_files": 2, "sanitized": 1},
            "workloads": [
                {
                    "name": "approved_target",
                    "definition_name": "approved_target",
                    "capture_paths": [str(blob_path)],
                    "workload_paths": [str(workload_path)],
                    "blob_paths": [str(blob_path)],
                    "sanitized_count": 1,
                }
            ],
        }),
        encoding="utf-8",
    )

    report = validate_run(
        approved_targets_path=approved_targets,
        definitions_dir=definitions_dir,
        parse_report_path=parse_report,
        workload_manifest_path=manifest,
    )
    markdown = render_run_review_markdown({
        "collect": {"manifest": {"summary": report["workload_summary"]}},
        "internal_validation": report,
    })

    assert report["summary"]["ok"] is True
    assert "sanitized entries: 1" in markdown


def test_validate_run_accepts_fitrace_dump_name_over_preview(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    definition_path = _write_definition(
        definitions_dir,
        "top_k_top_p_sampling_v128256",
        op_type="sampling",
    )
    payload = json.loads(definition_path.read_text(encoding="utf-8"))
    payload["tags"] = ["fi_api:flashinfer.sampling.top_k_top_p_sampling_from_probs", "status:verified"]
    definition_path.write_text(json.dumps(payload), encoding="utf-8")
    approved_targets = tmp_path / "approved_targets.json"
    approved_targets.write_text(
        json.dumps([
            {
                "name": "sampling",
                "target": "flashinfer.sampling.top_k_top_p_sampling_from_probs",
                "module": "flashinfer.sampling",
                "attr": "top_k_top_p_sampling_from_probs",
                "definition_source": "fitrace",
                "backend": "flashinfer",
                "collect": True,
                "definition_name": "top_k_top_p_sampling_from_probs_v128256",
                "op_type": "sampling",
                "capture": _capture_json(full_args=[0, 2]),
            }
        ]),
        encoding="utf-8",
    )

    report = validate_run(
        approved_targets_path=approved_targets,
        definitions_dir=definitions_dir,
    )

    assert report["summary"]["ok"] is True
    assert report["findings"] == []


def test_run_review_markdown_lists_repaired_definitions() -> None:
    markdown = render_run_review_markdown(
        {
            "summary": {"accepted": True, "internal_ok": True, "official_ok": True, "export_ok": True},
            "collect": {"manifest": {"summary": {"workloads": 1, "captures": 2, "workload_files": 3, "sanitized": 2}}},
            "definition_audit": {
                "summary": {"raw": 2, "passed": 1, "repaired": 1, "rejected": 0},
                "repaired": [
                    {
                        "source_name": "gqa_paged_decode_h32_kv128_d128_ps8",
                        "name": "gqa_paged_decode_h32_kv8_d128_ps1",
                        "repair": "gqa_paged_3d_cache_axes",
                        "reason": "gqa_paged num_kv_heads larger than num_qo_heads: 128>32",
                    }
                ],
            },
            "export": {"summary": {"definitions": 1, "missing_definitions": 0}},
            "internal_validation": {"summary": {"errors": 0, "warnings": 0}, "findings": []},
            "official_validation": {"ok": True, "returncode": 0},
        }
    )

    assert "### Repaired Definitions" in markdown
    assert "gqa_paged_decode_h32_kv128_d128_ps8 -> gqa_paged_decode_h32_kv8_d128_ps1" in markdown
    assert "gqa_paged_3d_cache_axes" in markdown


def test_export_run_dataset_copies_official_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    definition = tmp_path / "defs" / "rmsnorm" / "demo.json"
    definition.parent.mkdir(parents=True)
    definition.write_text(json.dumps({"name": "demo", "op_type": "rmsnorm"}), encoding="utf-8")
    (run_dir / "output" / "workloads" / "rmsnorm").mkdir(parents=True)
    (run_dir / "output" / "workloads" / "rmsnorm" / "demo.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "output" / "blob" / "workloads" / "rmsnorm" / "demo").mkdir(parents=True)
    (run_dir / "output" / "blob" / "workloads" / "rmsnorm" / "demo" / "x.safetensors").write_bytes(b"x")
    _write_collect_report(run_dir, plan={"targets": [{"definition_name": "demo", "op_type": "rmsnorm", "definition_path": str(definition)}]})

    report = export_run_dataset(run_dir=run_dir)
    dataset_dir = Path(report["dataset_dir"])

    assert report["summary"]["ok"] is True
    assert dataset_dir == run_dir / "output"
    assert (dataset_dir / "definitions" / "rmsnorm" / "demo.json").exists()
    assert (dataset_dir / "workloads" / "rmsnorm" / "demo.jsonl").exists()
    assert (dataset_dir / "blob" / "workloads" / "rmsnorm" / "demo" / "x.safetensors").exists()


def test_validate_cli_runs_internal_and_official_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fake_bench = tmp_path / "flashinfer_bench" / "cli"
    fake_bench.mkdir(parents=True)
    (tmp_path / "flashinfer_bench" / "__init__.py").write_text("", encoding="utf-8")
    (fake_bench / "__init__.py").write_text("", encoding="utf-8")
    (fake_bench / "main.py").write_text(
        "def cli():\n"
        "    print('flashinfer-bench fake validator ok')\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "demo"
    definitions_dir = run_dir / "output" / "definitions"
    definition = _write_definition(definitions_dir, "demo")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_inputs(
        run_dir,
        config={},
        approved=[
                {
                    "name": "demo",
                    "target": "flashinfer.norm.rmsnorm",
                    "module": "flashinfer.norm",
                    "attr": "rmsnorm",
                    "definition_name": "demo",
                    "collect": True,
                    }
        ],
    )
    update_run_report(
        run_dir,
        parse_report={
            "summary": {"events": 1, "non_warmup_events": 1, "missing_targets": 0},
            "observed_targets": [{"name": "demo"}],
            "missing_targets": [],
        },
    )
    workload_path = run_dir / "output" / "workloads" / "rmsnorm" / "demo.jsonl"
    blob_path = run_dir / "output" / "blob" / "workloads" / "rmsnorm" / "demo" / "000001.safetensors"
    workload_path.parent.mkdir(parents=True)
    blob_path.parent.mkdir(parents=True)
    workload_path.write_text("{}\n", encoding="utf-8")
    blob_path.write_bytes(b"blob")
    _write_collect_report(
        run_dir,
        plan={"targets": [{"definition_name": "demo", "op_type": "rmsnorm", "definition_path": str(definition)}]},
        manifest={
            "summary": {"workloads": 1, "captures": 1, "workload_files": 2, "sanitized": 1},
            "workloads": [
                {
                    "name": "demo",
                    "definition_name": "demo",
                    "capture_paths": [str(blob_path)],
                    "workload_paths": [str(workload_path)],
                    "blob_paths": [str(blob_path)],
                    "sanitized_count": 1,
                }
            ],
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashinfer_trace.cli",
            "validate",
            "--run",
            "demo",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[1])]),
        },
    )

    assert "internal validation ok: True" in result.stdout
    assert "official validation ok: True" in result.stdout
    assert "run accepted: True" in result.stdout
    assert "run report: runs/demo/reports/run_report.json" in result.stdout
    run_report = json.loads((run_dir / "reports" / "run_report.json").read_text(encoding="utf-8"))
    assert run_report["summary"]["accepted"] is True
    assert run_report["collect"]["manifest"]["summary"]["workloads"] == 1
    assert run_report["internal_validation"]["summary"]["ok"] is True
    assert run_report["official_validation"]["ok"] is True
    assert run_report["official_validation"]["returncode"] == 0
    assert (run_dir / "reports" / "review.md").exists()
    assert not (run_dir / "validate" / "validation_report.json").exists()
    assert (run_dir / "output" / "definitions" / "rmsnorm" / "demo.json").exists()
    assert not (run_dir / "official_validate" / "official_validation_report.json").exists()


def test_build_probe_plan_routes_warmup_role_into_warmup_hooks() -> None:
    approved = [
        ApprovedTarget(
            name="decode",
            target="pkg.mod.kernel",
            module="pkg.mod",
            attr="kernel",
            backend="flashinfer",
            collect=True,
            op_type="gqa_paged",
            capture=_capture_spec(),
        ),
        ApprovedTarget(
            name="sglang_warmup",
            role="warmup",
            module="pkg.mod",
            attr="warmup",
        ),
    ]

    plan = build_probe_plan(approved)

    assert [t.name for t in plan.targets] == ["decode"]
    assert [h.name for h in plan.warmup_hooks] == ["sglang_warmup"]
    assert plan.warmup_hooks[0].module == "pkg.mod"
    assert plan.warmup_hooks[0].attr == "warmup"

    # Round-trips through the modal probe-plan payload.
    rebuilt = ProbePlan.from_jsonable(plan.to_jsonable())
    assert [h.attr for h in rebuilt.warmup_hooks] == ["warmup"]


def test_capture_session_tags_events_inside_warmup_window(tmp_path: Path) -> None:
    module_dir = tmp_path / "warmup_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "mod.py").write_text(
        "def kernel(x):\n"
        "    return x\n"
        "\n"
        "def warmup():\n"
        "    return kernel(1)\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        import importlib

        mod = importlib.import_module("warmup_pkg.mod")

        plan = ProbePlan(
            targets=[
                ProbeTarget(
                    name="kernel",
                    target="warmup_pkg.mod.kernel",
                    module="warmup_pkg.mod",
                    attr="kernel",
                    op_type="gqa_paged",
                    capture=_capture_spec(),
                )
            ],
            skipped=[],
            warmup_hooks=[WarmupHook(name="warmup", module="warmup_pkg.mod", attr="warmup")],
        )

        output_dir = tmp_path / "probe_out"
        session = CaptureSession(probe_plan=plan, output_dir=output_dir, max_captures_per_target=4)
        session.install()
        try:
            mod.warmup()  # kernel called inside the warmup window
            mod.kernel(2)  # kernel called as a real request
        finally:
            session.uninstall()

        events = [
            json.loads(line)
            for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        warmup_flags = [event["is_warmup"] for event in events]
        assert warmup_flags == [True, False]
        # Warmup window fully unwound after the run.
        assert session.is_warmup is False
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("warmup_pkg.mod", None)
        sys.modules.pop("warmup_pkg", None)


def test_companion_capture_merges_forward_args_into_run_payload(tmp_path: Path) -> None:
    import torch

    module_dir = tmp_path / "companion_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    # `forward` is the deprecated alias: it stashes sm_scale and calls run(),
    # mirroring how FlashInfer attention wrappers behave.
    (module_dir / "mod.py").write_text(
        "class Wrapper:\n"
        "    def forward(self, q, sm_scale=None):\n"
        "        self._sm_scale = sm_scale\n"
        "        return self.run(q)\n"
        "\n"
        "    def run(self, q):\n"
        "        return q\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        import importlib

        mod = importlib.import_module("companion_pkg.mod")

        plan = ProbePlan(
            targets=[
                ProbeTarget(
                    name="decode",
                    target="companion_pkg.mod.Wrapper.run",
                    module="companion_pkg.mod",
                    attr="Wrapper.run",
                    op_type="gqa_paged",
                    companion_attrs=["forward"],
                    capture=_capture_spec(full_kwargs=["sm_scale"]),
                )
            ],
            skipped=[],
        )

        output_dir = tmp_path / "probe_out"
        session = CaptureSession(probe_plan=plan, output_dir=output_dir, max_captures_per_target=4)
        session.install()
        try:
            wrapper = mod.Wrapper()
            # SGLang passes sm_scale to forward, not run.
            wrapper.forward("dummy_q", sm_scale=0.125)
        finally:
            session.uninstall()

        # The companion hook installed and the run capture recovered sm_scale.
        companion_status = [s for s in session.install_status if s.get("kind") == "companion"]
        assert companion_status and companion_status[0]["installed"] is True

        capture_files = sorted((output_dir / "captures").glob("*.pt"))
        assert len(capture_files) == 1
        payload = torch.load(capture_files[0], weights_only=False)
        kwargs = payload["payload"]["kwargs"]
        assert kwargs["sm_scale"]["value"] == 0.125
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("companion_pkg.mod", None)
        sys.modules.pop("companion_pkg", None)


def test_probe_plan_round_trips_companion_attrs() -> None:
    plan = ProbePlan(
        targets=[
            ProbeTarget(
                name="decode",
                target="flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
                module="flashinfer.decode",
                attr="BatchDecodeWithPagedKVCacheWrapper.run",
                op_type="gqa_paged",
                companion_attrs=["forward", "begin_forward"],
                capture=_capture_spec(),
            )
        ],
        skipped=[],
    )

    rebuilt = ProbePlan.from_jsonable(plan.to_jsonable())

    assert rebuilt.targets[0].companion_attrs == ["forward", "begin_forward"]
