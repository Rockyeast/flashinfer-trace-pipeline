"""Modal probe runner planning and execution."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from flashinfer_trace.core.capture import (
    CAPTURE_SCOPE_FILENAME,
    HF_CONFIG_OVERRIDE_ENV,
    CaptureSession,
    install_hf_config_override,
)
from flashinfer_trace.core.definition_audit import audit_and_repair_definitions, prepare_definition_for_output
from flashinfer_trace.core.events import build_event_report, load_jsonl
from flashinfer_trace.core.planning import (
    build_collect_plan_from_probe_plan,
    load_definitions,
)
from flashinfer_trace.core.schemas import CollectPlan, ProbePlan, max_captures_from_plan_payload

DEFAULT_REMOTE_OUTPUT_DIR = "/tmp/flashinfer-trace-probe"
DEFAULT_WORKER_SITE_DIR = "/tmp/flashinfer-trace-site"
FITRACE_DUMP_ENV = "FLASHINFER_TRACE_DUMP"
FITRACE_DUMP_DIR_ENV = "FLASHINFER_TRACE_DUMP_DIR"


def _supported_engine_kwarg_names() -> set[str]:
    try:
        import inspect
        from sglang.srt.server_args import ServerArgs

        return set(inspect.signature(ServerArgs).parameters)
    except Exception as exc:  # noqa: BLE001 - surface SGLang environment issues clearly
        raise RuntimeError(f"could not inspect SGLang ServerArgs: {exc}") from exc


def _filter_supported_engine_kwargs(
    engine_kwargs: dict[str, Any],
    *,
    optional_kwargs: set[str] | None = None,
) -> dict[str, Any]:
    """Reject unsupported reviewed kwargs and drop unsupported optional runner knobs."""
    supported = _supported_engine_kwarg_names()
    optional = optional_kwargs or set()
    unsupported = sorted(key for key in engine_kwargs if key not in supported)
    required_unsupported = sorted(key for key in unsupported if key not in optional)
    if required_unsupported:
        raise RuntimeError(f"SGLang Engine does not support kwargs: {required_unsupported}")
    return {key: value for key, value in engine_kwargs.items() if key not in unsupported}


def _pop_decrypted_config_json(engine_kwargs: dict[str, Any]) -> dict[str, Any] | None:
    raw = engine_kwargs.pop("decrypted_config_json", None)
    if raw is None:
        return None
    if "decrypted_config_file" in engine_kwargs:
        raise RuntimeError("engine_kwargs cannot set both decrypted_config_json and decrypted_config_file")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("engine_kwargs.decrypted_config_json must be a non-empty JSON string")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"engine_kwargs.decrypted_config_json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("engine_kwargs.decrypted_config_json must decode to a JSON object")
    return data


def _shutdown_engine(engine: Any) -> None:
    for method_name in ("shutdown", "release", "close"):
        method = getattr(engine, method_name, None)
        if callable(method):
            method()
            return


def _write_worker_sitecustomize(site_dir: Path) -> None:
    """Write a startup hook imported automatically by child Python workers."""
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "sitecustomize.py").write_text(
        "\n".join(
            [
                "import importlib",
                "import sys",
                "try:",
                "    capture = importlib.import_module('flashinfer_trace.core.capture')",
                "    capture.install_hf_config_override_from_env()",
                "    capture.install_from_env()",
                "except Exception as exc:",
                "    print(f'[flashinfer_trace sitecustomize] failed: {exc}', file=sys.stderr, flush=True)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _prepend_pythonpath(path: Path) -> None:
    current = os.environ.get("PYTHONPATH")
    parts = [str(path)]
    if current:
        parts.append(current)
    os.environ["PYTHONPATH"] = ":".join(parts)


def prepare_worker_injection(
    *,
    modal_probe_plan: dict[str, Any],
    output_dir: Path,
    site_dir: Path,
) -> Path:
    """Prepare the sitecustomize-based hook injection for child workers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "events.jsonl").write_text("", encoding="utf-8")
    (output_dir / CAPTURE_SCOPE_FILENAME).write_text("{}\n", encoding="utf-8")
    shutil.rmtree(output_dir / "captures", ignore_errors=True)
    shutil.rmtree(output_dir / "hook_status", ignore_errors=True)

    plan_path = output_dir / "worker_probe_plan.json"
    plan_path.write_text(json.dumps(modal_probe_plan, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_worker_sitecustomize(site_dir)
    _prepend_pythonpath(site_dir)
    os.environ["FLASHINFER_TRACE_PROBE_PLAN"] = str(plan_path)
    os.environ["FLASHINFER_TRACE_OUTPUT_DIR"] = str(output_dir)
    return plan_path


def _prepare_fitrace_dump(output_dir: Path) -> Path:
    """Enable FlashInfer fitrace definition dumping for this remote run."""
    definitions_dir = output_dir / "definitions"
    shutil.rmtree(definitions_dir, ignore_errors=True)
    definitions_dir.mkdir(parents=True, exist_ok=True)
    os.environ[FITRACE_DUMP_ENV] = "1"
    os.environ[FITRACE_DUMP_DIR_ENV] = str(definitions_dir)
    print(f"[flashinfer_trace] fitrace dump enabled: {definitions_dir}", flush=True)
    return definitions_dir


def _paged_probe_page_sizes(modal_probe_plan: dict[str, Any]) -> list[int | None]:
    """Return approved page_size values requested by paged targets."""
    raw_targets = (
        modal_probe_plan.get("probe_plan", {}).get("targets", [])
        if isinstance(modal_probe_plan.get("probe_plan"), dict)
        else []
    )
    sizes = sorted({
        int(target["page_size"])
        for target in raw_targets
        if (
            isinstance(target, dict)
            and target.get("probe_mode") in {"paged", "both"}
            and isinstance(target.get("page_size"), int)
            and int(target["page_size"]) > 0
        )
    })
    return sizes or [None]


def _probe_passes(modal_probe_plan: dict[str, Any]) -> list[tuple[bool, int | None]]:
    """Return ordered SGLang passes as ``(use_paged_prefill, page_size)``."""
    modes = set(modal_probe_plan.get("probe_modes") or [])
    paged_passes = [(True, size) for size in _paged_probe_page_sizes(modal_probe_plan)]
    if "both" in modes:
        return [(False, None), *paged_passes]
    passes: list[tuple[bool, int | None]] = []
    if "default" in modes or not modes:
        passes.append((False, None))
    if "paged" in modes:
        passes.extend(paged_passes)
    return passes or [(False, None)]


def _default_param_config(modal_probe_plan: dict[str, Any]) -> dict[str, Any]:
    """Return the normal one-shot generation params for non-specialized probe."""
    sampling = modal_probe_plan.get("sampling", {})
    if not isinstance(sampling, dict) or "max_new_tokens" not in sampling:
        raise ValueError("modal probe plan missing sampling.max_new_tokens")
    max_new_tokens = int(sampling["max_new_tokens"])
    return {"max_new_tokens": max(1, max_new_tokens)}


def _supplemental_runs(modal_probe_plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw_runs = modal_probe_plan.get("supplemental_runs")
    if not isinstance(raw_runs, list):
        raise ValueError("modal probe plan missing supplemental_runs")
    runs: list[dict[str, Any]] = []
    for index, item in enumerate(raw_runs):
        if not isinstance(item, dict):
            raise ValueError(f"supplemental_runs[{index}] must be an object")
        name = item.get("name")
        params = item.get("sampling_params")
        if not isinstance(name, str) or not name:
            raise ValueError(f"supplemental_runs[{index}].name must be a non-empty string")
        if not isinstance(params, dict):
            raise ValueError(f"supplemental_runs[{index}].sampling_params must be an object")
        allowed = item.get("allowed_op_types")
        if allowed is not None and not (
            isinstance(allowed, list) and all(isinstance(value, str) for value in allowed)
        ):
            raise ValueError(f"supplemental_runs[{index}].allowed_op_types must be a list of strings")
        use_scenario_tokens = item.get("use_scenario_tokens", False)
        if type(use_scenario_tokens) is not bool:
            raise ValueError(f"supplemental_runs[{index}].use_scenario_tokens must be a boolean")
        runs.append({
            "name": name,
            "sampling_params": dict(params),
            "allowed_op_types": list(allowed) if isinstance(allowed, list) else None,
            "use_scenario_tokens": use_scenario_tokens,
        })
    return runs


def _prompt_scenarios(modal_probe_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return prompt scenarios with per-scenario generation lengths."""
    raw_scenarios = modal_probe_plan.get("prompt_scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("modal probe plan missing prompt_scenarios")
    scenarios: list[dict[str, Any]] = []
    for index, scenario in enumerate(raw_scenarios, 1):
        if not isinstance(scenario, dict):
            continue
        prompts = scenario.get("prompts")
        if not isinstance(prompts, list):
            continue
        batch = [str(item) for item in prompts if str(item)]
        if not batch:
            continue
        if "max_new_tokens" not in scenario:
            raise ValueError(f"prompt_scenarios[{index}] missing max_new_tokens")
        max_new_tokens = int(scenario["max_new_tokens"])
        scenarios.append(
            {
                "name": str(scenario.get("name") or f"scenario_{index}"),
                "prompts": batch,
                "max_new_tokens": max(1, max_new_tokens),
            }
        )
    if not scenarios:
        raise ValueError("modal probe plan has no valid prompt_scenarios")
    return scenarios


def _write_capture_scope(
    *,
    output_dir: Path,
    name: str,
    allowed_op_types: list[str] | None = None,
) -> None:
    """Write the capture scope consumed by parent and worker hooks."""
    payload: dict[str, Any] = {"name": name}
    if allowed_op_types:
        payload["allowed_op_types"] = allowed_op_types
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / CAPTURE_SCOPE_FILENAME).write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_generation_requests(
    runner: Any,
    modal_probe_plan: dict[str, Any],
    *,
    output_dir: Path,
    early_check: Callable[[], bool] | None = None,
) -> bool:
    """Run prompt/sampling scenarios against an Engine-like runner."""
    prompt_scenarios = _prompt_scenarios(modal_probe_plan)
    _write_capture_scope(output_dir=output_dir, name="base")
    all_params: list[tuple[dict[str, Any], str, list[str] | None, bool]] = [
        (_default_param_config(modal_probe_plan), "base", None, True)
    ]
    all_params.extend(
        (
            run["sampling_params"],
            run["name"],
            run["allowed_op_types"],
            bool(run["use_scenario_tokens"]),
        )
        for run in _supplemental_runs(modal_probe_plan)
    )
    for params, scope_name, allowed_op_types, use_scenario_tokens in all_params:
        _write_capture_scope(
            output_dir=output_dir,
            name=scope_name,
            allowed_op_types=allowed_op_types,
        )
        for scenario_index, scenario in enumerate(prompt_scenarios):
            prompts = scenario["prompts"]
            run_params = dict(params)
            if use_scenario_tokens:
                run_params["max_new_tokens"] = int(scenario["max_new_tokens"])
            else:
                run_params.setdefault("max_new_tokens", int(scenario["max_new_tokens"]))
            prompt_input: str | list[str] = prompts[0] if len(prompts) == 1 else prompts
            try:
                runner.generate(prompt_input, sampling_params=run_params)
            except TypeError:
                runner.generate(prompt_input, max_new_tokens=int(run_params["max_new_tokens"]))
            if scope_name == "base" and scenario_index == 0 and early_check is not None and early_check():
                return False
    return True


def _run_sglang_pass(
    modal_probe_plan: dict[str, Any],
    *,
    paged: bool,
    page_size: int | None = None,
    early_check: Callable[[], bool] | None = None,
) -> bool:
    """Run one SGLang prompt pass using one prefill mode."""
    runtime = modal_probe_plan.get("runtime", {})
    model_name = str(modal_probe_plan.get("model_name") or runtime.get("model_name"))
    tp_size = int(runtime.get("tp_size") or 1)
    output_dir = Path(os.environ.get("FLASHINFER_TRACE_OUTPUT_DIR") or DEFAULT_REMOTE_OUTPUT_DIR)
    sglang_config = modal_probe_plan.get("sglang") if isinstance(modal_probe_plan.get("sglang"), dict) else {}
    disable_cuda_graph = bool(sglang_config.get("disable_cuda_graph", True))
    configured_piecewise = sglang_config.get("enable_piecewise_cuda_graph")
    enable_piecewise_cuda_graph = bool(paged if configured_piecewise is None else configured_piecewise)
    force_flashinfer_backends = bool(sglang_config.get("force_flashinfer_backends", True))
    mem_fraction_static = sglang_config.get("mem_fraction_static")
    cuda_graph_max_bs = sglang_config.get("cuda_graph_max_bs")
    reviewed_engine_kwargs = sglang_config.get("engine_kwargs")
    if reviewed_engine_kwargs is None:
        reviewed_engine_kwargs = {}
    if not isinstance(reviewed_engine_kwargs, dict):
        raise RuntimeError("sglang.engine_kwargs must be an object")

    os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")
    old_paged = os.environ.get("SGLANG_FLASHINFER_USE_PAGED")
    old_probe_mode = os.environ.get("FLASHINFER_TRACE_ACTIVE_PROBE_MODE")
    if paged:
        os.environ["SGLANG_FLASHINFER_USE_PAGED"] = "1"
        os.environ["FLASHINFER_TRACE_ACTIVE_PROBE_MODE"] = (
            f"paged_ps{page_size}" if page_size is not None else "paged"
        )
    else:
        os.environ.pop("SGLANG_FLASHINFER_USE_PAGED", None)
        os.environ["FLASHINFER_TRACE_ACTIVE_PROBE_MODE"] = "default"
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    try:
        engine_kwargs = {
            "model_path": model_name,
            "tp_size": tp_size,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "attention_backend": "flashinfer",
            "disable_cuda_graph": disable_cuda_graph,
            "disable_piecewise_cuda_graph": not enable_piecewise_cuda_graph,
            "log_level": "info",
        }
        if force_flashinfer_backends:
            engine_kwargs.update(
                {
                    "prefill_attention_backend": "flashinfer",
                    "decode_attention_backend": "flashinfer",
                    "sampling_backend": "flashinfer",
                }
            )
        if enable_piecewise_cuda_graph:
            engine_kwargs["enable_deterministic_inference"] = True
        if paged and page_size is not None:
            engine_kwargs["page_size"] = page_size
        if isinstance(cuda_graph_max_bs, int) and cuda_graph_max_bs >= 0:
            engine_kwargs["cuda_graph_max_bs"] = cuda_graph_max_bs
        if isinstance(mem_fraction_static, (int, float)) and float(mem_fraction_static) >= 0:
            engine_kwargs["mem_fraction_static"] = float(mem_fraction_static)
        protected_kwargs = {
            "model_path",
            "tp_size",
            "trust_remote_code",
            "disable_cuda_graph",
            "disable_piecewise_cuda_graph",
            "page_size",
        }
        protected = sorted(key for key in reviewed_engine_kwargs if key in protected_kwargs)
        if protected:
            raise RuntimeError(f"engine_kwargs cannot override managed kwargs: {protected}")
        hf_config_override = _pop_decrypted_config_json(reviewed_engine_kwargs)
        if hf_config_override is not None:
            os.environ[HF_CONFIG_OVERRIDE_ENV] = json.dumps(hf_config_override, separators=(",", ":"))
        reviewed_engine_kwargs = _filter_supported_engine_kwargs(reviewed_engine_kwargs)
        engine_kwargs.update(reviewed_engine_kwargs)
        engine_kwargs = _filter_supported_engine_kwargs(
            engine_kwargs,
            optional_kwargs={
                "disable_piecewise_cuda_graph",
                "enable_deterministic_inference",
                "cuda_graph_max_bs",
                "mem_fraction_static",
            },
        )
        pass_record = {
            "mode": "paged" if paged else "default",
            "requested_page_size": page_size,
            "runner": "engine",
            "engine_kwargs": engine_kwargs,
            "enable_piecewise_cuda_graph": enable_piecewise_cuda_graph,
            "force_flashinfer_backends": force_flashinfer_backends,
        }
        print(f"[flashinfer_trace] sglang pass: {json.dumps(pass_record, sort_keys=True)}", flush=True)
        import sglang as sgl

        install_hf_config_override(hf_config_override)
        engine = sgl.Engine(**engine_kwargs)
        try:
            return _run_generation_requests(
                engine,
                modal_probe_plan,
                output_dir=output_dir,
                early_check=early_check,
            )
        finally:
            _write_capture_scope(output_dir=output_dir, name="complete")
            _shutdown_engine(engine)
    finally:
        if old_paged is None:
            os.environ.pop("SGLANG_FLASHINFER_USE_PAGED", None)
        else:
            os.environ["SGLANG_FLASHINFER_USE_PAGED"] = old_paged
        if old_probe_mode is None:
            os.environ.pop("FLASHINFER_TRACE_ACTIVE_PROBE_MODE", None)
        else:
            os.environ["FLASHINFER_TRACE_ACTIVE_PROBE_MODE"] = old_probe_mode


def run_sglang_model(
    modal_probe_plan: dict[str, Any],
    *,
    early_check: Callable[[], bool] | None = None,
) -> bool:
    """Run SGLang prompt passes requested by probe_mode."""
    for paged, page_size in _probe_passes(modal_probe_plan):
        completed = _run_sglang_pass(
            modal_probe_plan,
            paged=paged,
            page_size=page_size,
            early_check=early_check,
        )
        if not completed:
            return False
    return True


def _observed_probe_plan(probe_plan: ProbePlan, events: list[dict[str, Any]]) -> ProbePlan:
    observed = {
        event.get("name")
        for event in events
        if isinstance(event.get("name"), str) and not bool(event.get("is_warmup"))
    }
    if not observed:
        return ProbePlan(targets=[], skipped=[], warmup_hooks=[])
    return ProbePlan(
        targets=[target for target in probe_plan.targets if target.name in observed],
        skipped=[],
        warmup_hooks=[],
    )


def _early_failure_reason(audit_report: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    audit_summary = audit_report.get("summary") if isinstance(audit_report, dict) else None
    if isinstance(audit_summary, dict) and int(audit_summary.get("rejected") or 0) > 0:
        return "definition_audit_rejected"
    for item in manifest.get("skipped", []) if isinstance(manifest, dict) else []:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "")
        if reason == "sanitize_failed" or "missing definition" in reason:
            return reason
    return None


def _build_remote_post_capture_outputs(
    *,
    modal_probe_plan: dict[str, Any],
    output_dir: Path,
    fitrace_definitions_dir: Path,
    probe_plan: ProbePlan,
    observed_only: bool,
) -> dict[str, Any]:
    events_path = output_dir / "events.jsonl"
    events = load_jsonl(events_path)
    stage_probe_plan = _observed_probe_plan(probe_plan, events) if observed_only else probe_plan
    collect_dir = output_dir / "collect"
    audited_definitions_dir = output_dir / "audited_definitions"
    definition_hints_dir = output_dir / "definition_hints"
    definition_audit_report = audit_and_repair_definitions(
        raw_definitions_dir=fitrace_definitions_dir,
        output_definitions_dir=audited_definitions_dir,
        output_hints_dir=definition_hints_dir,
        events_path=events_path,
        report_dir=output_dir / "definition_audit",
    )
    reviewed_overwrites = _materialize_reviewed_artifacts(
        modal_probe_plan=modal_probe_plan,
        definitions_dir=audited_definitions_dir,
        hints_dir=definition_hints_dir,
    )
    if reviewed_overwrites:
        definition_audit_report.setdefault("reviewed_overwrites", {}).update(reviewed_overwrites)
    definition_aliases = _definition_aliases(definition_audit_report)
    collect_plan = build_collect_plan_from_probe_plan(
        definitions=load_definitions(audited_definitions_dir),
        probe_plan=stage_probe_plan,
        events=events,
        definition_aliases=definition_aliases,
    )
    collect_plan_payload = collect_plan.to_jsonable()
    manifest = _build_remote_collect_outputs(
        events_path=events_path,
        collect_plan=collect_plan_payload,
        output_dir=collect_dir,
        hints_dir=definition_hints_dir,
        definition_aliases=definition_aliases,
    )
    return {
        "collect_dir": collect_dir,
        "audited_definitions_dir": audited_definitions_dir,
        "definition_hints_dir": definition_hints_dir,
        "definition_audit_report": definition_audit_report,
        "collect_plan": collect_plan_payload,
        "workload_manifest": manifest,
    }


def _definition_aliases(audit_report: dict[str, Any]) -> dict[str, str]:
    aliases = audit_report.get("aliases")
    if not isinstance(aliases, dict):
        return {}
    return {
        source: target
        for source, target in aliases.items()
        if isinstance(source, str) and source and isinstance(target, str) and target
    }


def _materialize_reviewed_artifacts(
    *,
    modal_probe_plan: dict[str, Any],
    definitions_dir: Path,
    hints_dir: Path,
) -> dict[str, list[str]]:
    artifacts = modal_probe_plan.get("reviewed_artifacts")
    if not isinstance(artifacts, dict):
        return {}
    overwritten_definitions = _write_reviewed_artifact_group(
        root=definitions_dir,
        artifacts=artifacts.get("definitions"),
        label="reviewed definition",
    )
    overwritten_hints = _write_reviewed_artifact_group(
        root=hints_dir,
        artifacts=artifacts.get("definition_hints"),
        label="reviewed definition hint",
    )
    return {
        "overwritten_definitions": overwritten_definitions,
        "overwritten_hints": overwritten_hints,
    }


def _write_reviewed_artifact_group(*, root: Path, artifacts: Any, label: str) -> list[str]:
    if artifacts is None:
        return []
    if not isinstance(artifacts, list):
        raise ValueError(f"{label} artifacts must be a list")
    overwritten: list[str] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise ValueError(f"{label} artifact #{index} must be an object")
        raw_path = item.get("path")
        data = item.get("data")
        if not isinstance(raw_path, str) or not raw_path.endswith(".json"):
            raise ValueError(f"{label} artifact #{index} has invalid path")
        if not isinstance(data, dict):
            raise ValueError(f"{label} artifact {raw_path} data must be an object")
        if label == "reviewed definition":
            data, fixes, issue = prepare_definition_for_output(data)
            if issue is not None:
                raise ValueError(f"{label} artifact {raw_path} failed schema gate: {issue}")
            if fixes:
                audit = data.get("audit")
                if not isinstance(audit, dict):
                    audit = {}
                audit["schema_fixes"] = fixes
                data["audit"] = audit
        relative = Path(raw_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"{label} artifact has unsafe path: {raw_path}")
        dst = root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            overwritten.append(str(relative))
        dst.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return overwritten


def run_remote_probe_entrypoint(
    *,
    modal_probe_plan: dict[str, Any],
    output_dir: Path,
    run_model: Callable[[Callable[[], bool] | None], bool],
) -> dict[str, Any]:
    """Install hooks, run the supplied model callback, and return capture metadata."""
    probe_plan = ProbePlan.from_jsonable(modal_probe_plan)
    max_captures_per_target = max_captures_from_plan_payload(modal_probe_plan)
    diagnostics = modal_probe_plan.get("diagnostics")
    diagnostic_full_scan = bool(isinstance(diagnostics, dict) and diagnostics.get("full_scan") is True)
    fitrace_definitions_dir = _prepare_fitrace_dump(output_dir)
    prepare_worker_injection(
        modal_probe_plan=modal_probe_plan,
        output_dir=output_dir,
        site_dir=Path(DEFAULT_WORKER_SITE_DIR),
    )
    hook_status: list[dict[str, Any]] = []
    early_stop: dict[str, Any] | None = None

    def early_check() -> bool:
        nonlocal early_stop
        stage = _build_remote_post_capture_outputs(
            modal_probe_plan=modal_probe_plan,
            output_dir=output_dir,
            fitrace_definitions_dir=fitrace_definitions_dir,
            probe_plan=probe_plan,
            observed_only=True,
        )
        reason = _early_failure_reason(stage["definition_audit_report"], stage["workload_manifest"])
        if reason is None:
            return False
        early_stop = {"reason": reason}
        print(f"[flashinfer_trace] early collect failed: {reason}", flush=True)
        return True

    with CaptureSession(
        probe_plan=probe_plan,
        output_dir=output_dir,
        reset_output=False,
        max_captures_per_target=max_captures_per_target,
    ) as session:
        hook_status = list(session.install_status)
        run_model(None if diagnostic_full_scan else early_check)

    events_path = output_dir / "events.jsonl"
    capture_root = output_dir / "captures"
    worker_status = _read_worker_status(output_dir / "hook_status")
    events = load_jsonl(events_path)
    result: dict[str, Any] = {
        "parse_report": build_event_report(events, probe_plan),
        "hook_status": hook_status,
        "worker_hook_status": worker_status,
        "summary": {
            "targets": len(probe_plan.targets),
            "skipped": len(probe_plan.skipped),
            "hooks_installed": sum(1 for item in hook_status if item.get("installed")),
            "worker_processes": len(worker_status),
            "worker_hooks_installed": sum(
                int(item.get("hooks_installed") or 0)
                for item in worker_status
                if isinstance(item, dict)
            ),
            "events_written": _count_lines(events_path),
            "captures_written": _count_files(capture_root),
            "max_captures_per_target": max_captures_per_target,
            "diagnostic_full_scan": diagnostic_full_scan,
        },
    }
    stage = _build_remote_post_capture_outputs(
        modal_probe_plan=modal_probe_plan,
        output_dir=output_dir,
        fitrace_definitions_dir=fitrace_definitions_dir,
        probe_plan=probe_plan,
        observed_only=False,
    )
    collect_dir = stage["collect_dir"]
    audited_definitions_dir = stage["audited_definitions_dir"]
    definition_hints_dir = stage["definition_hints_dir"]
    definition_audit_report = stage["definition_audit_report"]
    collect_plan_payload = stage["collect_plan"]
    manifest = stage["workload_manifest"]
    result["definition_audit_report"] = definition_audit_report
    result["collect_plan"] = collect_plan_payload
    result["workload_manifest"] = manifest
    if early_stop is not None:
        result["early_stop"] = early_stop
        result["summary"]["early_stopped"] = True
        result["summary"]["early_stop_reason"] = early_stop["reason"]
    result["collect_archive_b64"] = _read_dir_archive_b64(
        collect_dir,
        archive_name="collect.tar.gz",
        arcname="collect",
        label="collect",
    )
    result["definitions_archive_b64"] = _read_dir_archive_b64(
        audited_definitions_dir,
        archive_name="definitions.tar.gz",
        arcname="definitions",
        label="definitions",
    )
    result["definition_hints_archive_b64"] = _read_dir_archive_b64(
        definition_hints_dir,
        archive_name="definition_hints.tar.gz",
        arcname="definition_hints",
        label="definition hints",
    )
    result["summary"]["fitrace_definitions"] = _count_files(fitrace_definitions_dir)
    result["summary"]["audited_definitions"] = _count_files(audited_definitions_dir)
    result["summary"]["definition_hints"] = _count_files(definition_hints_dir)
    result["summary"]["definition_audit_repaired"] = definition_audit_report.get("summary", {}).get("repaired", 0)
    result["summary"]["definition_audit_rejected"] = definition_audit_report.get("summary", {}).get("rejected", 0)
    result["summary"]["workloads"] = manifest.get("summary", {}).get("workloads", 0)
    result["summary"]["sanitized"] = manifest.get("summary", {}).get("sanitized", 0)
    return result


def run_remote_sglang_probe(modal_probe_plan: dict[str, Any], remote_output_dir: str = DEFAULT_REMOTE_OUTPUT_DIR) -> dict[str, Any]:
    """Remote Modal entrypoint: install hooks, run SGLang, return captured events."""
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    print("[flashinfer_trace] remote probe entrypoint started", flush=True)
    output_dir = Path(remote_output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_remote_probe_entrypoint(
        modal_probe_plan=modal_probe_plan,
        output_dir=output_dir,
        run_model=lambda early_check: run_sglang_model(modal_probe_plan, early_check=early_check),
    )
    print(
        "[flashinfer_trace] remote probe complete: "
        f"events={result.get('summary', {}).get('events_written', 0)} "
        f"captures={result.get('summary', {}).get('captures_written', 0)}",
        flush=True,
    )
    return result


def run_modal_probe(
    *,
    modal_probe_plan: dict[str, Any],
    output_dir: Path,
    timeout: int = 3600,
    resume_call_id: str | None = None,
) -> dict[str, Any]:
    """Launch the remote Modal probe using ``modal run`` for streaming logs."""
    runtime = modal_probe_plan.get("runtime", {})
    image_name = str(runtime.get("image") or "lmsysorg/sglang:v0.5.12.post1")
    gpu = str(runtime.get("gpu") or "")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "modal_probe_plan.json"
    plan_path.write_text(json.dumps(modal_probe_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["FLASHINFER_TRACE_MODAL_IMAGE"] = image_name
    env["FLASHINFER_TRACE_MODAL_TIMEOUT"] = str(timeout)
    if gpu:
        env["FLASHINFER_TRACE_MODAL_GPU"] = gpu
    else:
        env.pop("FLASHINFER_TRACE_MODAL_GPU", None)

    cmd = [
        "modal",
        "run",
        "-m",
        "flashinfer_trace.runners.modal_app::probe",
        "--plan-path",
        str(plan_path),
        "--output-dir",
        str(output_dir),
    ]
    if resume_call_id:
        cmd.extend(["--resume-call-id", resume_call_id])
    print(
        "[flashinfer_trace] launching Modal CLI probe: "
        f"image={image_name} gpu={gpu or 'none'} timeout={timeout}s",
        flush=True,
    )
    print(f"[flashinfer_trace] command: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)

    result_path = output_dir / "modal_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Modal probe did not write result: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Result materialization (local/result-IO side of the Modal probe)
# ---------------------------------------------------------------------------
#
# Turn the result dict returned by the remote entrypoint into local reports and
# remote-collect outputs on disk.


def materialize_modal_result(result: dict[str, Any], output_dir: Path) -> None:
    """Materialize returned Modal probe artifacts into the run directory.

    ``output_dir`` is a transient Modal CLI handoff directory. Standard run
    artifacts are written under the parent run's ``output`` and ``reports``
    directories; raw Modal handoff files are not part of the public run layout.
    """
    print("[flashinfer_trace] remote result received; materializing local outputs", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir.parent
    shutil.rmtree(output_dir / "captures", ignore_errors=True)
    output_root = run_dir / "output"
    definitions_dir = output_root / "definitions"
    _materialize_definition_outputs(definitions_dir, result)
    _materialize_definition_hints_outputs(output_root / "definition_hints", result)
    _materialize_collect_outputs(
        result,
        collect_dir=output_dir / "collect",
        output_root=output_root,
        definitions_dir=definitions_dir,
    )
    result_path = output_dir / "modal_result.json"
    result_path.write_text(
        json.dumps(_redact_modal_result(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _redact_modal_result(result: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(result)
    for key in (
        "collect_archive_b64",
        "definitions_archive_b64",
        "definition_hints_archive_b64",
    ):
        value = redacted.get(key)
        if isinstance(value, str):
            redacted[key] = {
                "redacted": True,
                "encoded_bytes": len(value),
            }
    return redacted


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _build_remote_collect_outputs(
    *,
    events_path: Path,
    collect_plan: dict[str, Any],
    output_dir: Path,
    hints_dir: Path,
    definition_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    from flashinfer_trace.core.events import build_workload_manifest, load_jsonl

    events = load_jsonl(events_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    collect_plan_path = output_dir / "collect_plan.json"
    collect_plan_path.write_text(
        json.dumps(collect_plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = build_workload_manifest(
        events,
        CollectPlan.from_jsonable(collect_plan),
        output_dir=output_dir,
        hints_dir=hints_dir,
        definition_aliases=definition_aliases,
    )
    (output_dir / "workload_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_dir_archive_b64(root: Path, *, archive_name: str, arcname: str, label: str) -> str:
    if not root.exists():
        return ""
    archive_path = root.parent / archive_name
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(root, arcname=arcname)
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"[flashinfer_trace] remote {label} archive: {size_mb:.2f} MiB", flush=True)
    return base64.b64encode(archive_path.read_bytes()).decode("ascii")


def _read_worker_status(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    statuses: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            data = {"path": str(path), "error_type": type(exc).__name__, "error": str(exc)}
        statuses.append(data if isinstance(data, dict) else {"path": str(path), "data": data})
    return statuses


def _materialize_collect_outputs(
    result: dict[str, Any],
    *,
    collect_dir: Path,
    output_root: Path,
    definitions_dir: Path,
) -> None:
    archive_b64 = result.get("collect_archive_b64")
    if not isinstance(archive_b64, str) or not archive_b64:
        return

    _write_named_archive(root=collect_dir, archive_b64=archive_b64, expected_root="collect")

    # Incremental workload merge: move only new/updated targets, preserve existing ones.
    src_workloads = collect_dir / "workloads"
    dst_workloads = output_root / "workloads"
    if src_workloads.exists():
        for src_op_dir in src_workloads.iterdir():
            if not src_op_dir.is_dir():
                continue
            dst_op_dir = dst_workloads / src_op_dir.name
            dst_op_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_op_dir.iterdir():
                shutil.move(str(src_file), str(dst_op_dir / src_file.name))

    src_blob = collect_dir / "blob"
    dst_blob = output_root / "blob"
    if src_blob.exists():
        for src_op_dir in src_blob.glob("workloads/*"):
            if not src_op_dir.is_dir():
                continue
            dst_op_dir = dst_blob / "workloads" / src_op_dir.name
            # Remove stale safetensors for this target before writing new ones
            shutil.rmtree(dst_op_dir, ignore_errors=True)
            dst_op_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_op_dir.iterdir():
                shutil.move(str(src_file), str(dst_op_dir / src_file.name))

    rewrite_collect_output_paths(
        collect_dir,
        local_collect_dir=output_root,
        local_definitions_dir=definitions_dir,
    )
    plan_path = collect_dir / "collect_plan.json"
    manifest_path = collect_dir / "workload_manifest.json"
    if plan_path.exists():
        result["collect_plan"] = json.loads(plan_path.read_text(encoding="utf-8"))
    if manifest_path.exists():
        new_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Merge with existing manifest: new entries overwrite, existing skipped entries preserved
        existing_manifest: dict[str, Any] = {}
        # Check run_report for previously stored manifest
        run_report_path = output_root.parent / "reports" / "run_report.json"
        if run_report_path.exists():
            try:
                run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
                prev = run_report.get("collect", {}).get("manifest")
                if isinstance(prev, dict):
                    existing_manifest = prev
            except Exception:  # noqa: BLE001
                pass
        if existing_manifest:
            new_workload_names = {
                w["definition_name"]
                for w in (new_manifest.get("workloads") or [])
                if isinstance(w, dict) and w.get("definition_name")
            }
            preserved = [
                w for w in (existing_manifest.get("workloads") or [])
                if isinstance(w, dict) and w.get("definition_name") not in new_workload_names
            ]
            merged_workloads = preserved + (new_manifest.get("workloads") or [])
            new_manifest["workloads"] = merged_workloads
            # Recompute summary counts
            summary = new_manifest.get("summary") or {}
            summary["workloads"] = len(merged_workloads)
            summary["captures"] = sum(int(w.get("event_count", 0)) for w in merged_workloads if isinstance(w, dict))
            summary["sanitized"] = sum(int(w.get("sanitized_count", 0)) for w in merged_workloads if isinstance(w, dict))
            summary["workload_files"] = sum(len(w.get("blob_paths") or []) for w in merged_workloads if isinstance(w, dict))
            new_manifest["summary"] = summary
        result["workload_manifest"] = new_manifest
    shutil.rmtree(collect_dir, ignore_errors=True)


def _materialize_definition_outputs(root: Path, result: dict[str, Any]) -> None:
    archive_b64 = result.get("definitions_archive_b64")
    if isinstance(archive_b64, str) and archive_b64:
        _write_named_archive(root=root, archive_b64=archive_b64, expected_root="definitions")


def _materialize_definition_hints_outputs(root: Path, result: dict[str, Any]) -> None:
    archive_b64 = result.get("definition_hints_archive_b64")
    if isinstance(archive_b64, str) and archive_b64:
        _write_named_archive(root=root, archive_b64=archive_b64, expected_root="definition_hints")


def rewrite_collect_output_paths(
    collect_dir: Path,
    *,
    local_collect_dir: Path,
    local_definitions_dir: Path | None = None,
) -> None:
    """Rewrite remote absolute paths in collect outputs to local extracted paths."""
    for path in (collect_dir / "workload_manifest.json", collect_dir / "collect_plan.json"):
        _rewrite_collect_json_paths(
            path,
            local_collect_dir=local_collect_dir,
            local_definitions_dir=local_definitions_dir,
        )


def _rewrite_collect_json_paths(
    path: Path,
    *,
    local_collect_dir: Path,
    local_definitions_dir: Path | None,
) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    remote_collect_prefix = f"{DEFAULT_REMOTE_OUTPUT_DIR}/collect/"
    remote_definitions_prefix = f"{DEFAULT_REMOTE_OUTPUT_DIR}/definitions/"
    remote_audited_definitions_prefix = f"{DEFAULT_REMOTE_OUTPUT_DIR}/audited_definitions/"

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith(remote_collect_prefix):
                return str(local_collect_dir / value.removeprefix(remote_collect_prefix))
            if local_definitions_dir is not None and value.startswith(remote_definitions_prefix):
                return str(local_definitions_dir / value.removeprefix(remote_definitions_prefix))
            if local_definitions_dir is not None and value.startswith(remote_audited_definitions_prefix):
                return str(local_definitions_dir / value.removeprefix(remote_audited_definitions_prefix))
            return value
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    rewritten = rewrite(payload)
    path.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_named_archive(*, root: Path, archive_b64: str, expected_root: str) -> None:
    root_parent = root.parent
    root_parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(root, ignore_errors=True)
    archive_path = root_parent / f"{expected_root}.tar.gz"
    archive_path.write_bytes(base64.b64decode(archive_b64.encode("ascii")))
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = root_parent / member.name
            if not member_path.resolve().is_relative_to(root_parent.resolve()):
                raise ValueError(f"unsafe archive member: {member.name}")
        try:
            archive.extractall(root_parent, filter="data")
        except TypeError:
            archive.extractall(root_parent)
    archive_path.unlink(missing_ok=True)
