"""Planning layer: reviewed targets and run config into deterministic plans.

This module is the pure, side-effect-light planning stage of the pipeline:

* load reviewed runtime target config into ``ApprovedTarget`` entries;
* turn reviewed targets into hook specs (``build_probe_plan``);
* pair reviewed probe targets with audited/reviewed definitions for collect;
* validate explicit Modal/SGLang runtime settings (``plan_runtime``);
* shape reviewed prompt/scenario settings into a JSON-able ``modal_probe_plan``.

None of it touches Modal or a GPU, which keeps it cheap to unit-test. Execution
lives in capture/modal_runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flashinfer_trace.core.schemas import (
    BACKENDS,
    DEFINITION_SOURCES,
    PROBE_MODES,
    ApprovedTarget,
    CollectPlan,
    CollectTarget,
    DefinitionRef,
    ProbePlan,
    ProbeTarget,
    WarmupHook,
    capture_spec_from_jsonable,
    dispatch_spec_from_jsonable,
)

DEFAULT_SHAREGPT_PATH = Path("sharegpt_100.json")
APPROVED_TARGET_FIELDS = {
    "name",
    "role",
    "target",
    "module",
    "attr",
    "definition_source",
    "backend",
    "collect",
    "definition_name",
    "op_type",
    "variant",
    "probe_mode",
    "page_size",
    "dispatch_value",
    "companion_attrs",
    "capture",
    "dispatch",
}


# ---------------------------------------------------------------------------
# Shared JSON helper
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reviewed target config: config/approved_targets.json -> ApprovedTarget[]
# ---------------------------------------------------------------------------


def load_approved_targets(path: Path) -> list[ApprovedTarget]:
    """Load reviewed runtime target entries."""
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"approved targets must be a list: {path}")

    targets: list[ApprovedTarget] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"approved target #{index} must be an object")
        unexpected = sorted(set(item) - APPROVED_TARGET_FIELDS)
        if unexpected:
            raise ValueError(f"approved target #{index} has unexpected fields: {unexpected}")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"approved target #{index} has no name")
        role = item.get("role", "target")
        if role not in {"target", "warmup"}:
            raise ValueError(f"approved target {name} has invalid role: {role}")
        target = item.get("target")
        if isinstance(target, str):
            target = target.strip() or None
        elif target is not None:
            raise ValueError(f"approved target {name} has non-string target")
        module = item.get("module")
        if isinstance(module, str):
            module = module.strip() or None
        else:
            module = None
        attr = item.get("attr")
        if isinstance(attr, str):
            attr = attr.strip() or None
        else:
            attr = None

        op_type = item.get("op_type")
        if op_type is not None and (not isinstance(op_type, str) or not op_type.strip()):
            raise ValueError(f"approved target {name} has invalid op_type: {op_type}")
        variant = item.get("variant")
        if variant is not None and (not isinstance(variant, str) or not variant.strip()):
            raise ValueError(f"approved target {name} has invalid variant: {variant}")
        backend = item.get("backend", "unknown")
        if not isinstance(backend, str) or backend not in BACKENDS:
            raise ValueError(f"approved target {name} has invalid backend: {backend}")
        definition_source = item.get("definition_source", "unknown")
        if not isinstance(definition_source, str) or definition_source not in DEFINITION_SOURCES:
            raise ValueError(f"approved target {name} has invalid definition_source: {definition_source}")
        if "collect" not in item:
            raise ValueError(f"approved target {name} must declare collect")
        collect = item.get("collect")
        if not isinstance(collect, bool):
            raise ValueError(f"approved target {name} has invalid collect: {collect}")
        definition_name = item.get("definition_name")
        if isinstance(definition_name, str):
            definition_name = definition_name.strip() or None
        elif definition_name is not None:
            raise ValueError(f"approved target {name} has non-string definition_name")
        probe_mode = item.get("probe_mode", "default")
        if not isinstance(probe_mode, str) or probe_mode not in PROBE_MODES:
            raise ValueError(f"approved target {name} has invalid probe_mode: {probe_mode}")
        raw_page_size = item.get("page_size")
        if raw_page_size is None:
            page_size = None
        elif type(raw_page_size) is int and raw_page_size > 0:
            page_size = raw_page_size
        else:
            raise ValueError(f"approved target {name} has invalid page_size: {raw_page_size}")
        raw_dispatch_value = item.get("dispatch_value")
        if raw_dispatch_value is None:
            dispatch_value = None
        elif type(raw_dispatch_value) is int:
            dispatch_value = raw_dispatch_value
        else:
            raise ValueError(f"approved target {name} has invalid dispatch_value: {raw_dispatch_value}")
        raw_companions = item.get("companion_attrs", [])
        if isinstance(raw_companions, list) and all(isinstance(c, str) for c in raw_companions):
            companion_attrs = [c.strip() for c in raw_companions if c.strip()]
        else:
            raise ValueError(f"approved target {name} has invalid companion_attrs: {raw_companions}")
        dispatch = dispatch_spec_from_jsonable(item.get("dispatch"), context=f"approved target {name}")
        raw_capture = item.get("capture")
        if role == "warmup":
            if raw_capture is not None:
                raise ValueError(f"warmup hook {name} must not declare capture")
            if dispatch is not None:
                raise ValueError(f"warmup hook {name} must not declare dispatch")
            capture = None
        else:
            capture = capture_spec_from_jsonable(raw_capture, context=f"approved target {name}")
        targets.append(
            ApprovedTarget(
                name=name.strip(),
                role=role,
                target=target,
                module=module,
                attr=attr,
                definition_source=definition_source,
                backend=backend,
                collect=collect,
                definition_name=definition_name,
                op_type=op_type.strip() if isinstance(op_type, str) else None,
                variant=variant.strip() if isinstance(variant, str) else None,
                probe_mode=probe_mode,
                page_size=page_size,
                dispatch_value=dispatch_value,
                companion_attrs=companion_attrs,
                capture=capture,
                dispatch=dispatch,
            )
        )
    return targets

# ---------------------------------------------------------------------------
# Probe plan: reviewed targets -> hook specs
# ---------------------------------------------------------------------------


def build_probe_plan(approved_targets: list[ApprovedTarget]) -> ProbePlan:
    """Return approved dotted Python call targets that should be hooked during probe."""
    probe_targets: list[ProbeTarget] = []
    warmup_hooks: list[WarmupHook] = []
    skipped: list[dict[str, str]] = []
    seen_names: set[str] = set()
    seen_warmup: set[str] = set()

    for entry in approved_targets:
        if entry.role == "warmup":
            if not entry.module or not entry.attr:
                raise ValueError(f"warmup hook {entry.name} has no explicit module/attr")
            if entry.name in seen_warmup:
                raise ValueError(f"duplicate warmup hook: {entry.name}")
            seen_warmup.add(entry.name)
            warmup_hooks.append(WarmupHook(name=entry.name, module=entry.module, attr=entry.attr))
            continue

        if not entry.target:
            raise ValueError(f"reviewed target {entry.name} has no target API")
        if not entry.module or not entry.attr:
            raise ValueError(f"reviewed target {entry.name} has no explicit hook module/attr")
        if entry.capture is None:
            raise ValueError(f"reviewed target {entry.name} has no capture spec")
        if entry.collect and entry.backend != "flashinfer" and not entry.definition_name:
            raise ValueError(f"reviewed non-fitrace target {entry.name} collect requires definition_name")
        if entry.name in seen_names:
            raise ValueError(f"duplicate reviewed target: {entry.name}")

        seen_names.add(entry.name)
        probe_target = ProbeTarget(
            name=entry.name,
            target=entry.target,
            module=entry.module,
            attr=entry.attr,
            definition_name=entry.definition_name,
            op_type=entry.op_type,
            variant=entry.variant,
            backend=entry.backend,
            collect=entry.collect,
            probe_mode=entry.probe_mode,
            page_size=entry.page_size,
            dispatch_value=entry.dispatch_value,
            companion_attrs=list(entry.companion_attrs),
            capture=entry.capture,
            dispatch=entry.dispatch,
        )
        if probe_target.dispatch is not None and _dispatch_expected_value(probe_target) is None:
            raise ValueError(f"reviewed target {entry.name} dispatch has no expected value")
        if probe_target.page_size is not None and probe_target.dispatch is None:
            raise ValueError(f"reviewed target {entry.name} page_size requires reviewed dispatch")
        probe_targets.append(probe_target)

    probe_targets.sort(key=lambda target: target.name)
    warmup_hooks.sort(key=lambda hook: hook.name)
    return ProbePlan(targets=probe_targets, skipped=skipped, warmup_hooks=warmup_hooks)


def _dispatch_expected_value(target: ProbeTarget) -> int | None:
    if target.dispatch is None:
        return None
    if target.dispatch_value is not None:
        return target.dispatch_value
    value = getattr(target, target.dispatch.field, None)
    return value if isinstance(value, int) else None


# ---------------------------------------------------------------------------
# Collect plan: reviewed/audited definitions + ProbePlan -> CollectPlan
# ---------------------------------------------------------------------------


def load_definitions(definitions_dir: Path) -> dict[str, DefinitionRef]:
    """Load minimal definition metadata by definition name."""
    definitions: dict[str, DefinitionRef] = {}
    for path in sorted(definitions_dir.rglob("*.json")):
        data = _load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"definition must be an object: {path}")
        name = data.get("name")
        op_type = data.get("op_type")
        if not isinstance(name, str) or not name:
            raise ValueError(f"definition has invalid name: {path}")
        if not isinstance(op_type, str) or not op_type:
            raise ValueError(f"definition {name} has invalid op_type: {path}")
        raw_tags = data.get("tags", [])
        if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
            raise ValueError(f"definition {name} has invalid tags: {path}")
        raw_axes = data.get("axes", {})
        if not isinstance(raw_axes, dict):
            raise ValueError(f"definition {name} has invalid axes: {path}")
        if name in definitions:
            raise ValueError(f"duplicate definition name {name}: {path}")
        definitions[name] = DefinitionRef(
            name=name,
            op_type=op_type,
            path=path,
            tags=raw_tags,
            axes=raw_axes,
        )
    return definitions


def build_collect_plan_from_probe_plan(
    *,
    definitions: dict[str, DefinitionRef],
    probe_plan: ProbePlan,
    events: list[dict[str, Any]],
    definition_aliases: dict[str, str] | None = None,
) -> CollectPlan:
    """Build collect targets from already-reviewed probe targets.

    This is used by the remote collect path. FlashInfer targets normally get
    definition names from same-run fitrace events. Non-FI targets must carry a
    reviewed ``definition_name`` in the target; capture writes that name into
    the event, and this function only accepts it if the reviewed definition is
    present in ``definitions``.
    """
    collect_targets: list[CollectTarget] = []
    skipped: list[dict[str, str]] = list(probe_plan.skipped)
    seen: set[tuple[str, str]] = set()
    aliases = definition_aliases or {}
    events_by_target = _events_by_target_definition(events)

    for target in probe_plan.targets:
        if not target.collect:
            skipped.append({
                "name": target.name,
                "reason": "collect is false",
            })
            continue
        if target.backend != "flashinfer" and not target.definition_name:
            skipped.append({
                "name": target.name,
                "reason": f"non-fitrace collect target has no reviewed definition_name: {target.backend}",
            })
            continue
        matched_definitions = []
        for raw_definition_name in sorted(events_by_target.get(target.name, set())):
            definition_name = aliases.get(raw_definition_name, raw_definition_name)
            definition = definitions.get(definition_name)
            if definition is None:
                skipped.append({
                    "name": target.name,
                    "reason": f"event referenced missing definition: {raw_definition_name}",
                })
                continue
            page_size_axis = definition.axes.get("page_size")
            matched_page_size = (
                int(page_size_axis["value"])
                if isinstance(page_size_axis, dict) and isinstance(page_size_axis.get("value"), int)
                else target.page_size
            )
            matched_definitions.append((definition, matched_page_size))
        if not matched_definitions:
            skipped.append({"name": target.name, "reason": "no event-linked definition"})
            continue

        for definition, matched_page_size in matched_definitions:
            key = (target.name, definition.name)
            if key in seen:
                skipped.append({
                    "name": target.name,
                    "reason": f"duplicate collect target for definition: {definition.name}",
                })
                continue
            seen.add(key)
            collect_targets.append(
                CollectTarget(
                    name=target.name,
                    definition_name=definition.name,
                    op_type=definition.op_type,
                    target=target.target,
                    backend=target.backend,
                    collect=target.collect,
                    definition_path=definition.path,
                    page_size=matched_page_size,
                )
            )

    collect_targets.sort(key=lambda item: (item.name, item.definition_name))
    return CollectPlan(targets=collect_targets, skipped=skipped)


def _events_by_target_definition(events: list[dict[str, Any]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for event in events:
        if bool(event.get("is_warmup")):
            continue
        name = event.get("name")
        definition_name = event.get("definition_name")
        if isinstance(name, str) and name and isinstance(definition_name, str) and definition_name:
            grouped.setdefault(name, set()).add(definition_name)
    return grouped


# ---------------------------------------------------------------------------
# Runtime planning: validate explicit Modal/SGLang settings for a probe run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimePlan:
    """Runtime settings needed by a Modal/SGLang probe."""

    model_name: str
    tp_size: int
    gpu: str
    image: str
    sources: dict[str, str]
    warnings: list[str]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "backend": "modal",
            "model_name": self.model_name,
            "tp_size": self.tp_size,
            "gpu": self.gpu,
            "image": self.image,
            "sources": self.sources,
            "warnings": self.warnings,
        }


def plan_runtime(
    *,
    model_name: str,
    image: str,
    gpu: str,
    tp_size: int,
) -> RuntimePlan:
    """Validate reviewed runtime settings.

    Runtime choices are onboarding/review decisions. The core does not query
    HF, sgl-cookbook, or model-name heuristics while building a run plan.
    """
    if not model_name:
        raise ValueError("model_name is required")
    if not image:
        raise ValueError("image is required in run_config.json or CLI")
    if not gpu:
        raise ValueError("gpu is required in run_config.json or CLI")
    try:
        final_tp_size = int(tp_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("tp_size must be >= 1") from exc
    if final_tp_size < 1:
        raise ValueError("tp_size must be >= 1")

    sources = {
        "tp_size": "reviewed",
        "gpu": "reviewed",
        "image": "reviewed",
    }
    return RuntimePlan(
        model_name=model_name,
        tp_size=final_tp_size,
        gpu=gpu,
        image=image,
        sources=sources,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Prompt/scenario planning for Modal execution
# ---------------------------------------------------------------------------


def build_modal_probe_plan(
    *,
    probe_plan: ProbePlan,
    model_name: str,
    output_dir: Path,
    image: str,
    gpu: str,
    tp_size: int,
    batch_sizes: list[int] | None = None,
    max_new_tokens: int | None = None,
    supplemental_runs: list[dict[str, Any]] | None = None,
    max_captures_per_target: int | None = None,
    disable_cuda_graph: bool | None = None,
    enable_piecewise_cuda_graph: bool | None = None,
    force_flashinfer_backends: bool | None = None,
    mem_fraction_static: float | None = None,
    cuda_graph_max_bs: int | None = None,
    engine_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a plan for a Modal-backed probe run."""
    effective_max_new_tokens = _reviewed_max_new_tokens(max_new_tokens=max_new_tokens)
    selected_batch_sizes = _reviewed_batch_sizes(batch_sizes=batch_sizes)
    selected_prompts = _load_sharegpt_prompts(DEFAULT_SHAREGPT_PATH)
    selected_prompt_scenarios = _build_sharegpt_scenarios(
        prompts=selected_prompts,
        batch_sizes=selected_batch_sizes,
        max_new_cap=effective_max_new_tokens,
    )
    selected_supplemental_runs = _reviewed_supplemental_runs(
        supplemental_runs=supplemental_runs,
    )
    selected_max_captures = _reviewed_max_captures_per_target(
        max_captures_per_target=max_captures_per_target,
    )
    runtime_plan = plan_runtime(
        model_name=model_name,
        image=image,
        gpu=gpu,
        tp_size=tp_size,
    )
    probe_modes = sorted({target.probe_mode for target in probe_plan.targets})
    return {
        "model_name": model_name,
        "runtime": runtime_plan.to_jsonable(),
        "probe_modes": probe_modes,
        "prompt_scenarios": selected_prompt_scenarios,
        "sampling": {
            "max_new_tokens": effective_max_new_tokens,
        },
        "supplemental_runs": selected_supplemental_runs,
        "sglang": {
            "disable_cuda_graph": True if disable_cuda_graph is None else bool(disable_cuda_graph),
            "enable_piecewise_cuda_graph": enable_piecewise_cuda_graph,
            "force_flashinfer_backends": True if force_flashinfer_backends is None else bool(force_flashinfer_backends),
            "mem_fraction_static": 0.7 if mem_fraction_static is None else float(mem_fraction_static),
            "cuda_graph_max_bs": cuda_graph_max_bs,
            "engine_kwargs": _reviewed_engine_kwargs(engine_kwargs=engine_kwargs),
        },
        "capture_limits": {
            "max_captures_per_target": selected_max_captures,
        },
        "probe_plan": probe_plan.to_jsonable(),
    }


def _reviewed_max_new_tokens(*, max_new_tokens: int | None) -> int:
    return _reviewed_positive_int(max_new_tokens, "max_new_tokens")


def _reviewed_batch_sizes(*, batch_sizes: list[int] | None) -> tuple[int, ...]:
    if batch_sizes is None:
        raise ValueError("collect run requires reviewed batch_sizes")
    if not batch_sizes:
        raise ValueError("batch_sizes must contain at least one positive integer")
    for index, item in enumerate(batch_sizes):
        _reviewed_positive_int(item, f"batch_sizes[{index}]")
    return tuple(batch_sizes)


def _reviewed_supplemental_runs(*, supplemental_runs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if supplemental_runs is None:
        raise ValueError("collect run requires reviewed supplemental_runs")
    reviewed: list[dict[str, Any]] = []
    for index, item in enumerate(supplemental_runs):
        if not isinstance(item, dict):
            raise ValueError(f"supplemental_runs[{index}] must be an object")
        unexpected = sorted(set(item) - {"name", "sampling_params", "allowed_op_types", "use_scenario_tokens"})
        if unexpected:
            raise ValueError(f"supplemental_runs[{index}] has unsupported keys: {unexpected}")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"supplemental_runs[{index}].name must be a non-empty string")
        raw_params = item.get("sampling_params")
        if not isinstance(raw_params, dict):
            raise ValueError(f"supplemental_runs[{index}].sampling_params must be an object")
        sampling_params: dict[str, Any] = {}
        for key, value in raw_params.items():
            if key not in {"max_new_tokens", "temperature", "top_k", "top_p"}:
                raise ValueError(f"supplemental_runs[{index}].sampling_params has unsupported key: {key}")
            if key in {"max_new_tokens", "top_k"}:
                if type(value) is not int:
                    raise ValueError(f"supplemental_runs[{index}].sampling_params.{key} must be an integer")
                sampling_params[key] = value
            else:
                if type(value) not in {int, float}:
                    raise ValueError(f"supplemental_runs[{index}].sampling_params.{key} must be numeric")
                sampling_params[key] = float(value)
        if "max_new_tokens" in sampling_params and sampling_params["max_new_tokens"] < 1:
            raise ValueError(f"supplemental_runs[{index}].sampling_params.max_new_tokens must be >= 1")
        raw_allowed = item.get("allowed_op_types")
        if raw_allowed is None:
            allowed_op_types = None
        elif isinstance(raw_allowed, list) and all(isinstance(value, str) and value for value in raw_allowed):
            allowed_op_types = list(raw_allowed)
        else:
            raise ValueError(f"supplemental_runs[{index}].allowed_op_types must be a list of strings")
        use_scenario_tokens = item.get("use_scenario_tokens", False)
        if type(use_scenario_tokens) is not bool:
            raise ValueError(f"supplemental_runs[{index}].use_scenario_tokens must be a boolean")
        reviewed.append({
            "name": name,
            "sampling_params": sampling_params,
            "allowed_op_types": allowed_op_types,
            "use_scenario_tokens": use_scenario_tokens,
        })
    return reviewed


def _reviewed_engine_kwargs(*, engine_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    if engine_kwargs is None:
        return {}
    if not isinstance(engine_kwargs, dict):
        raise ValueError("engine_kwargs must be an object")
    reviewed: dict[str, Any] = {}
    for key, value in engine_kwargs.items():
        if not isinstance(key, str) or not key:
            raise ValueError("engine_kwargs keys must be non-empty strings")
        if type(value) not in {str, int, float, bool} and value is not None:
            raise ValueError(f"engine_kwargs.{key} must be a JSON scalar")
        reviewed[key] = value
    return reviewed


def _reviewed_max_captures_per_target(*, max_captures_per_target: int | None) -> int:
    return _reviewed_positive_int(max_captures_per_target, "max_captures_per_target")


def _reviewed_positive_int(value: int | None, field: str) -> int:
    if value is None:
        raise ValueError(f"collect run requires reviewed {field}")
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _load_sharegpt_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"collect run requires {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of prompts")
    prompts: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str):
            raise ValueError(f"{path}[{index}] must be a string prompt")
        prompt = item.strip()
        if prompt:
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"{path} must contain at least one non-empty prompt")
    return prompts


# Fixed collect strategy: turn the reviewed ShareGPT prompt file plus reviewed
# batch sizes into concrete request batches for the remote SGLang run.
def _build_sharegpt_scenarios(
    *,
    prompts: list[str],
    batch_sizes: tuple[int, ...],
    max_new_cap: int,
) -> list[dict[str, Any]]:
    ranked = sorted((str(prompt) for prompt in prompts if str(prompt)), key=len)
    if not ranked:
        raise ValueError("collect run requires at least one prompt")
    n = len(ranked)
    default_specs = [
        ("b1_long", 1, "long", 96),
        ("b2_medium", 2, "medium", 64),
        ("b4_medium", 4, "medium", 32),
        ("b8_short", 8, "short", 32),
        ("b16_short", 16, "short", 16),
        ("b32_very_short", 32, "very_short", 8),
        ("b64_very_short", 64, "very_short", 8),
    ]
    default_by_batch = {batch: spec for spec in default_specs for batch in [spec[1]]}

    def auto_spec(batch_size: int) -> tuple[str, int, str, int]:
        if batch_size <= 1:
            return (f"b{batch_size}_long", batch_size, "long", 96)
        if batch_size <= 4:
            return (f"b{batch_size}_medium", batch_size, "medium", 64)
        if batch_size <= 16:
            return (f"b{batch_size}_short", batch_size, "short", 32)
        return (f"b{batch_size}_very_short", batch_size, "very_short", 8)

    def pool(bucket: str, batch_size: int) -> list[str]:
        if bucket == "very_short":
            return ranked[:max(n // 4, 1)]
        if bucket == "short":
            return ranked[:max(batch_size, n // 2, 1)]
        if bucket == "medium":
            lo, hi = n // 4, max(n * 3 // 4, n // 4 + batch_size)
            return ranked[lo:min(hi, n)] or ranked
        return ranked[-max(batch_size, n // 4, 1):]

    scenarios: list[dict[str, Any]] = []
    for index, batch_size in enumerate(batch_sizes):
        name, batch_size, bucket, max_new = default_by_batch.get(batch_size, auto_spec(batch_size))
        candidates = pool(bucket, batch_size)
        batch = [candidates[(index * 7 + i) % len(candidates)] for i in range(batch_size)]
        scenarios.append(
            {
                "name": name,
                "prompts": batch,
                "max_new_tokens": min(max_new, max_new_cap),
            }
        )
    return scenarios
