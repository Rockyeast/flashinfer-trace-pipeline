"""Parse probe events into a report and sanitize captures into workloads.

This module owns the remote post-capture pipeline: parsing the remote internal
event log into a deterministic report and turning captured call payloads into
official-style workload entries (the sanitizer half).
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from flashinfer_trace.definition_repairs.hint_rules import apply_axis_hint_rule
from flashinfer_trace.core.schemas import CollectPlan, CollectTarget, ProbePlan


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL event objects, rejecting malformed lines."""
    events: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"events file not found: {path}")
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f"event line {line_no} must be a JSON object")
            events.append(data)
    return events


def build_event_report(events: list[dict[str, Any]], probe_plan: ProbePlan | None = None) -> dict[str, Any]:
    """Aggregate event counts and warmup filtering diagnostics."""
    approved_names = {target.name for target in probe_plan.targets} if probe_plan else set()
    counts: dict[str, Counter[str]] = {}
    metadata_by_name: dict[str, dict[str, Any]] = {}

    for event in events:
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("event has no non-empty name")
        if approved_names and name not in approved_names:
            raise ValueError(f"event name not in probe plan: {name}")
        metadata_by_name.setdefault(
            name,
            {
                "target": event.get("target"),
                "definition_name": event.get("definition_name"),
                "op_type": event.get("op_type"),
                "variant": event.get("variant"),
                "probe_mode": event.get("probe_mode"),
            },
        )
        bucket = counts.setdefault(name, Counter())
        bucket["total"] += 1
        active_probe_mode = event.get("active_probe_mode")
        if isinstance(active_probe_mode, str) and active_probe_mode:
            bucket[f"active:{active_probe_mode}"] += 1
        if bool(event.get("is_warmup")):
            bucket["warmup"] += 1
        else:
            bucket["non_warmup"] += 1

    observed = []
    for name, count in sorted(counts.items()):
        item = {
            "name": name,
            "total": count["total"],
            "warmup": count["warmup"],
            "non_warmup": count["non_warmup"],
            "active_probe_modes": {
                key.removeprefix("active:"): value
                for key, value in count.items()
                if key.startswith("active:")
            },
        }
        for key, value in metadata_by_name.get(name, {}).items():
            if isinstance(value, str) and value:
                item[key] = value
        observed.append(item)
    missing = sorted(name for name in approved_names if counts.get(name, Counter())["non_warmup"] == 0)
    dispatch_errors = _dispatch_errors(events, probe_plan)

    total_warmup = sum(count["warmup"] for count in counts.values())
    total_non_warmup = sum(count["non_warmup"] for count in counts.values())
    report = {
        "summary": {
            "events": len(events),
            "warmup_events": total_warmup,
            "non_warmup_events": total_non_warmup,
            "observed_targets": len(observed),
            "missing_targets": len(missing),
        },
        "observed_targets": observed,
        "missing_targets": missing,
    }
    if dispatch_errors:
        report["summary"]["dispatch_errors"] = len(dispatch_errors)
        report["dispatch_errors"] = dispatch_errors
    return report


def _dispatch_errors(events: list[dict[str, Any]], probe_plan: ProbePlan | None) -> list[dict[str, Any]]:
    if probe_plan is None:
        return []
    by_name: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if bool(event.get("is_warmup")):
            continue
        name = event.get("name")
        if isinstance(name, str) and name:
            by_name.setdefault(name, []).append(event)

    errors: list[dict[str, Any]] = []
    for target in probe_plan.targets:
        if target.dispatch is None:
            continue
        target_events = by_name.get(target.name, [])
        if not target_events:
            continue
        field = target.dispatch.field
        expected = target.dispatch_value if target.dispatch_value is not None else getattr(target, field, None)
        if expected is None:
            errors.append({"name": target.name, "field": field, "reason": "target has no expected dispatch value"})
            continue
        missing = sum(1 for event in target_events if field not in event)
        values = sorted({event[field] for event in target_events if type(event.get(field)) is int})
        invalid_values = sorted({
            repr(event.get(field))
            for event in target_events
            if field in event and type(event.get(field)) is not int
        })
        if missing:
            errors.append({
                "name": target.name,
                "field": field,
                "reason": "event missing dispatch field",
                "count": missing,
            })
        if invalid_values:
            errors.append({
                "name": target.name,
                "field": field,
                "reason": "event has non-integer dispatch value",
                "values": invalid_values,
            })
        if values and (len(values) != 1 or values[0] != expected):
            errors.append({
                "name": target.name,
                "field": field,
                "reason": "observed dispatch value does not match target",
                "expected": expected,
                "observed": values,
            })
    return errors

def _safe_name(value: str) -> str:
    text = value.strip()
    chars = [char if char.isalnum() or char in {"_", "-", "."} else "_" for char in text]
    compact = "_".join(part for part in "".join(chars).split("_") if part)
    if not compact:
        raise ValueError("empty name cannot be used as a path component")
    return compact


def _write_dataset_workloads(
    *,
    name: str,
    definition_name: str,
    op_type: Any,
    definition: dict[str, Any],
    hints: dict[str, Any] | None,
    capture_paths: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Write sanitized workload JSONL plus referenced tensor blobs."""
    if not isinstance(op_type, str) or not op_type:
        raise ValueError(f"collect target {name} has no op_type")
    safe_op_type = _safe_name(op_type)
    safe_definition_name = _safe_name(definition_name)
    jsonl_path = output_dir / "workloads" / safe_op_type / f"{safe_definition_name}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    blob_paths: list[str] = []
    workload_paths: list[str] = []
    sanitized = 0
    reject_reasons: Counter[str] = Counter()
    reject_examples: dict[str, Any] = {}
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for raw_path in capture_paths:
            src = Path(raw_path)
            if not src.exists():
                reject_reasons["missing_capture_file"] += 1
                continue
            entry: dict[str, Any] | None = None
            captured, load_error = load_capture_payload(src)
            if captured is not None:
                entry, diagnostic = build_sanitized_workload_entry(
                    captured=captured,
                    definition=definition,
                    hints=hints,
                    output_dir=output_dir,
                )
                if entry is not None:
                    sanitized += 1
                else:
                    reason = str(diagnostic.get("reason") or "unknown")
                    reject_reasons[reason] += 1
                    reject_examples.setdefault(reason, diagnostic)
            elif load_error:
                reject_reasons[load_error] += 1
            if entry is None:
                continue
            for input_spec in entry["workload"]["inputs"].values():
                if isinstance(input_spec, dict) and input_spec.get("type") == "safetensors":
                    raw_blob = input_spec.get("path")
                    if isinstance(raw_blob, str) and raw_blob.startswith("./"):
                        blob_paths.append(str(output_dir / raw_blob[2:]))
            jsonl.write(json.dumps(entry, sort_keys=True) + "\n")
            workload_paths.append(str(jsonl_path))
    if sanitized == 0 and jsonl_path.exists():
        jsonl_path.unlink()

    return {
        "jsonl_path": str(jsonl_path) if sanitized else None,
        "blob_paths": blob_paths,
        "workload_paths": sorted(set(workload_paths)),
        "sanitized_count": sanitized,
        "reject_reasons": dict(reject_reasons),
        "reject_examples": reject_examples,
    }


def build_workload_manifest(
    events: list[dict[str, Any]],
    collect_plan: CollectPlan,
    *,
    output_dir: Path,
    hints_dir: Path | None = None,
    definition_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build workload artifacts from non-warmup events and collect targets."""
    definition_aliases = definition_aliases or {}
    events_by_name: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if bool(event.get("is_warmup")):
            continue
        name = event.get("name")
        if isinstance(name, str) and name:
            events_by_name.setdefault(name, []).append(event)

    workloads: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = list(collect_plan.skipped)
    for target in collect_plan.targets:
        name = target.name
        target_events = events_by_name.get(name, [])
        target_events = [
            event
            for event in target_events
            if definition_aliases.get(str(event.get("definition_name")), event.get("definition_name"))
            == target.definition_name
        ]
        if not target_events:
            skipped.append({"name": name, "reason": "no non-warmup event payloads"})
            continue
        capture_paths: list[str] = []
        for event in target_events:
            raw_path = event.get("capture_path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"event for collect target {name} has no capture_path")
            capture_paths.append(str(Path(raw_path)))
        definition = _target_definition(target)
        hints = _target_hints(target, hints_dir)
        real_inputs = sorted(_real_inputs(hints))
        dataset_paths = _write_dataset_workloads(
            name=name,
            definition_name=target.definition_name,
            op_type=target.op_type,
            definition=definition,
            hints=hints,
            capture_paths=capture_paths,
            output_dir=output_dir,
        )
        if int(dataset_paths["sanitized_count"]) == 0:
            skipped.append({
                "name": name,
                "reason": "sanitize_failed",
                "sanitize_reject_reasons": dataset_paths["reject_reasons"],
                "sanitize_reject_examples": dataset_paths["reject_examples"],
            })
            continue
        workloads.append({
            "name": name,
            "definition_name": target.definition_name,
            "op_type": target.op_type,
            "target": target.target,
            "definition_path": str(target.definition_path),
            "event_count": len(target_events),
            "capture_paths": capture_paths,
            "blob_paths": dataset_paths["blob_paths"],
            "workload_paths": dataset_paths["workload_paths"],
            "sanitized_count": dataset_paths["sanitized_count"],
            "real_inputs": real_inputs,
        })

    return {
        "summary": {
            "workloads": len(workloads),
            "skipped": len(skipped),
            "captures": sum(len(workload["capture_paths"]) for workload in workloads),
            "workload_files": sum(len(workload["blob_paths"]) for workload in workloads),
            "sanitized": sum(int(workload["sanitized_count"]) for workload in workloads),
        },
        "workloads": workloads,
        "skipped": skipped,
    }


def _target_definition(target: CollectTarget) -> dict[str, Any]:
    path = target.definition_path
    if not path.exists():
        raise ValueError(f"collect target definition_path does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"definition must be an object: {path}")
    return data


def _target_hints(target: CollectTarget, hints_dir: Path | None) -> dict[str, Any] | None:
    if hints_dir is None:
        return None
    path = hints_dir / target.op_type / f"{target.definition_name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"definition hints must be an object: {path}")
    return data


def load_capture_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if path.suffix != ".pt":
        return None, "unsupported_capture_format"
    try:
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return None, f"capture_load_failed:{type(exc).__name__}"
    if not isinstance(data, dict):
        return None, "capture_payload_not_object"
    return data, None


# ---------------------------------------------------------------------------
# Sanitizer: captured call payloads -> official-style workload entries
# ---------------------------------------------------------------------------


def _shape(value: dict[str, Any]) -> list[int] | None:
    summary = value.get("summary")
    if not isinstance(summary, dict):
        return None
    shape = summary.get("shape")
    if not isinstance(shape, list):
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def _is_full_tensor(value: dict[str, Any]) -> bool:
    summary = value.get("summary")
    return (
        isinstance(summary, dict)
        and bool(value.get("saved"))
        and "value" in value
        and isinstance(summary.get("shape"), list)
    )


def _payload_debug(payload: dict[str, Any]) -> dict[str, Any]:
    """Return lightweight capture structure for sanitize failure diagnostics."""
    args = payload.get("args")
    kwargs = payload.get("kwargs")
    debug: dict[str, Any] = {}
    if isinstance(args, list):
        debug["args"] = [
            {
                "index": index,
                "type": item.get("summary", {}).get("type") if isinstance(item.get("summary"), dict) else None,
                "shape": item.get("summary", {}).get("shape") if isinstance(item.get("summary"), dict) else None,
                "attrs": {
                    name: value.get("summary", {}) if isinstance(value, dict) else {}
                    for name, value in sorted(item.get("attrs", {}).items())
                } if isinstance(item.get("attrs"), dict) else {},
            }
            for index, item in enumerate(args)
            if isinstance(item, dict)
        ]
    if isinstance(kwargs, dict):
        debug["kwargs"] = sorted(kwargs)
    return debug


def _is_structural_tensor(value: dict[str, Any]) -> bool:
    summary = value.get("summary")
    dtype = str(summary.get("dtype", "") if isinstance(summary, dict) else "").lower()
    return any(token in dtype for token in ("int", "uint", "bool", "long"))


def _real_inputs(hints: dict[str, Any] | None) -> set[str]:
    if not isinstance(hints, dict):
        return set()
    raw = hints.get("real_inputs", [])
    if not isinstance(raw, list):
        return set()
    return {item for item in raw if isinstance(item, str) and item}


def _optional_input_missing(input_spec: dict[str, Any], captured_value: dict[str, Any] | None) -> bool:
    if not bool(input_spec.get("optional")):
        return False
    if captured_value is None:
        return True
    summary = captured_value.get("summary")
    if isinstance(summary, dict) and summary.get("type") in {"NoneType", "None"}:
        return True
    return captured_value.get("value") is None and _shape(captured_value) is None


def _const_axis(definition: dict[str, Any], axis_name: Any) -> int | None:
    if isinstance(axis_name, int):
        return axis_name
    if not isinstance(axis_name, str):
        return None
    axes = definition.get("axes")
    if not isinstance(axes, dict):
        return None
    axis = axes.get(axis_name)
    if not isinstance(axis, dict) or axis.get("type") != "const":
        return None
    value = axis.get("value")
    return int(value) if type(value) is int and value > 0 else None


def _shape_known_without_capture(
    shape_template: Any,
    *,
    definition: dict[str, Any],
    axes: dict[str, int],
) -> bool:
    if not isinstance(shape_template, list):
        return False
    for dim in shape_template:
        if isinstance(dim, int):
            continue
        if isinstance(dim, str) and dim in axes:
            continue
        if _const_axis(definition, dim) is not None:
            continue
        return False
    return True


def _tuple_tensor_summary(value: dict[str, Any], index: int) -> dict[str, Any] | None:
    summary = value.get("summary")
    if not isinstance(summary, dict) or summary.get("type") != "tuple":
        return None
    elements = summary.get("elements")
    if not isinstance(elements, list) or index >= len(elements):
        return None
    element = elements[index]
    if not isinstance(element, dict) or not isinstance(element.get("shape"), list):
        return None
    return {
        "saved": bool(value.get("saved")),
        "summary": element,
    }


class _HintExecutor:
    """Execute definition_hints against one captured call payload."""

    def __init__(
        self,
        *,
        payload: dict[str, Any],
        input_names: list[str],
        definition_inputs: dict[str, Any],
        definition: dict[str, Any],
        hints: dict[str, Any] | None,
    ) -> None:
        self.payload = payload
        self.input_names = input_names
        self.definition_inputs = definition_inputs
        self.definition = definition
        self.hints = hints if isinstance(hints, dict) else {}

    def lookup_input(self, input_name: str, input_spec: dict[str, Any]) -> dict[str, Any] | None:
        hinted = self._lookup_from_hints(input_name, input_spec)
        if hinted is not None:
            return hinted
        if self._has_input_hints(input_name):
            return None
        return self._lookup_by_convention(input_name, input_spec)

    def axis_value_from_input(
        self,
        value: dict[str, Any],
        shape_template: list[Any],
        axis_name: str,
        *,
        input_name: str,
    ) -> int | None:
        shape = _shape(value)
        if shape is None:
            return None
        dim_index = shape_template.index(axis_name)
        squeezed_axes = self._squeezed_axes(input_name, shape_template)
        if squeezed_axes:
            if len(shape) != len(shape_template) - len(squeezed_axes):
                return None
            if axis_name in squeezed_axes:
                squeezed_value = self._const_axis(axis_name)
                return int(squeezed_value) if squeezed_value else None
            dim_index -= sum(1 for item in squeezed_axes if shape_template.index(item) < dim_index)
        if dim_index >= len(shape):
            return None
        return int(shape[dim_index])

    def inferred_axis_value(self, axis_name: str, axes: dict[str, int]) -> int | None:
        rule = self._axis_rules().get(axis_name)
        if not isinstance(rule, dict):
            return None
        input_name = rule.get("input")
        if not isinstance(input_name, str):
            return None
        input_spec = self.definition_inputs.get(input_name)
        if not isinstance(input_spec, dict):
            return None
        tensor = self._tensor_value(self.lookup_input(input_name, input_spec))
        return apply_axis_hint_rule(rule, tensor, axes)

    def slice_limit_axis(self, input_name: str) -> str | None:
        raw = self.hints.get("tensor_slices")
        if not isinstance(raw, dict):
            return None
        item = raw.get(input_name)
        if not isinstance(item, dict):
            return None
        limit_axis = item.get("limit_axis")
        return limit_axis if isinstance(limit_axis, str) else None

    def _shape_rank_matches(self, value: dict[str, Any] | None, shape_template: Any, *, input_name: str) -> bool:
        if value is None:
            return False
        if not isinstance(shape_template, list):
            return True
        if shape_template == []:
            return "value" in value
        shape = _shape(value)
        if shape is None:
            return False
        if len(shape) == len(shape_template):
            return True
        squeezed_axes = self._squeezed_axes(input_name, shape_template)
        return bool(squeezed_axes) and len(shape) == len(shape_template) - len(squeezed_axes)

    def _lookup_from_hints(self, input_name: str, input_spec: dict[str, Any]) -> dict[str, Any] | None:
        rules = self._input_rules(input_name)
        if not isinstance(rules, list):
            return None
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            candidate = self._rule_value(rule, input_name=input_name)
            if isinstance(candidate, dict) and self._shape_rank_matches(
                candidate,
                input_spec.get("shape"),
                input_name=input_name,
            ):
                return candidate
            if isinstance(candidate, dict) and _shape(candidate) is None:
                return candidate
        return None

    def _lookup_by_convention(self, input_name: str, input_spec: dict[str, Any]) -> dict[str, Any] | None:
        shape_template = input_spec.get("shape")
        kwargs = self.payload.get("kwargs")
        if not isinstance(kwargs, dict):
            return None
        candidate = kwargs.get(input_name)
        if not isinstance(candidate, dict):
            return None
        if shape_template is None or _is_full_tensor(candidate):
            return candidate
        if self._shape_rank_matches(candidate, shape_template, input_name=input_name):
            return candidate
        return None

    def _rule_value(self, rule: dict[str, Any], *, input_name: str) -> dict[str, Any] | None:
        source = rule.get("source")
        kwargs = self.payload.get("kwargs")
        args = self.payload.get("args")
        if source == "kwarg" and isinstance(kwargs, dict):
            name = rule.get("name")
            value = kwargs.get(name if isinstance(name, str) else input_name)
            return value if isinstance(value, dict) else None
        if source in {"arg", "arg_tuple"} and isinstance(args, list):
            arg_index = rule.get("arg_index")
            if type(arg_index) is not int or arg_index < 0 or arg_index >= len(args):
                return None
            value = args[arg_index]
            if source == "arg":
                return value if isinstance(value, dict) else None
            tuple_index = rule.get("tuple_index")
            if type(tuple_index) is not int or tuple_index < 0 or not isinstance(value, dict):
                return None
            return _tuple_tensor_summary(value, tuple_index)
        if source == "attr" and isinstance(args, list):
            pattern = rule.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                return None
            pattern = pattern.lower()
            for arg in args:
                if not isinstance(arg, dict):
                    continue
                attrs = arg.get("attrs")
                if not isinstance(attrs, dict):
                    continue
                for attr_name, attr_value in attrs.items():
                    if pattern in str(attr_name).strip("_").lower() and isinstance(attr_value, dict):
                        return attr_value
        return None

    def _squeezed_axes(self, input_name: str, shape_template: list[Any]) -> list[str]:
        raw_axes = self._shape_override(input_name, "squeezed_axes")
        if not isinstance(raw_axes, list):
            return []
        axes = [axis for axis in raw_axes if isinstance(axis, str) and axis in shape_template]
        return list(dict.fromkeys(axes))

    def _shape_override(self, input_name: str, key: str) -> Any:
        raw = self.hints.get("shape_overrides")
        if not isinstance(raw, dict):
            return None
        item = raw.get(input_name)
        if not isinstance(item, dict):
            return None
        return item.get(key)

    def _input_rules(self, input_name: str) -> Any:
        raw_inputs = self.hints.get("inputs")
        return raw_inputs.get(input_name) if isinstance(raw_inputs, dict) else None

    def _has_input_hints(self, input_name: str) -> bool:
        return self._input_rules(input_name) is not None

    def _axis_rules(self) -> dict[str, Any]:
        raw = self.hints.get("axes")
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _tensor_value(value: dict[str, Any] | None) -> Any:
        if not isinstance(value, dict) or not _is_full_tensor(value):
            return None
        return value.get("value")

    def _const_axis(self, name: str) -> int | None:
        axes = self.definition.get("axes")
        if not isinstance(axes, dict):
            return None
        axis = axes.get(name)
        if not isinstance(axis, dict) or axis.get("type") != "const":
            return None
        try:
            return int(axis.get("value"))
        except (TypeError, ValueError):
            return None


def infer_axes_from_capture(
    captured: dict[str, Any],
    definition: dict[str, Any],
    *,
    hints: dict[str, Any] | None = None,
) -> tuple[dict[str, int] | None, str | None]:
    """Infer variable axes and reject const-axis mismatches."""
    axes: dict[str, int] = {}
    definition_inputs = definition.get("inputs")
    definition_axes = definition.get("axes")
    if not isinstance(definition_inputs, dict) or not definition_inputs:
        return None, "definition_has_no_inputs"
    if not isinstance(definition_axes, dict):
        definition_axes = {}

    input_names = list(definition_inputs)
    payload = captured.get("payload")
    if not isinstance(payload, dict):
        return None, "capture_has_no_payload"
    payload = dict(payload)
    payload["_definition_inputs"] = definition_inputs
    hint_executor = _HintExecutor(
        payload=payload,
        input_names=input_names,
        definition_inputs=definition_inputs,
        definition=definition,
        hints=hints,
    )

    for axis_name, axis_spec in definition_axes.items():
        if not isinstance(axis_spec, dict):
            continue
        axis_type = axis_spec.get("type")
        expected = axis_spec.get("value") if axis_type == "const" else None
        value: int | None = None

        for input_name, input_spec in definition_inputs.items():
            if not isinstance(input_spec, dict):
                continue
            shape_template = input_spec.get("shape")
            if not isinstance(shape_template, list) or axis_name not in shape_template:
                continue
            captured_value = hint_executor.lookup_input(input_name, input_spec)
            if captured_value is None:
                continue
            actual = hint_executor.axis_value_from_input(
                captured_value,
                shape_template,
                axis_name,
                input_name=input_name,
            )
            if actual is None:
                continue
            if expected is not None and int(expected) != actual:
                return None, f"const_axis_mismatch:{axis_name}:{actual}!={expected}"
            value = actual
            break

        if axis_type == "var" and value is not None:
            axes[axis_name] = value

    for axis_name in hint_executor._axis_rules():
        if axis_name not in definition_axes:
            continue
        value = hint_executor.inferred_axis_value(axis_name, axes)
        if value is not None:
            axes[axis_name] = value

    return axes, None


def build_sanitized_workload_entry(
    *,
    captured: dict[str, Any],
    definition: dict[str, Any],
    hints: dict[str, Any] | None = None,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build one workload entry and write referenced safetensors payloads."""
    axes, reason = infer_axes_from_capture(captured, definition, hints=hints)
    if axes is None:
        return None, {"reason": reason or "axes_unresolved"}

    definition_inputs = definition.get("inputs")
    if not isinstance(definition_inputs, dict) or not definition_inputs:
        return None, {"reason": "definition_has_no_inputs"}

    payload = captured.get("payload")
    if not isinstance(payload, dict):
        return None, {"reason": "capture_has_no_payload"}
    payload = dict(payload)
    payload["_definition_inputs"] = definition_inputs

    input_names = list(definition_inputs)
    op_type = str(definition.get("op_type") or captured.get("op_type") or "unknown")
    definition_name = str(definition.get("name") or captured.get("definition_name"))
    workload_inputs: dict[str, Any] = {}
    pending_tensors: dict[str, Any] = {}
    hint_executor = _HintExecutor(
        payload=payload,
        input_names=input_names,
        definition_inputs=definition_inputs,
        definition=definition,
        hints=hints,
    )
    real_inputs = _real_inputs(hints)

    for input_name, input_spec in definition_inputs.items():
        if not isinstance(input_spec, dict):
            continue
        shape_template = input_spec.get("shape")
        captured_value = hint_executor.lookup_input(input_name, input_spec)

        if _optional_input_missing(input_spec, captured_value):
            continue

        if shape_template is None or shape_template == []:
            # Scalars (e.g. sm_scale) arrive either on the run call or merged in
            # from a reviewed companion method; both land in kwargs by name and
            # are found by _lookup_input. Core does not fabricate a default.
            scalar = captured_value.get("value") if isinstance(captured_value, dict) else None
            if scalar is None:
                return None, {"reason": f"missing_scalar:{input_name}"}
            workload_inputs[input_name] = {"type": "scalar", "value": scalar}
            continue

        needs_real_tensor = input_name in real_inputs
        if (
            isinstance(captured_value, dict)
            and _is_full_tensor(captured_value)
            and (_is_structural_tensor(captured_value) or needs_real_tensor)
        ):
            tensor = captured_value.get("value")
            if not hasattr(tensor, "contiguous"):
                return None, {"reason": f"invalid_tensor:{input_name}"}
            limit_axis = hint_executor.slice_limit_axis(input_name)
            if limit_axis in axes:
                tensor = tensor.reshape(-1)[: int(axes[limit_axis])]
            pending_tensors[input_name] = tensor.contiguous().clone()
            workload_inputs[input_name] = {
                "type": "safetensors",
                "path": "",
                "tensor_key": input_name,
            }
            continue

        if needs_real_tensor:
            return None, {"reason": f"missing_real_tensor:{input_name}"}
        if not isinstance(captured_value, dict) or _shape(captured_value) is None:
            if isinstance(captured_value, dict) and _shape_known_without_capture(
                shape_template,
                definition=definition,
                axes=axes,
            ):
                workload_inputs[input_name] = {"type": "random"}
                continue
            return None, {
                "reason": f"missing_tensor_summary:{input_name}",
                "payload_debug": _payload_debug(payload),
            }
        workload_inputs[input_name] = {"type": "random"}

    if not workload_inputs:
        return None, {"reason": "no_inputs"}

    workload_id = captured.get("_workload_uuid")
    if not isinstance(workload_id, str) or not workload_id:
        workload_id = str(uuid.uuid4())
    if pending_tensors:
        filename = f"{definition_name}_{workload_id}.safetensors"
        rel_path = f"./blob/workloads/{op_type}/{definition_name}/{filename}"
        blob_dir = output_dir / "blob" / "workloads" / op_type / definition_name
        blob_dir.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file

        save_file(pending_tensors, str(blob_dir / filename))
        for input_name in pending_tensors:
            workload_inputs[input_name]["path"] = rel_path

    return {
        "definition": definition_name,
        "evaluation": None,
        "solution": None,
        "workload": {
            "uuid": workload_id,
            "axes": axes,
            "inputs": workload_inputs,
        },
    }, {}
