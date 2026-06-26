"""Audit and repair fitrace definitions before workload collect.

This is a pre-collect gate: fitrace output is useful evidence, but definitions
whose semantic axes contradict the captured call should not enter collect
unchanged.
"""

from __future__ import annotations

import ast
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from flashinfer_trace.definition_repairs import DEFINITION_REPAIRERS, DefinitionRepairer
from flashinfer_trace.core.events import load_capture_payload, load_jsonl


_ALLOWED_DTYPES = {
    "float32",
    "float16",
    "bfloat16",
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1",
    "int64",
    "int32",
    "int16",
    "int8",
    "bool",
}


def audit_and_repair_definitions(
    *,
    raw_definitions_dir: Path,
    output_definitions_dir: Path,
    output_hints_dir: Path | None = None,
    events_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    """Stage audited definitions into ``output_definitions_dir``.

    Invalid-but-repairable fitrace definitions are rewritten conservatively and
    marked in the report. Invalid and unrepaired definitions are quarantined.
    """
    events = _events_with_capture_payload(load_jsonl(events_path) if events_path.exists() else [])
    if output_definitions_dir.exists():
        shutil.rmtree(output_definitions_dir)
    output_definitions_dir.mkdir(parents=True, exist_ok=True)
    if output_hints_dir is not None:
        if output_hints_dir.exists():
            shutil.rmtree(output_hints_dir)
        output_hints_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = report_dir / "rejected_definitions"
    if rejected_dir.exists():
        shutil.rmtree(rejected_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    passed: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    hints_written = 0

    for src in sorted(raw_definitions_dir.rglob("*.json")):
        try:
            definition = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rejected.append({"source": str(src), "reason": f"invalid_json:{exc}"})
            continue
        if not isinstance(definition, dict):
            rejected.append({"source": str(src), "reason": "definition_not_object"})
            continue
        repairer, issue = _definition_issue(definition)
        if issue is None:
            hints = _definition_hints(definition, events)
            definition, fixes, schema_issue = prepare_definition_for_output(definition, events=events, hints=hints)
            if schema_issue is not None:
                rejected.append({"source": str(src), "name": definition.get("name"), "reason": schema_issue})
                _write_rejected_source(rejected_dir, raw_definitions_dir, src, definition)
                continue
            dst = _write_definition(output_definitions_dir, definition)
            hints_path = _write_definition_hints(output_hints_dir, definition, hints)
            hints_written += int(hints_path is not None)
            passed.append({
                "name": definition.get("name"),
                "path": str(dst),
                "source": str(src),
                "hints_path": str(hints_path) if hints_path else None,
                "fixes": fixes,
            })
            continue

        assert repairer is not None
        repaired_definition, repair_reason = repairer.repair(definition, events)
        if repaired_definition is not None:
            hints = _definition_hints(repaired_definition, events)
            repaired_definition, fixes, schema_issue = prepare_definition_for_output(
                repaired_definition,
                events=events,
                hints=hints,
            )
            if schema_issue is not None:
                rejected.append({"source": str(src), "name": definition.get("name"), "reason": schema_issue})
                _write_rejected_source(rejected_dir, raw_definitions_dir, src, definition)
                continue
            dst = _write_definition(output_definitions_dir, repaired_definition)
            hints_path = _write_definition_hints(output_hints_dir, repaired_definition, hints)
            hints_written += int(hints_path is not None)
            repaired.append(
                {
                    "source_name": definition.get("name"),
                    "name": repaired_definition.get("name"),
                    "path": str(dst),
                    "reason": issue,
                    "repair": repair_reason,
                    "hints_path": str(hints_path) if hints_path else None,
                    "fixes": fixes,
                }
            )
            _write_rejected_source(rejected_dir, raw_definitions_dir, src, definition)
            continue

        rejected.append({"source": str(src), "name": definition.get("name"), "reason": issue})
        _write_rejected_source(rejected_dir, raw_definitions_dir, src, definition)

    report = {
        "summary": {
            "raw": len(list(raw_definitions_dir.rglob("*.json"))) if raw_definitions_dir.exists() else 0,
            "passed": len(passed),
            "repaired": len(repaired),
            "rejected": len(rejected),
            "accepted": len(passed) + len(repaired),
            "hints": hints_written,
            "ok": not rejected,
        },
        "aliases": {
            str(item["source_name"]): str(item["name"])
            for item in repaired
            if isinstance(item.get("source_name"), str) and isinstance(item.get("name"), str)
        },
        "passed": passed,
        "repaired": repaired,
        "rejected": rejected,
    }
    return report


def prepare_definition_for_output(
    definition: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
    hints: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str], str | None]:
    """Return an official-schema-ready definition or a blocking issue."""
    prepared = deepcopy(definition)
    fixes: list[str] = []
    repairer, issue = _definition_issue(prepared)
    if issue is not None:
        if repairer is None:
            return prepared, fixes, issue
        repaired, repair_reason = repairer.repair(prepared, events or [])
        if repaired is None:
            return prepared, fixes, issue
        prepared = repaired
        fixes.append(f"definition_repair:{repairer.name}:{repair_reason}")
    reference_fix = _ensure_reference_run(prepared)
    if reference_fix:
        fixes.append(reference_fix)
    fixes.extend(_fill_input_dtypes(prepared, events or [], hints))
    issue = _official_schema_issue(prepared)
    return prepared, fixes, issue


def _definition_issue(definition: dict[str, Any]) -> tuple[DefinitionRepairer | None, str | None]:
    for repairer in DEFINITION_REPAIRERS:
        issue = repairer.issue(definition)
        if issue:
            return repairer, issue
    return None, None


def _events_with_capture_payload(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for event in events:
        item = dict(event)
        raw_path = event.get("capture_path")
        if isinstance(raw_path, str) and raw_path:
            captured, _ = load_capture_payload(Path(raw_path))
            payload = captured.get("payload") if isinstance(captured, dict) else None
            if isinstance(payload, dict):
                for key in ("args", "kwargs"):
                    value = payload.get(key)
                    if value is not None:
                        item[key] = value
        enriched.append(item)
    return enriched


def _write_definition(root: Path, definition: dict[str, Any]) -> Path:
    name = _required_definition_field(definition, "name")
    op_type = _required_definition_field(definition, "op_type")
    dst = root / op_type / f"{name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst


def _definition_hints(definition: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for repairer in DEFINITION_REPAIRERS:
        if repairer.hints is None:
            continue
        hints = repairer.hints(definition, events)
        if hints:
            return hints
    return None


def _write_definition_hints(root: Path | None, definition: dict[str, Any], hints: dict[str, Any] | None) -> Path | None:
    if root is None:
        return None
    if not hints:
        return None
    name = _required_definition_field(definition, "name")
    op_type = _required_definition_field(definition, "op_type")
    dst = root / op_type / f"{name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(hints, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst


def _required_definition_field(definition: dict[str, Any], field: str) -> str:
    value = definition.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"definition has invalid {field}: {value!r}")
    return value.strip()


def _write_rejected_source(root: Path, raw_root: Path, source: Path, definition: dict[str, Any]) -> None:
    dst = root / source.relative_to(raw_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ensure_reference_run(definition: dict[str, Any]) -> str | None:
    reference = definition.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        return None
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return None
    if _top_level_function(tree, "run") is not None:
        return None
    target = _reference_target_function(tree)
    if target is None:
        return None
    params = ast.get_source_segment(reference, target.args) or _args_source(target.args)
    call = _call_args_source(target.args)
    definition["reference"] = f"{reference.rstrip()}\n\n\ndef run({params}):\n    return {target.name}({call})\n"
    return f"wrapped_reference:{target.name}->run"


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _reference_target_function(tree: ast.Module) -> ast.FunctionDef | None:
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    candidates = [node for node in functions if node.name.endswith("_reference")]
    if len(candidates) == 1:
        return candidates[0]
    return functions[0] if len(functions) == 1 else None


def _args_source(args: ast.arguments) -> str:
    names = [arg.arg for arg in [*args.posonlyargs, *args.args]]
    if args.vararg is not None:
        names.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        names.append("*")
    names.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(f"**{args.kwarg.arg}")
    return ", ".join(names)


def _call_args_source(args: ast.arguments) -> str:
    parts = [arg.arg for arg in [*args.posonlyargs, *args.args]]
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    parts.extend(f"{arg.arg}={arg.arg}" for arg in args.kwonlyargs)
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _fill_input_dtypes(
    definition: dict[str, Any],
    events: list[dict[str, Any]],
    hints: dict[str, Any] | None,
) -> list[str]:
    inputs = definition.get("inputs")
    if not isinstance(inputs, dict):
        return []
    fixes: list[str] = []
    for input_name, input_spec in inputs.items():
        if not isinstance(input_name, str) or not isinstance(input_spec, dict):
            continue
        dtype = _normalize_dtype(input_spec.get("dtype"))
        if dtype is not None:
            if dtype != input_spec.get("dtype"):
                input_spec["dtype"] = dtype
                fixes.append(f"normalized_dtype:{input_name}:{dtype}")
            continue
        inferred = _infer_input_dtype(definition, events, hints, input_name)
        if inferred is not None:
            input_spec["dtype"] = inferred
            fixes.append(f"inferred_dtype:{input_name}:{inferred}")
    return fixes


def _normalize_dtype(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().lower()
    if text.startswith("torch."):
        text = text.removeprefix("torch.")
    if text in {"long"}:
        text = "int64"
    if text in _ALLOWED_DTYPES:
        return text
    return None


def _infer_input_dtype(
    definition: dict[str, Any],
    events: list[dict[str, Any]],
    hints: dict[str, Any] | None,
    input_name: str,
) -> str | None:
    for event in events:
        if not _event_matches_definition(definition, event):
            continue
        value = _lookup_event_input(event, hints, input_name)
        dtype = _capture_dtype(value)
        if dtype is not None:
            return dtype
    return None


def _event_matches_definition(definition: dict[str, Any], event: dict[str, Any]) -> bool:
    name = definition.get("name")
    if isinstance(name, str) and event.get("definition_name") == name:
        return True
    audit = definition.get("audit")
    event_name = audit.get("event_name") if isinstance(audit, dict) else None
    if isinstance(event_name, str) and event.get("name") == event_name:
        return True
    target = _definition_tag_value(definition, "fi_api:")
    if target and event.get("target") != target:
        return False
    stage = _definition_tag_value(definition, "stage:")
    if stage and event.get("variant") != stage:
        return False
    return bool(target or stage)


def _definition_tag_value(definition: dict[str, Any], prefix: str) -> str | None:
    tags = definition.get("tags")
    if not isinstance(tags, list):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(prefix):
            return tag.removeprefix(prefix)
    return None


def _lookup_event_input(event: dict[str, Any], hints: dict[str, Any] | None, input_name: str) -> dict[str, Any] | None:
    rules = hints.get("inputs", {}).get(input_name) if isinstance(hints, dict) and isinstance(hints.get("inputs"), dict) else None
    if isinstance(rules, list):
        for rule in rules:
            value = _lookup_event_input_rule(event, rule, input_name)
            if value is not None:
                return value
    kwargs = event.get("kwargs")
    value = kwargs.get(input_name) if isinstance(kwargs, dict) else None
    return value if isinstance(value, dict) else None


def _lookup_event_input_rule(event: dict[str, Any], rule: Any, input_name: str) -> dict[str, Any] | None:
    if not isinstance(rule, dict):
        return None
    source = rule.get("source")
    kwargs = event.get("kwargs")
    args = event.get("args")
    if source == "kwarg" and isinstance(kwargs, dict):
        key = rule.get("name")
        value = kwargs.get(key if isinstance(key, str) else input_name)
        return value if isinstance(value, dict) else None
    if source in {"arg", "arg_tuple"} and isinstance(args, list):
        index = rule.get("arg_index")
        if type(index) is not int or index < 0 or index >= len(args):
            return None
        value = args[index]
        if source == "arg":
            return value if isinstance(value, dict) else None
        tuple_index = rule.get("tuple_index")
        return _tuple_tensor_summary(value, tuple_index) if type(tuple_index) is int else None
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


def _tuple_tensor_summary(value: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary = value.get("summary")
    if not isinstance(summary, dict) or summary.get("type") != "tuple":
        return None
    elements = summary.get("elements")
    if not isinstance(elements, list) or index < 0 or index >= len(elements):
        return None
    item = elements[index]
    return {"summary": item} if isinstance(item, dict) else None


def _capture_dtype(value: dict[str, Any] | None) -> str | None:
    if not isinstance(value, dict):
        return None
    summary = value.get("summary")
    if not isinstance(summary, dict):
        return None
    return _normalize_dtype(summary.get("dtype"))


def _official_schema_issue(definition: dict[str, Any]) -> str | None:
    reference = definition.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        return "missing_reference"
    try:
        tree = ast.parse(reference)
    except SyntaxError as exc:
        return f"invalid_reference:{exc.msg}"
    if _top_level_function(tree, "run") is None:
        return "reference_missing_run"
    for section in ("inputs", "outputs"):
        values = definition.get(section)
        if not isinstance(values, dict):
            return f"{section}_not_object"
        for name, spec in values.items():
            if not isinstance(spec, dict):
                return f"{section}.{name}_not_object"
            if _normalize_dtype(spec.get("dtype")) is None:
                return f"{section}.{name}.invalid_dtype:{spec.get('dtype')!r}"
    return None
