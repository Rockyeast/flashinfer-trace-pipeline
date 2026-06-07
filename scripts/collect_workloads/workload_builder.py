"""Convert captured hook records into workload JSONL and safetensors blobs."""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from collect_workloads.hook_specs import extract_hook_api, should_store_tensor
from artifact_schemas import validate_definition, validate_workload_entry


def infer_axes(captured: dict, definition: dict) -> dict | None:
    """Infer variable axes from captured tensor shapes and validate const axes."""
    axes: dict[str, int] = {}
    inp_names = list(definition["inputs"].keys())

    def _lookup(inp_name: str) -> dict | None:
        pos_idx = inp_names.index(inp_name) if inp_name in inp_names else -1
        return (
            captured.get(f"kwarg_{inp_name}")
            or (captured.get(f"arg_{pos_idx}") if pos_idx >= 0 else None)
        )

    for axis_name, axis_def in definition["axes"].items():
        if axis_def["type"] == "const":
            expected = axis_def.get("value")
            if expected is None:
                continue
            for inp_name, inp_spec in definition["inputs"].items():
                shape_tmpl = inp_spec.get("shape")
                if not shape_tmpl or axis_name not in shape_tmpl:
                    continue
                dim_idx = shape_tmpl.index(axis_name)
                summary = _lookup(inp_name)
                if summary is None:
                    continue
                if summary["type"] == "full":
                    tensor = summary["tensor"]
                    actual = tensor.shape[dim_idx] if dim_idx < len(tensor.shape) else None
                elif summary["type"] == "shape":
                    shape = summary["shape"]
                    actual = shape[dim_idx] if dim_idx < len(shape) else None
                else:
                    continue
                if actual is not None and actual != expected:
                    return None
            continue

        value = None
        for inp_name, inp_spec in definition["inputs"].items():
            shape_tmpl = inp_spec.get("shape")
            if not shape_tmpl or axis_name not in shape_tmpl:
                continue
            dim_idx = shape_tmpl.index(axis_name)
            summary = _lookup(inp_name)
            if summary is None:
                continue
            if summary["type"] == "full":
                tensor = summary["tensor"]
                if dim_idx < len(tensor.shape):
                    value = int(tensor.shape[dim_idx])
                    break
            elif summary["type"] == "shape":
                shape = summary["shape"]
                if dim_idx < len(shape):
                    value = int(shape[dim_idx])
                    break
        if value is not None:
            axes[axis_name] = value

    if "num_pages" in definition["axes"] and "num_pages" not in axes:
        kv_idx_summary = captured.get("kwarg_kv_indices")
        if kv_idx_summary and kv_idx_summary["type"] == "full":
            tensor = kv_idx_summary["tensor"]
            if tensor.numel() > 0:
                axes["num_pages"] = int(tensor.max().item()) + 1

    return axes


def build_workload_entry(
    captured: dict,
    definition: dict,
    reject_reason: list[str] | None = None,
) -> dict | None:
    """Build one workload candidate from a captured hook record."""
    import torch

    axes = infer_axes(captured, definition)
    if axes is None:
        if reject_reason is not None:
            reject_reason.append("axes_const_mismatch_or_unresolved")
        return None

    def_name = definition["name"]
    op_type = definition.get("op_type", "unknown")
    workload_uuid = str(uuid.uuid4())
    inp_names = list(definition["inputs"].keys())

    def _lookup_captured(inp_name: str) -> dict | None:
        pos_idx = inp_names.index(inp_name) if inp_name in inp_names else -1
        return (
            captured.get(f"kwarg_{inp_name}")
            or (captured.get(f"arg_{pos_idx}") if pos_idx >= 0 else None)
        )

    workload_inputs: dict = {}
    pending_tensors: dict = {}

    for inp_name, inp_spec in definition["inputs"].items():
        shape_tmpl = inp_spec.get("shape")
        dtype = inp_spec.get("dtype", "float32")

        if shape_tmpl is None:
            summary = _lookup_captured(inp_name) or captured.get("kwarg_sm_scale")
            value = summary["value"] if summary and summary["type"] == "scalar" else 0.08838834764831843
            workload_inputs[inp_name] = {"type": "scalar", "value": float(value)}
            continue

        if not should_store_tensor(op_type, inp_name, dtype):
            workload_inputs[inp_name] = {"type": "random"}
            continue

        summary = _lookup_captured(inp_name)
        if summary is None or summary["type"] != "full":
            if reject_reason is not None:
                reject_reason.append(f"missing_full_tensor:{inp_name}")
            return None
        tensor = summary["tensor"]
        if not isinstance(tensor, torch.Tensor):
            if reject_reason is not None:
                reject_reason.append(f"invalid_tensor:{inp_name}")
            return None

        pending_tensors[inp_name] = tensor.contiguous().clone()
        workload_inputs[inp_name] = {
            "type": "pending_safetensors",
            "tensor_key": inp_name,
        }

    if not workload_inputs:
        if reject_reason is not None:
            reject_reason.append("no_inputs")
        return None

    return {
        "definition": def_name,
        "solution": None,
        "workload": {
            "uuid": workload_uuid,
            "axes": axes,
            "inputs": workload_inputs,
        },
        "evaluation": None,
        "_op_type": op_type,
        "_pending_tensors": pending_tensors,
    }


def finalize_entry(entry: dict, trace_dir: Path) -> dict:
    """Write pending tensors to safetensors and return a JSONL-ready entry."""
    from safetensors.torch import save_file

    pending = entry.pop("_pending_tensors", {})
    op_type = entry.pop("_op_type", "unknown")
    def_name = entry["definition"]

    if pending:
        workload_uuid = entry["workload"]["uuid"]
        fname = f"{def_name}_{workload_uuid}.safetensors"
        rel_path = f"./blob/workloads/{op_type}/{def_name}/{fname}"
        blob_dir = trace_dir / "blob" / "workloads" / op_type / def_name
        blob_dir.mkdir(parents=True, exist_ok=True)
        save_file(pending, str(blob_dir / fname))

        for inp_name in pending:
            entry["workload"]["inputs"][inp_name] = {
                "type": "safetensors",
                "path": rel_path,
                "tensor_key": inp_name,
            }
    else:
        for inp_name, value in entry["workload"]["inputs"].items():
            if isinstance(value, dict) and value.get("type") == "pending_safetensors":
                entry["workload"]["inputs"][inp_name] = {"type": "random"}

    validate_workload_entry(entry)
    return entry


def select_diverse(entries: list[dict], definition: dict, max_count: int = 20) -> list[dict]:
    """Select a diverse subset of workload entries across variable axes."""
    if len(entries) <= max_count:
        return entries

    var_axes = [name for name, spec in definition["axes"].items() if spec.get("type") == "var"]
    if not var_axes:
        return entries[:max_count]

    def _vec(entry: dict) -> list[float]:
        axes = entry["workload"]["axes"]
        return [float(axes.get(axis, 0)) for axis in var_axes]

    def _min_dist(value: list[float], selected: list[list[float]]) -> float:
        best = float("inf")
        for selected_value in selected:
            norm = max(max(abs(a), abs(b), 1.0) for a, b in zip(value, selected_value))
            dist = sum(((a - b) / norm) ** 2 for a, b in zip(value, selected_value)) ** 0.5
            best = min(best, dist)
        return best

    by_bs: dict = defaultdict(list)
    for entry in entries:
        by_bs[entry["workload"]["axes"].get("batch_size", 0)].append(entry)

    selected_entries: list[dict] = []
    selected_vecs: list[list[float]] = []
    remaining = list(entries)

    for batch_size in sorted(by_bs):
        if len(selected_entries) >= max_count:
            break
        group = by_bs[batch_size]
        if not selected_vecs:
            pick = group[0]
        else:
            pick = max(group, key=lambda entry: _min_dist(_vec(entry), selected_vecs))
        selected_entries.append(pick)
        selected_vecs.append(_vec(pick))
        remaining.remove(pick)

    while len(selected_entries) < max_count and remaining:
        best = max(range(len(remaining)), key=lambda idx: _min_dist(_vec(remaining[idx]), selected_vecs))
        selected_entries.append(remaining[best])
        selected_vecs.append(_vec(remaining[best]))
        remaining.pop(best)

    return selected_entries


def process_captures(
    capture_dir: Path,
    def_files: list[Path],
    trace_dir: Path,
    *,
    replace: bool = False,
    max_entries: int = 20,
    max_dups_per_axes: int = 2,
    log: Callable[[str], None] = print,
) -> dict[str, object]:
    """Read .pt captures, match definitions, dedupe, and write workloads."""
    import torch

    definitions: dict[str, dict] = {}
    api_to_defs: dict[str, list[str]] = defaultdict(list)
    for path in def_files:
        defn = json.loads(path.read_text())
        validate_definition(defn)
        hook_api = extract_hook_api(defn)
        if not hook_api:
            continue
        definitions[defn["name"]] = defn
        api_to_defs[hook_api].append(defn["name"])

    capture_files = sorted(capture_dir.glob("*.pt"))
    log(f"Processing {len(capture_files)} capture files from {capture_dir}")

    captures_by_fi_api: dict[str, list[dict]] = defaultdict(list)
    capture_summary: dict[str, dict] = {}
    for pt_path in capture_files:
        try:
            record = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            fi_api = record.get("fi_api", "")
            if fi_api:
                call_type = record.get("call_type", "unknown")
                item = capture_summary.setdefault(
                    fi_api,
                    {"records": 0, "call_types": defaultdict(int)},
                )
                item["records"] += 1
                item["call_types"][call_type] += 1
            if fi_api in api_to_defs:
                if record.get("_incomplete"):
                    log(f"  SKIPPED incomplete capture: {pt_path.name} ({fi_api})")
                    continue
                captures_by_fi_api[fi_api].append(record)
        except Exception as exc:
            log(f"  WARNING: could not load {pt_path.name}: {exc}")
    for item in capture_summary.values():
        item["call_types"] = dict(item["call_types"])

    candidates: dict[str, list[dict]] = defaultdict(list)
    seen_axes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    discarded_counts: dict[str, int] = defaultdict(int)
    discard_reasons: dict[str, Counter] = defaultdict(Counter)
    def_name_to_fi_api: dict[str, str] = {}

    for fi_api, records in captures_by_fi_api.items():
        def_names = api_to_defs.get(fi_api, [])
        for def_name in def_names:
            def_name_to_fi_api[def_name] = fi_api
        for record in records:
            captured = record.get("captured", {})
            for def_name in def_names:
                defn = definitions[def_name]
                reason: list[str] = []
                entry = build_workload_entry(captured, defn, reject_reason=reason)
                if entry is None:
                    discarded_counts[def_name] += 1
                    discard_reasons[def_name][reason[0] if reason else "unknown"] += 1
                    continue
                axes_key = json.dumps(entry["workload"]["axes"], sort_keys=True)
                if seen_axes[def_name][axes_key] < max_dups_per_axes:
                    seen_axes[def_name][axes_key] += 1
                    candidates[def_name].append(entry)

    results: dict[str, int] = {}
    diagnostics: dict[str, dict] = {
        "_capture_summary": {
            "total_capture_files": len(capture_files),
            "expected_fi_apis": {
                fi_api: list(def_names)
                for fi_api, def_names in api_to_defs.items()
            },
            "captured_fi_apis": capture_summary,
            "matched_capture_counts": {
                fi_api: len(records)
                for fi_api, records in captures_by_fi_api.items()
            },
        }
    }

    for def_name, entries in candidates.items():
        defn = definitions[def_name]
        if max_entries and max_entries > 0:
            selected = select_diverse(entries, defn, max_entries)
        else:
            selected = entries
        selected = [finalize_entry(entry, trace_dir) for entry in selected]
        op_type = defn.get("op_type", "unknown")

        out_dir = trace_dir / "workloads" / op_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{def_name}.jsonl"

        mode = "w" if replace or not out_path.exists() else "a"
        action = (
            "Replaced" if (replace and out_path.exists())
            else "Appended" if (not replace and out_path.exists())
            else "Created"
        )

        with open(out_path, mode) as file:
            for entry in selected:
                file.write(json.dumps(entry) + "\n")

        log(f"{action} {out_path}: {len(selected)}/{len(entries)} workloads for {def_name}")
        results[def_name] = len(selected)
        diagnostics[def_name] = {
            "candidates": len(entries),
            "selected": len(selected),
            "discarded": discarded_counts.get(def_name, 0),
            "discard_reasons": dict(discard_reasons.get(def_name, {})),
            "max_entries": max_entries,
            "max_dups_per_axes": max_dups_per_axes,
        }

    for def_name, count in discarded_counts.items():
        if count > 0:
            fi_api_label = def_name_to_fi_api.get(def_name, "?")
            reason_summary = ", ".join(
                f"{reason}={reason_count}"
                for reason, reason_count in discard_reasons[def_name].most_common(3)
            )
            log(
                f"  WARNING: {def_name} (fi_api={fi_api_label}): "
                f"{count} capture(s) discarded ({reason_summary})"
            )
            diagnostics.setdefault(def_name, {
                "candidates": 0,
                "selected": 0,
                "discarded": 0,
                "discard_reasons": {},
                "max_entries": max_entries,
                "max_dups_per_axes": max_dups_per_axes,
            })
            diagnostics[def_name]["discarded"] = count
            diagnostics[def_name]["discard_reasons"] = dict(discard_reasons[def_name])

    for fi_api, def_names in api_to_defs.items():
        for def_name in def_names:
            if def_name not in results:
                log(f"  WARNING: no captures matched definition {def_name} (fi_api={fi_api})")
                diagnostics.setdefault(def_name, {
                    "candidates": 0,
                    "selected": 0,
                    "discarded": discarded_counts.get(def_name, 0),
                    "discard_reasons": dict(discard_reasons.get(def_name, {})),
                    "max_entries": max_entries,
                    "max_dups_per_axes": max_dups_per_axes,
                })

    return {"counts": results, "diagnostics": diagnostics}
