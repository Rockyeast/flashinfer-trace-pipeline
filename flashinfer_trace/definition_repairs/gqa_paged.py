"""Repair rules for gqa_paged fitrace definitions."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from flashinfer_trace.definition_repairs.base import DefinitionRepairer


def _issue(definition: dict[str, Any]) -> str | None:
    if definition.get("op_type") != "gqa_paged":
        return None
    q_heads = _const_axis(definition, "num_qo_heads")
    kv_heads = _const_axis(definition, "num_kv_heads")
    if q_heads is None or kv_heads is None:
        return None
    if kv_heads > q_heads:
        return f"gqa_paged num_kv_heads larger than num_qo_heads: {kv_heads}>{q_heads}"
    if q_heads % kv_heads != 0:
        return f"gqa_paged num_qo_heads not divisible by num_kv_heads: {q_heads}%{kv_heads}"
    return None


def _repair(definition: dict[str, Any], events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    event = _matching_event(definition, events)
    if event is None:
        return None, "no_matching_event"

    q_shape = _first_q_shape(event)
    k_shape = _first_kv_cache_shape(event)
    if not q_shape or not k_shape or len(q_shape) < 3:
        return None, "missing_q_or_cache_shape"
    if len(k_shape) != 3:
        return None, f"unsupported_cache_rank:{len(k_shape)}"

    q_heads = int(q_shape[-2])
    head_dim = int(q_shape[-1])
    kv_heads = int(k_shape[1])
    cache_head_dim = int(k_shape[2])
    if head_dim != cache_head_dim:
        return None, f"head_dim_mismatch:q{head_dim}_cache{cache_head_dim}"
    page_size = _event_page_size(event) or 1
    stage = _definition_stage(definition) or _event_stage(event)
    if stage not in {"decode", "prefill"}:
        return None, "unknown_stage"

    repaired = deepcopy(definition)
    repaired["name"] = (
        f"gqa_paged_decode_h{q_heads}_kv{kv_heads}_d{head_dim}_ps{page_size}"
        if stage == "decode"
        else f"gqa_paged_prefill_causal_h{q_heads}_kv{kv_heads}_d{head_dim}_ps{page_size}"
    )
    repaired["op_type"] = "gqa_paged"
    axes = repaired.setdefault("axes", {})
    if isinstance(axes, dict):
        _set_const_axis(axes, "num_qo_heads", q_heads)
        _set_const_axis(axes, "num_kv_heads", kv_heads)
        _set_const_axis(axes, "head_dim", head_dim)
        _set_const_axis(axes, "page_size", page_size)
    tags = repaired.get("tags")
    tag_set = {tag for tag in tags if isinstance(tag, str)} if isinstance(tags, list) else set()
    tag_set.discard("status:verified")
    tag_set.add("status:repaired")
    tag_set.add("source:fitrace_repaired")
    tag_set.add(f"stage:{stage}")
    repaired["tags"] = sorted(tag_set)
    repaired["audit"] = {
        "source": "fitrace_repair",
        "reason": "repaired gqa_paged axes from captured 3D KV cache",
        "event_name": event.get("name"),
        "q_shape": q_shape,
        "k_cache_shape": k_shape,
    }
    return repaired, "gqa_paged_3d_cache_axes"


GQA_PAGED_REPAIRER = DefinitionRepairer(
    name="gqa_paged",
    issue=_issue,
    repair=_repair,
    hints=lambda definition, events: _hints(definition),
)


def _hints(definition: dict[str, Any]) -> dict[str, Any] | None:
    if definition.get("op_type") != "gqa_paged":
        return None
    name = definition.get("name")
    if not isinstance(name, str) or not name:
        return None
    return {
        "schema_version": 1,
        "definition_name": name,
        "op_type": "gqa_paged",
        "inputs": {
            "q": [{"source": "arg", "arg_index": 1}],
            "k_cache": [{"source": "arg_tuple", "arg_index": 2, "tuple_index": 0}],
            "v_cache": [{"source": "arg_tuple", "arg_index": 2, "tuple_index": 1}],
            "kv_indptr": [{"source": "attr", "pattern": "paged_kv_indptr"}],
            "kv_indices": [{"source": "attr", "pattern": "paged_kv_indices"}],
            "qo_indptr": [{"source": "attr", "pattern": "qo_indptr"}],
            "kv_last_page_len": [
                {"source": "attr", "pattern": "paged_kv_last_page_len"},
                {"source": "attr", "pattern": "last_page_len"},
            ],
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


def _matching_event(definition: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    fi_api = _definition_fi_api(definition)
    stage = _definition_stage(definition)
    for event in events:
        if fi_api and event.get("target") != fi_api:
            continue
        if stage and _event_stage(event) != stage:
            continue
        if _first_q_shape(event) and _first_kv_cache_shape(event):
            return event
    return None


def _definition_fi_api(definition: dict[str, Any]) -> str | None:
    tags = definition.get("tags")
    if not isinstance(tags, list):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("fi_api:"):
            return tag.removeprefix("fi_api:")
    return None


def _definition_stage(definition: dict[str, Any]) -> str | None:
    tags = definition.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if tag == "stage:decode":
                return "decode"
            if tag == "stage:prefill":
                return "prefill"
    name = str(definition.get("name") or "").lower()
    if "decode" in name:
        return "decode"
    if "prefill" in name:
        return "prefill"
    return None


def _event_stage(event: dict[str, Any]) -> str | None:
    variant = event.get("variant")
    if variant in {"decode", "prefill"}:
        return str(variant)
    text = f"{event.get('name', '')} {event.get('target', '')}".lower()
    if "decode" in text:
        return "decode"
    if "prefill" in text:
        return "prefill"
    return None


def _event_page_size(event: dict[str, Any]) -> int | None:
    value = event.get("page_size")
    if isinstance(value, int) and value > 0:
        return value
    mode = event.get("active_probe_mode")
    if isinstance(mode, str):
        match = re.search(r"ps(\d+)", mode)
        if match:
            return int(match.group(1))
    return None


def _first_q_shape(event: dict[str, Any]) -> list[int] | None:
    for item in event.get("args", []):
        shape = _shape(item)
        if shape and len(shape) == 3:
            return shape
    return None


def _first_kv_cache_shape(event: dict[str, Any]) -> list[int] | None:
    for item in event.get("args", []):
        for shape in _tuple_shapes(item):
            if len(shape) in {3, 4}:
                return shape
    return None


def _shape(item: Any) -> list[int] | None:
    if not isinstance(item, dict):
        return None
    raw = item.get("shape")
    if not isinstance(raw, list):
        summary = item.get("summary")
        raw = summary.get("shape") if isinstance(summary, dict) else None
    if not isinstance(raw, list):
        return None
    try:
        return [int(dim) for dim in raw]
    except (TypeError, ValueError):
        return None


def _tuple_shapes(item: Any) -> list[list[int]]:
    if not isinstance(item, dict):
        return []
    elements = item.get("elements")
    if not isinstance(elements, list):
        summary = item.get("summary")
        elements = summary.get("elements") if isinstance(summary, dict) else None
    if not isinstance(elements, list):
        return []
    shapes: list[list[int]] = []
    for element in elements:
        shape = _shape(element)
        if shape:
            shapes.append(shape)
    return shapes


def _const_axis(definition: dict[str, Any], name: str) -> int | None:
    axes = definition.get("axes")
    if not isinstance(axes, dict):
        return None
    axis = axes.get(name)
    if not isinstance(axis, dict) or axis.get("type") != "const":
        return None
    value = axis.get("value")
    return int(value) if isinstance(value, int) else None


def _set_const_axis(axes: dict[str, Any], name: str, value: int) -> None:
    axis = axes.get(name)
    if not isinstance(axis, dict):
        axis = {"type": "const"}
    axis["type"] = "const"
    axis["value"] = int(value)
    axes[name] = axis
