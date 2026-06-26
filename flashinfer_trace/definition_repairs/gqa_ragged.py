"""Sanitize hints for gqa_ragged fitrace definitions."""

from __future__ import annotations

from typing import Any

from flashinfer_trace.definition_repairs.base import DefinitionRepairer


def _issue(definition: dict[str, Any]) -> str | None:
    return None


def _repair(definition: dict[str, Any], events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    return None, "no_repair_needed"


def _hints(definition: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if definition.get("op_type") != "gqa_ragged":
        return None
    name = definition.get("name")
    if not isinstance(name, str) or not name:
        return None
    return {
        "schema_version": 1,
        "definition_name": name,
        "op_type": "gqa_ragged",
        "inputs": {
            "q": [{"source": "arg", "arg_index": 1}],
            "k": [{"source": "arg", "arg_index": 2}],
            "v": [{"source": "arg", "arg_index": 3}],
            "qo_indptr": [{"source": "attr", "pattern": "qo_indptr"}],
            "kv_indptr": [{"source": "attr", "pattern": "kv_indptr"}],
            "sm_scale": [{"source": "kwarg", "name": "sm_scale"}],
        },
        "axes": {
            "total_q": {"source": "tensor_last", "input": "qo_indptr"},
            "total_kv": {"source": "tensor_last", "input": "kv_indptr"},
        },
    }


GQA_RAGGED_REPAIRER = DefinitionRepairer(
    name="gqa_ragged",
    issue=_issue,
    repair=_repair,
    hints=_hints,
)
