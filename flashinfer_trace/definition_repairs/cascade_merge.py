"""Sanitize hints for cascade merge fitrace definitions."""

from __future__ import annotations

from typing import Any

from flashinfer_trace.definition_repairs.base import DefinitionRepairer


def _issue(definition: dict[str, Any]) -> str | None:
    return None


def _repair(definition: dict[str, Any], events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    return None, "no_repair_needed"


def _hints(definition: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if definition.get("op_type") != "cascade_merge":
        return None
    name = definition.get("name")
    inputs = definition.get("inputs")
    if not isinstance(name, str) or not name or not isinstance(inputs, dict):
        return None
    expected = ("v_a", "s_a", "v_b", "s_b")
    if not all(input_name in inputs for input_name in expected):
        return None
    return {
        "schema_version": 1,
        "definition_name": name,
        "op_type": "cascade_merge",
        "inputs": {
            input_name: [{"source": "arg", "arg_index": index}]
            for index, input_name in enumerate(expected)
        },
    }


CASCADE_MERGE_REPAIRER = DefinitionRepairer(
    name="cascade_merge",
    issue=_issue,
    repair=_repair,
    hints=_hints,
)
