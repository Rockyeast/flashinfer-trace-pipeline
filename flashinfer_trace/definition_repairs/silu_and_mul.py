"""Sanitize hints for SiluAndMul definition drafts."""

from __future__ import annotations

from typing import Any

from flashinfer_trace.definition_repairs.base import DefinitionRepairer


def _issue(definition: dict[str, Any]) -> str | None:
    return None


def _repair(definition: dict[str, Any], events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    return None, "no_repair_needed"


def _hints(definition: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if definition.get("op_type") != "silu_and_mul":
        return None
    name = definition.get("name")
    inputs = definition.get("inputs")
    if not isinstance(name, str) or not name or not isinstance(inputs, dict):
        return None
    if "input" not in inputs:
        return None
    return {
        "schema_version": 1,
        "definition_name": name,
        "op_type": "silu_and_mul",
        "inputs": {
            "input": [{"source": "arg", "arg_index": 0}],
        },
    }


SILU_AND_MUL_REPAIRER = DefinitionRepairer(
    name="silu_and_mul",
    issue=_issue,
    repair=_repair,
    hints=_hints,
)
