"""Sanitize hints for sampling fitrace definitions."""

from __future__ import annotations

from typing import Any

from flashinfer_trace.definition_repairs.base import DefinitionRepairer


def _issue(definition: dict[str, Any]) -> str | None:
    return None


def _repair(definition: dict[str, Any], events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    return None, "no_repair_needed"


def _hints(definition: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if definition.get("op_type") != "sampling":
        return None
    name = definition.get("name")
    if not isinstance(name, str) or not name:
        return None
    return {
        "schema_version": 1,
        "definition_name": name,
        "op_type": "sampling",
        "inputs": {
            "probs": [{"source": "arg", "arg_index": 0}],
            "top_k": [{"source": "arg", "arg_index": 1}],
            "top_p": [{"source": "arg", "arg_index": 2}],
        },
        "real_inputs": ["probs", "top_k", "top_p"],
    }


SAMPLING_REPAIRER = DefinitionRepairer(
    name="sampling",
    issue=_issue,
    repair=_repair,
    hints=_hints,
)
