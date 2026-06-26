"""Repair rules for RMSNorm definition drafts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from flashinfer_trace.definition_repairs.base import DefinitionRepairer


def _issue(definition: dict[str, Any]) -> str | None:
    if definition.get("op_type") != "rmsnorm":
        return None
    inputs = definition.get("inputs")
    eps = inputs.get("eps") if isinstance(inputs, dict) else None
    if isinstance(eps, dict) and eps.get("shape") == []:
        return "rmsnorm eps is scalar but definition declares tensor shape []"
    return None


def _repair(definition: dict[str, Any], events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    repaired = deepcopy(definition)
    inputs = repaired.get("inputs")
    eps = inputs.get("eps") if isinstance(inputs, dict) else None
    if not isinstance(eps, dict):
        return None, "missing_eps_input"
    eps["shape"] = None
    audit = repaired.get("audit")
    if not isinstance(audit, dict):
        audit = {}
    audit["rmsnorm_schema_fixes"] = ["eps:scalar"]
    repaired["audit"] = audit
    return repaired, "rmsnorm_scalar_eps"


def _hints(definition: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if definition.get("op_type") != "rmsnorm":
        return None
    name = definition.get("name")
    inputs = definition.get("inputs")
    if not isinstance(name, str) or not name or not isinstance(inputs, dict):
        return None
    input_name = "hidden_states" if "hidden_states" in inputs else "input" if "input" in inputs else None
    if input_name is None or "weight" not in inputs:
        return None

    input_rules: dict[str, list[dict[str, Any]]] = {
        input_name: [{"source": "arg", "arg_index": 0}],
    }
    if "residual" in inputs:
        input_rules["residual"] = [{"source": "arg", "arg_index": 1}]
        input_rules["weight"] = [{"source": "arg", "arg_index": 2}]
        if "eps" in inputs:
            input_rules["eps"] = [{"source": "arg", "arg_index": 3}]
    else:
        input_rules["weight"] = [{"source": "arg", "arg_index": 1}]
        if "eps" in inputs:
            input_rules["eps"] = [{"source": "arg", "arg_index": 2}]

    return {
        "schema_version": 1,
        "definition_name": name,
        "op_type": "rmsnorm",
        "inputs": input_rules,
    }


RMSNORM_REPAIRER = DefinitionRepairer(
    name="rmsnorm",
    issue=_issue,
    repair=_repair,
    hints=_hints,
)
