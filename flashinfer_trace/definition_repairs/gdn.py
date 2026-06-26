"""Sanitize hints for GDN fitrace definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from flashinfer_trace.definition_repairs.base import DefinitionRepairer


def _issue(definition: dict[str, Any]) -> str | None:
    if definition.get("op_type") != "gdn":
        return None
    inputs = definition.get("inputs")
    if not isinstance(inputs, dict):
        return None
    name = definition.get("name")
    removed = [input_name for input_name in _runtime_default_inputs(name) if input_name in inputs]
    if removed:
        return f"gdn definition includes runtime-default inputs not present in workload: {removed}"
    unknown = [
        name
        for name in ("A_log", "dt_bias")
        if isinstance(inputs.get(name), dict) and inputs[name].get("dtype") in {None, "", "unknown"}
    ]
    if unknown:
        return f"gdn optional inputs have unknown dtype: {unknown}"
    return None


def _repair(definition: dict[str, Any], events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    repaired = deepcopy(definition)
    inputs = repaired.get("inputs")
    if not isinstance(inputs, dict):
        return None, "gdn_inputs_not_object"
    name = repaired.get("name")
    removed = []
    for input_name in _runtime_default_inputs(name):
        if input_name in inputs:
            inputs.pop(input_name)
            removed.append(input_name)
    fixed: list[str] = []
    for name in ("A_log", "dt_bias"):
        spec = inputs.get(name)
        if isinstance(spec, dict) and spec.get("dtype") in {None, "", "unknown"}:
            spec["dtype"] = "float32"
            fixed.append(name)
    reference_fixed = _rewrite_reference(repaired)
    if not fixed and not removed and not reference_fixed:
        return None, "no_repair_needed"
    audit = repaired.get("audit")
    if not isinstance(audit, dict):
        audit = {}
    audit["gdn_schema_fixes"] = [
        *(f"removed_input:{name}" for name in removed),
        *(f"dtype:{name}:float32" for name in fixed),
        *(["reference_runtime_defaults"] if reference_fixed else []),
    ]
    repaired["audit"] = audit
    return repaired, "gdn_runtime_inputs"


def _hints(definition: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if definition.get("op_type") != "gdn":
        return None
    name = definition.get("name")
    if not isinstance(name, str) or not name:
        return None
    if "prefill" in name:
        return _prefill_hints(name)
    if "decode" in name:
        return _decode_hints(name)
    return None


def _prefill_hints(name: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "definition_name": name,
        "op_type": "gdn",
        "inputs": {
            "q": [{"source": "kwarg", "name": "q"}],
            "k": [{"source": "kwarg", "name": "k"}],
            "v": [{"source": "kwarg", "name": "v"}],
            "state": [{"source": "kwarg", "name": "initial_state"}],
            "a": [{"source": "kwarg", "name": "g"}],
            "b": [{"source": "kwarg", "name": "beta"}],
            "cu_seqlens": [{"source": "kwarg", "name": "cu_seqlens"}],
        },
        "axes": {
            "total_seq_len": {"source": "tensor_last", "input": "cu_seqlens"},
            "num_seqs": {"source": "tensor_numel_minus_one", "input": "cu_seqlens"},
        },
    }


def _decode_hints(name: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "definition_name": name,
        "op_type": "gdn",
        "inputs": {
            "q": [{"source": "kwarg", "name": "q"}],
            "k": [{"source": "kwarg", "name": "k"}],
            "v": [{"source": "kwarg", "name": "v"}],
            "state": [
                {"source": "kwarg", "name": "state"},
                {"source": "kwarg", "name": "initial_state"},
            ],
            "A_log": [{"source": "kwarg", "name": "A_log"}],
            "a": [{"source": "kwarg", "name": "a"}],
            "dt_bias": [{"source": "kwarg", "name": "dt_bias"}],
            "b": [{"source": "kwarg", "name": "b"}],
        },
    }


def _runtime_default_inputs(name: Any) -> tuple[str, ...]:
    if not isinstance(name, str):
        return ()
    if "prefill" in name:
        return ("A_log", "dt_bias", "scale")
    if "decode" in name:
        return ("scale",)
    return ()


def _rewrite_reference(definition: dict[str, Any]) -> bool:
    name = definition.get("name")
    reference = definition.get("reference")
    if not isinstance(name, str) or not isinstance(reference, str):
        return False
    updated = reference
    if "prefill" in name:
        updated = updated.replace(
            "def _gdn_prefill_reference(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):",
            "def _gdn_prefill_reference(q, k, v, state, a, b, cu_seqlens):",
        )
        updated = updated.replace(
            "def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):\n"
            "    return _gdn_prefill_reference(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)",
            "def run(q, k, v, state, a, b, cu_seqlens):\n"
            "    return _gdn_prefill_reference(q, k, v, state, a, b, cu_seqlens)",
        )
        updated = updated.replace(
            "    if scale is None or scale == 0.0:\n"
            "        scale = 1.0 / math.sqrt(head_size)\n\n"
            "    x = a.float() + dt_bias.float()  # [total_seq_len, HV]\n"
            "    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [total_seq_len, HV]\n"
            "    beta = torch.sigmoid(b.float())  # [total_seq_len, HV]\n",
            "    scale = 1.0 / math.sqrt(head_size)\n\n"
            "    g = a.float()  # precomputed gate from runtime g\n"
            "    beta = b.float()  # precomputed update gate from runtime beta\n",
        )
    if "decode" in name:
        updated = updated.replace(
            "def _gdn_decode_reference(q, k, v, state, A_log, a, dt_bias, b, scale):",
            "def _gdn_decode_reference(q, k, v, state, A_log, a, dt_bias, b):",
        )
        updated = updated.replace(
            "def run(q, k, v, state, A_log, a, dt_bias, b, scale):\n"
            "    return _gdn_decode_reference(q, k, v, state, A_log, a, dt_bias, b, scale)",
            "def run(q, k, v, state, A_log, a, dt_bias, b):\n"
            "    return _gdn_decode_reference(q, k, v, state, A_log, a, dt_bias, b)",
        )
        updated = updated.replace(
            "    if scale is None or scale == 0.0:\n"
            "        scale = 1.0 / math.sqrt(K)\n",
            "    scale = 1.0 / math.sqrt(K)\n",
        )
    if updated == reference:
        return False
    definition["reference"] = updated
    return True


GDN_REPAIRER = DefinitionRepairer(
    name="gdn",
    issue=_issue,
    repair=_repair,
    hints=_hints,
)
