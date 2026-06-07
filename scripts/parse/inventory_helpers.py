"""Inventory assembly helpers used by parse and downstream pipeline stages."""

from __future__ import annotations

import json

from parse.diagnostics import runtime_method_from_trace_id, runtime_status_from_evidence


FLASHINFER_METHOD_SUFFIXES = {
    "__call__",
    "begin_forward",
    "end_forward",
    "forward",
    "forward_decode",
    "forward_extend",
    "forward_return_lse",
    "plan",
    "run",
}

KERNEL_STATE_KEY = "inventory_status"
KERNEL_STATE_NEEDS_CONFIG = "needs_config"
KERNEL_STATE_NEW = "new"
KERNEL_STATE_EXISTING = "existing"
KERNEL_STATE_STATIC_CANDIDATE = "static_candidate"

_KERNEL_STATE_RANK = {
    KERNEL_STATE_NEEDS_CONFIG: 0,
    KERNEL_STATE_NEW: 1,
    KERNEL_STATE_EXISTING: 2,
}


def get_kernel_state(kernel: dict) -> str:
    """Return the pipeline status for one inventory kernel entry."""
    status = kernel.get(KERNEL_STATE_KEY, "")
    return status if isinstance(status, str) else ""


def set_kernel_state(kernel: dict, state: str) -> None:
    """Set the pipeline status for one inventory kernel entry."""
    kernel[KERNEL_STATE_KEY] = state


def is_existing_kernel(kernel: dict) -> bool:
    return get_kernel_state(kernel) == KERNEL_STATE_EXISTING


def is_new_kernel(kernel: dict) -> bool:
    return get_kernel_state(kernel) == KERNEL_STATE_NEW


def needs_config_kernel(kernel: dict) -> bool:
    return get_kernel_state(kernel) == KERNEL_STATE_NEEDS_CONFIG


def kernel_state_rank(kernel: dict) -> int:
    return _KERNEL_STATE_RANK.get(get_kernel_state(kernel), -1)


def observed_fi_api_from_trace_id(trace_id: str) -> str | None:
    """Extract official fi_api evidence from an observed flashinfer.* trace_id."""
    if not trace_id.startswith("flashinfer."):
        return None
    head, sep, tail = trace_id.rpartition(".")
    if sep and tail in FLASHINFER_METHOD_SUFFIXES:
        return head
    return trace_id


def split_api_fields(api: str | None) -> tuple[str | None, str | None]:
    """Split an API string into (fi_api, target_api)."""
    if not api:
        return None, None
    if api.startswith("flashinfer."):
        return api, None
    return None, api


def merge_matched_kernel(
    matched_kernels: dict[str, dict],
    *,
    key: str,
    op_type: str,
    variant: str,
    fi_api: str | None,
    source: str,
    trace_id: str,
    count: int,
    signatures: list[dict],
) -> None:
    """Merge one trace_id observation into the matched kernel bucket."""
    resolved_fi_api, target_api = split_api_fields(fi_api)
    if key not in matched_kernels:
        matched_kernels[key] = {
            "op_type": op_type,
            "variant": variant,
            "fi_api": resolved_fi_api,
            "target_api": target_api,
            "trace_ids": [],
            "trace_counts": {},
            "total_count": 0,
            "signatures": [],
            "runtime_evidence": {},
            "source": source,
            "classification_sources": [],
        }

    method = runtime_method_from_trace_id(trace_id)
    matched_kernels[key]["trace_ids"].append(trace_id)
    matched_kernels[key]["trace_counts"][trace_id] = (
        matched_kernels[key]["trace_counts"].get(trace_id, 0) + count
    )
    matched_kernels[key]["total_count"] += count
    matched_kernels[key]["signatures"].extend(signatures)
    matched_kernels[key]["runtime_evidence"][method] = (
        matched_kernels[key]["runtime_evidence"].get(method, 0) + count
    )
    if source not in matched_kernels[key]["classification_sources"]:
        matched_kernels[key]["classification_sources"].append(source)
    if matched_kernels[key].get("fi_api") is None and resolved_fi_api is not None:
        matched_kernels[key]["fi_api"] = resolved_fi_api
    if matched_kernels[key].get("target_api") is None and target_api is not None:
        matched_kernels[key]["target_api"] = target_api


def observation_fields(info: dict, obs_kw: list[str], obs_pn: list[str]) -> dict:
    """Common observation metadata copied into each generated kernel entry."""
    evidence = dict(info.get("runtime_evidence", {}))
    return {
        "call_count": info["total_count"],
        "observed_kwargs": obs_kw,
        "observed_param_names": obs_pn,
        "trace_ids": list(info.get("trace_ids", [])),
        "runtime_evidence": evidence,
        "runtime_status": runtime_status_from_evidence(evidence),
        **({"target_api": info["target_api"]} if info.get("target_api") else {}),
    }


def wrapper_observation_fields(info: dict, obs_kw: list[str], obs_pn: list[str]) -> dict:
    """Observation metadata for SGLang wrapper calls that do not prove a FI kernel ran."""
    fields = observation_fields(info, obs_kw, obs_pn)
    fields["runtime_evidence"] = {"wrapper_forward": info["total_count"]}
    fields["runtime_status"] = "wrapper_observed"
    return fields


def merge_list_fields(left: list | None, right: list | None) -> list:
    """Merge two small metadata lists while preserving order."""
    merged = []
    seen = set()
    for value in (left or []) + (right or []):
        key = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


def dedup_kernels_by_definition_name(kernels: list[dict]) -> list[dict]:
    """Collapse kernels that resolve to the same definition_name."""
    deduped: dict[str, dict] = {}
    for kernel in kernels:
        name = kernel.get("definition_name")
        if not name:
            continue
        if name not in deduped:
            deduped[name] = dict(kernel)
            continue

        existing = deduped[name]
        existing["call_count"] = max(
            int(existing.get("call_count") or 0),
            int(kernel.get("call_count") or 0),
        )
        existing["observed_kwargs"] = merge_list_fields(
            existing.get("observed_kwargs"),
            kernel.get("observed_kwargs"),
        )
        existing["observed_param_names"] = merge_list_fields(
            existing.get("observed_param_names"),
            kernel.get("observed_param_names"),
        )
        trace_ids = merge_list_fields(existing.get("trace_ids"), kernel.get("trace_ids"))
        if trace_ids:
            existing["trace_ids"] = trace_ids
        runtime_evidence = dict(existing.get("runtime_evidence") or {})
        for method, count in (kernel.get("runtime_evidence") or {}).items():
            runtime_evidence[method] = int(runtime_evidence.get(method, 0)) + int(count or 0)
        if runtime_evidence:
            existing["runtime_evidence"] = runtime_evidence
            existing["runtime_status"] = runtime_status_from_evidence(runtime_evidence)
        if existing.get("fi_api") is None and kernel.get("fi_api") is not None:
            existing["fi_api"] = kernel["fi_api"]
        if existing.get("target_api") is None and kernel.get("target_api") is not None:
            existing["target_api"] = kernel["target_api"]
        if kernel_state_rank(kernel) > kernel_state_rank(existing):
            set_kernel_state(existing, get_kernel_state(kernel))
        if existing.get("note") != kernel.get("note"):
            notes = [n for n in (existing.get("note"), kernel.get("note")) if n]
            if notes:
                existing["note"] = " | ".join(dict.fromkeys(notes))
    return list(deduped.values())
