"""Resolve NEEDS_CONFIG inventory entries with shared dimension resolver."""

from __future__ import annotations

from types import ModuleType

from adapters import ADAPTERS
from adapters._param_utils import extract_hf_dims, is_unknown
from parse.inventory_helpers import (
    KERNEL_STATE_EXISTING,
    KERNEL_STATE_NEW,
    set_kernel_state,
)


def _config_adapter_for_kernel(kernel: dict) -> ModuleType | None:
    """Return the adapter that can resolve HF config params for this kernel."""
    op_type = kernel.get("op_type")
    for adapter in ADAPTERS:
        op_types = getattr(adapter, "OP_TYPES", {adapter.OP_TYPE})
        if op_type in op_types and hasattr(adapter, "resolve_config_params"):
            return adapter
    return None


def _needs_config_resolution(kernel: dict, adapter: ModuleType) -> bool:
    """Return whether adapter-specific config enrichment should run."""
    hook = getattr(adapter, "needs_config_resolution", None)
    if hook is not None:
        return bool(hook(kernel))
    return "NEEDS_CONFIG" in kernel.get("definition_name", "")


def _append_note(kernel: dict, note: str) -> None:
    """Append a short human-facing note without duplicating text."""
    existing = kernel.get("note")
    if not existing:
        kernel["note"] = note
        return
    notes = [part.strip() for part in str(existing).split(" | ") if part.strip()]
    if note not in notes:
        notes.append(note)
    kernel["note"] = " | ".join(notes)


def _config_dim_keys(config: dict, tp_size: int | None) -> list[str]:
    """Return HF-derived dimension keys that currently have concrete values."""
    dims = extract_hf_dims(config, tp_size)
    return sorted(key for key, value in dims.items() if not is_unknown(value))


def _sorted_strings(values) -> list[str]:
    """Return deterministic string diagnostics for small metadata lists."""
    return sorted(str(value) for value in values)


def _set_config_resolution_failure(
    kernel: dict,
    *,
    adapter: ModuleType | None,
    config: dict,
    tp_size: int | None,
    reason: str,
    error: str | None = None,
) -> None:
    """Record why HF config enrichment could not resolve this kernel."""
    diagnostics = kernel.setdefault("diagnostics", {})
    payload = {
        "status": "failed",
        "reason": reason,
        "definition_name": kernel.get("definition_name"),
        "raw_param_keys": _sorted_strings((kernel.get("params") or {}).keys()),
        "observed_kwargs": _sorted_strings(kernel.get("observed_kwargs") or []),
        "observed_param_names": _sorted_strings(
            kernel.get("observed_param_names") or []
        ),
        "config_dim_keys": _config_dim_keys(config, tp_size),
    }
    if adapter is not None:
        payload["adapter"] = adapter.__name__.rsplit(".", 1)[-1]
        diagnose = getattr(adapter, "diagnose_config_resolution", None)
        if diagnose is not None:
            try:
                extra = diagnose(kernel, config, tp_size)
            except Exception as exc:
                payload["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
            else:
                if isinstance(extra, dict):
                    payload.update(extra)
    if error:
        payload["error"] = error
    payload["message"] = (
        f"Config enrichment failed for {kernel.get('op_type')}."
        f"{kernel.get('variant')}: {reason}"
    )
    diagnostics["config_resolution"] = payload
    _append_note(kernel, "Config enrichment failed; see diagnostics.config_resolution.")


def _clear_config_resolution_diagnostics(kernel: dict) -> None:
    """Remove stale config-resolution diagnostics after successful enrichment."""
    diagnostics = kernel.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return
    diagnostics.pop("config_resolution", None)
    if not diagnostics:
        kernel.pop("diagnostics", None)


def enrich_with_hf_config(
    kernels: list[dict],
    config: dict,
    existing_defs: set[str] | None = None,
    tp_size: int | None = None,
) -> None:
    """Use HF config/probe params to resolve NEEDS_CONFIG definition names."""

    def _set_status(kernel: dict) -> None:
        if existing_defs and kernel["definition_name"] in existing_defs:
            set_kernel_state(kernel, KERNEL_STATE_EXISTING)
        else:
            set_kernel_state(kernel, KERNEL_STATE_NEW)

    for kernel in kernels:
        adapter = _config_adapter_for_kernel(kernel)
        if adapter is None:
            if "NEEDS_CONFIG" in kernel.get("definition_name", ""):
                _set_config_resolution_failure(
                    kernel,
                    adapter=None,
                    config=config,
                    tp_size=tp_size,
                    reason="no_adapter_with_resolve_config_params",
                )
            continue
        if not _needs_config_resolution(kernel, adapter):
            continue
        try:
            params = adapter.resolve_config_params(kernel, config, tp_size)
        except Exception as exc:
            _set_config_resolution_failure(
                kernel,
                adapter=adapter,
                config=config,
                tp_size=tp_size,
                reason="resolver_exception",
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        if not params:
            _set_config_resolution_failure(
                kernel,
                adapter=adapter,
                config=config,
                tp_size=tp_size,
                reason="resolver_returned_none",
            )
            continue
        try:
            kernel["definition_name"] = adapter.definition_name(kernel["variant"], params)
        except (KeyError, ValueError) as exc:
            _set_config_resolution_failure(
                kernel,
                adapter=adapter,
                config=config,
                tp_size=tp_size,
                reason="definition_name_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        kernel["params"] = params
        _set_status(kernel)
        kernel.pop("note", None)
        _clear_config_resolution_diagnostics(kernel)
