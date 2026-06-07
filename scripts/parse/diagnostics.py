"""Runtime evidence and diagnostics helpers for probe parsing."""

from __future__ import annotations

DEFERRED_RUNTIME_STATUSES = {"prepared_only"}


def runtime_method_from_trace_id(trace_id: str) -> str:
    """Classify a trace_id into a lightweight runtime evidence bucket."""
    method = trace_id.rsplit(".", 1)[-1]
    if method in {"run", "forward", "apply", "plan", "begin_forward"}:
        return method
    return "function_call"


def runtime_status_from_evidence(evidence: dict[str, int]) -> str:
    """Return whether a kernel was actually executed or only prepared."""
    if any(evidence.get(k, 0) > 0 for k in ("run", "forward", "apply", "function_call")):
        return "observed_run"
    if any(evidence.get(k, 0) > 0 for k in ("plan", "begin_forward")):
        return "prepared_only"
    if evidence.get("wrapper_forward", 0) > 0:
        return "wrapper_observed"
    return "unknown"


def ignored_noise_reason(trace_id: str) -> str | None:
    """Return a reason when an unmatched trace_id is a known helper/noise call."""
    method = trace_id.rsplit(".", 1)[-1]
    if method.startswith("is_"):
        return "predicate_helper"
    if method.startswith("get_"):
        return "getter_helper"
    if method.startswith("use_"):
        return "backend_switch_helper"
    if method.startswith("device_support"):
        return "device_capability_helper"
    if method.startswith("canonicalize"):
        return "argument_normalization_helper"
    return None


def detect_model_backend(data: dict) -> str:
    """Infer whether probe ran through native SGLang model code or HF Transformers fallback."""
    for sig in data.get("top_signatures", []):
        for frame in sig.get("stack") or []:
            if "/transformers/models/" in frame:
                return "transformers_fallback"
            if "/sglang/srt/models/" in frame:
                return "sglang_native"
    return "unknown"


def deferred_reason(kernel: dict, model_backend: str) -> str:
    """Human-readable reason for keeping a discovered kernel out of the main flow."""
    runtime_status = kernel.get("runtime_status")
    if runtime_status == "prepared_only":
        return (
            "Observed plan/begin_forward preparation without a run/forward/apply/function_call "
            "execution; not collected by default."
        )
    if runtime_status == "wrapper_observed":
        if not kernel.get("fi_api"):
            return (
                "Observed a wrapper-level forward path without an observed flashinfer.* "
                f"API under {model_backend}; not collected by default."
            )
        return (
            "Observed a wrapper-level forward path, but not a concrete executable "
            f"FlashInfer run/plan capture under {model_backend}; not collected by default."
        )
    return "Runtime evidence is not strong enough for default definition/workload generation."


def split_deferred_kernels(kernels: list[dict], model_backend: str) -> tuple[list[dict], list[dict]]:
    """Separate formal kernels from weaker runtime observations."""
    formal: list[dict] = []
    deferred: list[dict] = []
    for kernel in kernels:
        runtime_status = kernel.get("runtime_status")
        should_defer = runtime_status in DEFERRED_RUNTIME_STATUSES
        # Wrapper-only observations are diagnostic unless parse has a concrete
        # observed flashinfer.* API. This keeps SGLang/target_api-only traces out
        # of the default definition/collect path.
        if runtime_status == "wrapper_observed" and not kernel.get("fi_api"):
            should_defer = True

        if should_defer:
            item = dict(kernel)
            item["deferred_reason"] = deferred_reason(kernel, model_backend)
            deferred.append(item)
        else:
            item = dict(kernel)
            item.pop("runtime_status", None)
            formal.append(item)
    return formal, deferred
