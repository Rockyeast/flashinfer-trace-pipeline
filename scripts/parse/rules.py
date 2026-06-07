"""Manual trace_id classification fallback.

The main path now lives in scripts/adapters/: each adapter owns classification,
inventory building, and definition rendering for its op_type.  Keep this file
small and explicit for one-off reviewed exceptions only.
"""

from __future__ import annotations


_MANUAL_TRACE_RULES: dict[str, tuple[str, str]] = {
    # Format: "trace_id": ("op_type", "variant")
    # Add entries only when:
    #   - no adapter recognizes the trace_id (it appears in the unmatched list)
    #   - the trace_id has been manually verified as a real FlashInfer kernel call
    #   - the case is not worth a dedicated adapter (one isolated trace_id exception)
    #
    # Example:
    # "flashinfer.BatchDecodeWithPagedKVCacheWrapper.forward_return_lse": ("gqa", "paged_decode"),
}

_TRACE_CLASSIFICATION_GOLDEN_CASES: dict[str, tuple[str, str] | None] = {
    "sgl_kernel.silu_and_mul": None,
    "sgl_kernel.gelu_tanh_and_mul": None,
}


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    """Classify a trace_id with manually reviewed exceptional rules only."""
    return _MANUAL_TRACE_RULES.get(trace_id)


def self_test_trace_classification() -> None:
    """Run golden trace_id classification checks."""
    failures = []
    for trace_id, expected in _TRACE_CLASSIFICATION_GOLDEN_CASES.items():
        got = classify_trace_id(trace_id)
        if got != expected:
            failures.append((trace_id, expected, got))
    if failures:
        detail = "\n".join(
            f"{trace_id}: expected {expected}, got {got}"
            for trace_id, expected, got in failures
        )
        raise AssertionError(f"trace_id classification self-test failed:\n{detail}")


if __name__ == "__main__":
    self_test_trace_classification()
    print("trace_id classification self-test passed")
