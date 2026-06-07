"""Reference test generator registry.

Strict tests use adapter-backed baseline solutions. If no adapter solution is
available, generation fails so missing correctness coverage is explicit.
"""

from __future__ import annotations

from . import adapter_solution


def generate_test_file(definition: dict) -> str:
    """Generate a pytest file for a definition."""
    op_type = definition.get("op_type", "")
    try:
        return adapter_solution.generate(definition)
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"no adapter-backed strict test generator for op_type={op_type!r}"
        ) from exc
