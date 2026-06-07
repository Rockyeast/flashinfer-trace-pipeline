"""Kernel adapters for trace classification, inventory, and definitions.

Adapters are the narrow path for new official FlashInfer-backed kernel families:
one module owns trace_id classification, signature-to-inventory extraction, and
definition rendering through official TraceTemplate when possible.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType

_SKIP_MODULES = {"__init__", "extractors", "official_templates"}
_REQUIRED_ATTRS = {"OP_TYPE", "classify_trace_id", "build_kernels", "generate_definition"}


def _is_adapter_module(module: ModuleType) -> bool:
    """Return True if the module exposes all required adapter attributes."""
    return all(hasattr(module, attr) for attr in _REQUIRED_ATTRS)


def _load_adapters() -> tuple[ModuleType, ...]:
    """Discover and load all adapter modules in this directory.

    Scans for *.py files (sorted alphabetically), skips helper modules and
    underscore-prefixed files, then keeps only modules that satisfy
    _is_adapter_module. No registration needed — adding a new adapter file is
    sufficient.
    """
    adapters: list[ModuleType] = []
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.stem in _SKIP_MODULES or path.stem.startswith("_"):
            continue
        module = import_module(f"{__name__}.{path.stem}")
        if _is_adapter_module(module):
            adapters.append(module)
    return tuple(adapters)


ADAPTERS = _load_adapters()


def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
    """Return (op_type, variant) for the first adapter that recognises trace_id.

    Adapters are tried in alphabetical file order. Returns None if no adapter
    matches. Each adapter should only match its own op_type so conflicts are
    unlikely, but first-match wins if they do occur.
    """
    for adapter in ADAPTERS:
        result = adapter.classify_trace_id(trace_id)
        if result is not None:
            return result
    return None


def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
    """Build kernel inventory entries from the aggregated trace buckets.

    Each adapter filters matched_kernels by its own op_type prefix (keys of the
    form "op_type:variant") and converts the aggregated data into kernel entry
    dicts. Results from all adapters are concatenated into a single list.
    """
    kernels: list[dict] = []
    for adapter in ADAPTERS:
        kernels.extend(adapter.build_kernels(matched_kernels))
    return kernels


def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict | None:
    for adapter in ADAPTERS:
        op_types = getattr(adapter, "OP_TYPES", {adapter.OP_TYPE})
        if kernel.get("op_type") in op_types:
            return adapter.generate_definition(kernel, model_tag, tp)
    return None


def ignored_observed_kwargs(kernel: dict, definition: dict) -> set[str]:
    """Return runtime kwargs intentionally excluded by the official template."""
    op_type = str(kernel.get("op_type") or definition.get("op_type") or "")
    adapter = adapter_for_op_type(op_type)
    hook = getattr(adapter, "ignored_observed_kwargs", None) if adapter else None
    if hook is None:
        return set()
    return set(hook(kernel, definition))


def ignored_definition_inputs_for_observed_params(
    kernel: dict,
    definition: dict,
) -> set[str]:
    """Return official inputs hidden behind runtime wrapper parameters."""
    op_type = str(kernel.get("op_type") or definition.get("op_type") or "")
    adapter = adapter_for_op_type(op_type)
    hook = (
        getattr(adapter, "ignored_definition_inputs_for_observed_params", None)
        if adapter
        else None
    )
    if hook is None:
        return set()
    return set(hook(kernel, definition))


def generate_baseline_solution(definition: dict) -> dict | None:
    """Return an optional op-specific FlashInfer baseline solution."""
    for adapter in ADAPTERS:
        op_types = getattr(adapter, "OP_TYPES", {adapter.OP_TYPE})
        if definition.get("op_type") in op_types and hasattr(
            adapter, "generate_baseline_solution"
        ):
            return adapter.generate_baseline_solution(definition)
    return None


def adapter_for_op_type(op_type: str) -> ModuleType | None:
    """Return the adapter module that owns op_type."""
    for adapter in ADAPTERS:
        op_types = getattr(adapter, "OP_TYPES", {adapter.OP_TYPE})
        if op_type in op_types:
            return adapter
    return None


def pr_reference_source(op_type: str) -> str:
    """Return the source description used in PR copy."""
    adapter = adapter_for_op_type(op_type)
    hook = getattr(adapter, "pr_reference_source", None) if adapter else None
    if hook is not None:
        return hook()
    return "FlashInfer unit tests"


def pr_kernel_description(def_name: str, op_type: str, model_label: str) -> str:
    """Return the human-readable kernel description used in PR copy."""
    adapter = adapter_for_op_type(op_type)
    hook = getattr(adapter, "pr_kernel_description", None) if adapter else None
    if hook is not None:
        return hook(def_name, model_label)
    return f"{model_label} {def_name}"


def pr_baseline_description(def_name: str, op_type: str) -> str:
    """Return the baseline solution description used in PR copy."""
    adapter = adapter_for_op_type(op_type)
    hook = getattr(adapter, "pr_baseline_description", None) if adapter else None
    if hook is not None:
        return hook(def_name)
    return f"flashinfer {op_type} wrapper"


def eval_validation_policy(definition: dict) -> str:
    """Return how eval trace should validate this definition's outputs."""
    op_type = str(definition.get("op_type") or "")
    adapter = adapter_for_op_type(op_type)
    hook = getattr(adapter, "eval_validation_policy", None) if adapter else None
    if hook is not None:
        return hook(definition)
    return "exact_close"


def skip_submit_definition(def_name: str, op_type: str) -> bool:
    """Return whether submit scripts should skip this definition family."""
    for adapter in ADAPTERS:
        hook = getattr(adapter, "skip_submit_definition", None)
        if hook is not None and hook(def_name, op_type):
            return True
    return False
