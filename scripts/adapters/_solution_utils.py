"""Shared helpers for adapter-generated baseline solution JSON files."""

from __future__ import annotations

import hashlib
import textwrap
from typing import Iterable

_DEFAULT_TARGET_HARDWARE = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A100",
    "NVIDIA H20",
    "NVIDIA H100",
    "NVIDIA H200",
    "NVIDIA B200",
]


def input_names(definition: dict) -> list[str]:
    """Return definition input names in call order."""
    inputs = definition.get("inputs")
    if not isinstance(inputs, dict):
        return []
    return list(inputs)


def has_inputs(definition: dict, required: Iterable[str]) -> bool:
    """Check whether a definition contains every required input."""
    return set(required).issubset(input_names(definition))


def fi_api_from_tags(definition: dict) -> str | None:
    """Extract the first ``fi_api:...`` tag from a definition."""
    tags = definition.get("tags", [])
    if not isinstance(tags, list):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("fi_api:"):
            return tag[len("fi_api:") :]
    return None


def solution_payload(
    definition: dict,
    source: str,
    *,
    name_prefix: str = "flashinfer_wrapper",
    author: str = "baseline",
    dependencies: Iterable[str] = ("flashinfer",),
    description: str | None = None,
    target_hardware: Iterable[str] | None = None,
    validation_policy: str = "exact_close",
    extra_sources: list[dict] | None = None,
) -> dict:
    """Build a baseline solution payload with a content-stable name."""
    def_name = definition["name"]
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]
    sources = [{"path": "main.py", "content": source}]
    if extra_sources:
        sources.extend(extra_sources)
    return {
        "name": f"{name_prefix}_{digest}",
        "definition": def_name,
        "author": author,
        "spec": {
            "language": "python",
            "target_hardware": list(target_hardware or _DEFAULT_TARGET_HARDWARE),
            "entry_point": "main.py::run",
            "dependencies": list(dependencies),
            "validation_policy": validation_policy,
            "destination_passing_style": False,
            "binding": None,
        },
        "sources": sources,
        "description": description or "Baseline solution using FlashInfer.",
    }


def direct_function_source(fi_api: str, args: list[str]) -> str:
    """Build source that calls a FlashInfer function positionally."""
    module_name, _, attr_name = fi_api.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Expected dotted FlashInfer API path, got {fi_api!r}")
    signature = ", ".join(args)
    call_args = ", ".join(args)
    return textwrap.dedent(f"""\
        import importlib

        _FN = getattr(importlib.import_module({module_name!r}), {attr_name!r})


        def run({signature}):
            return _FN({call_args})
        """) + "\n"


def direct_function_solution(
    definition: dict,
    *,
    fi_api: str | None = None,
    args: list[str] | None = None,
    description: str | None = None,
) -> dict | None:
    """Build a generic positional FlashInfer function baseline solution."""
    api = fi_api or fi_api_from_tags(definition)
    if not api:
        return None
    names = args or input_names(definition)
    if not names:
        return None
    source = direct_function_source(api, names)
    return solution_payload(definition, source, description=description)
