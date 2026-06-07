"""Build static candidate kernel inventory entries."""

from __future__ import annotations

from pathlib import Path

from adapters import ADAPTERS
from parse.inventory_helpers import (
    KERNEL_STATE_EXISTING,
    KERNEL_STATE_KEY,
    KERNEL_STATE_NEEDS_CONFIG,
    KERNEL_STATE_NEW,
)


def _status(definition_name: str, existing_defs: set[str]) -> str:
    if "NEEDS_CONFIG" in definition_name:
        return KERNEL_STATE_NEEDS_CONFIG
    if definition_name in existing_defs:
        return KERNEL_STATE_EXISTING
    return KERNEL_STATE_NEW


def _entry(
    *,
    op_type: str,
    variant: str,
    definition_name: str,
    params: dict,
    existing_defs: set[str],
    evidence: dict,
    fi_api: str | None = None,
) -> dict:
    return {
        "op_type": op_type,
        "variant": variant,
        "fi_api": fi_api,
        "params": params,
        "definition_name": definition_name,
        KERNEL_STATE_KEY: _status(definition_name, existing_defs),
        "source": "static",
        "runtime_status": "static_candidate",
        "evidence": evidence,
    }


def _entry_from_candidate(candidate: dict, *, existing_defs: set[str], evidence: dict) -> dict | None:
    """Convert an adapter static candidate into a static inventory entry."""
    required = ("op_type", "variant", "params", "definition_name")
    if any(not candidate.get(key) for key in required):
        return None
    entry_evidence = dict(evidence)
    params = candidate["params"]
    if isinstance(params, dict) and params.get("page_size") is not None:
        entry_evidence["page_size"] = params["page_size"]
        entry_evidence["page_size_source"] = candidate.get(
            "page_size_source",
            evidence.get("page_size_source", "unknown"),
        )
    return _entry(
        op_type=candidate["op_type"],
        variant=candidate["variant"],
        fi_api=candidate.get("fi_api"),
        params=params,
        definition_name=candidate["definition_name"],
        existing_defs=existing_defs,
        evidence=entry_evidence,
    )


def collect_existing_definitions(definitions_dir: Path | None) -> set[str]:
    """Return all existing definition names under definitions_dir."""
    if definitions_dir is None or not definitions_dir.exists():
        return set()
    return {p.stem for p in definitions_dir.rglob("*.json")}


def build_static_kernels(
    *,
    dims: dict,
    analysis: dict,
    model_name: str,
    tp: int,
    definitions_dir: Path | None,
    page_size: int = 1,
    page_size_source: str = "sglang_default",
) -> list[dict]:
    """Build candidate kernel entries from static model/config facts."""
    existing_defs = collect_existing_definitions(definitions_dir)
    evidence = {
        "type": "static",
        "model": model_name,
        "tp": tp,
        "model_file": analysis.get("model_file"),
    }
    kernels: list[dict] = []

    for adapter in ADAPTERS:
        build_candidates = getattr(adapter, "static_candidates", None)
        if build_candidates is None:
            continue
        for candidate in build_candidates(
            dims,
            analysis,
            tp=tp,
            page_size=page_size,
        ):
            candidate.setdefault("page_size_source", page_size_source)
            entry = _entry_from_candidate(
                candidate,
                existing_defs=existing_defs,
                evidence=evidence,
            )
            if entry is not None:
                kernels.append(entry)

    return kernels
