"""Merge static-only kernel candidates into runtime inventory."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from parse.inventory_helpers import KERNEL_STATE_STATIC_CANDIDATE, set_kernel_state


Log = Callable[[str], None]


def merge_static_candidates_into_inventory(
    runtime_inventory_path: Path,
    static_inventory_path: Path,
    *,
    log: Log,
) -> bool:
    """Add static-only candidates to runtime inventory without changing main kernels."""
    if not runtime_inventory_path.exists() or not static_inventory_path.exists():
        return False

    runtime_inv = json.loads(runtime_inventory_path.read_text(encoding="utf-8"))
    static_inv = json.loads(static_inventory_path.read_text(encoding="utf-8"))

    runtime_names = {
        item.get("definition_name")
        for item in runtime_inv.get("kernels", [])
        if item.get("definition_name")
    }
    runtime_names.update(
        item.get("definition_name")
        for item in runtime_inv.get("deferred_kernels", [])
        if item.get("definition_name")
    )

    static_candidates = []
    for item in static_inv.get("kernels", []):
        name = item.get("definition_name")
        if not name or name in runtime_names:
            continue
        candidate = dict(item)
        set_kernel_state(candidate, KERNEL_STATE_STATIC_CANDIDATE)
        candidate["runtime_status"] = "static_only"
        static_candidates.append(candidate)

    runtime_inv["discovery_source"] = "runtime_with_static"
    runtime_inv["static_inventory_file"] = str(static_inventory_path)
    runtime_inv["static_candidates"] = static_candidates

    summary = runtime_inv.setdefault("summary", {})
    summary["static_candidates"] = len(static_candidates)
    summary["runtime_observed"] = summary.get("observed_run", 0)

    static_summary = static_inv.get("summary", {})
    runtime_inv["static_analysis"] = {
        "candidate_count": len(static_inv.get("kernels", [])),
        "static_only_count": len(static_candidates),
        "page_sizes": static_inv.get("page_sizes"),
        "page_size_source": static_inv.get("page_size_source"),
        "summary": static_summary,
        "source_file": str(static_inventory_path),
    }

    runtime_inventory_path.write_text(
        json.dumps(runtime_inv, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(
        f"✅ Runtime inventory: added {len(static_candidates)} static-only "
        f"candidates from {static_inventory_path}"
    )
    return True
