"""Final pipeline completion summary."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from parse.inventory_helpers import is_new_kernel, needs_config_kernel


Log = Callable[[str], None]


def log_pipeline_summary(
    *,
    inventory_path: Path | None,
    run_output_root: Path | None,
    definitions_dir: Path,
    validate_step: bool,
    official_validate_strict: bool,
    log: Log,
) -> bool:
    """Log final artifact summary and return whether the caller should exit early."""
    log("=" * 60)
    log("Pipeline Complete!")
    log("=" * 60)

    if inventory_path and inventory_path.exists():
        inv = json.loads(inventory_path.read_text(encoding="utf-8"))
        summary = inv.get("summary", {})
        log(
            f"Kernels: {summary.get('total', '?')} total "
            f"({summary.get('existing', '?')} existing, "
            f"{summary.get('new', '?')} new, "
            f"{summary.get('needs_config', '?')} needs_config)"
        )
        if summary.get("deferred", 0):
            log(
                f"Deferred kernels: {summary.get('deferred', 0)} "
                f"({summary.get('deferred_prepared_only', 0)} prepared_only, "
                f"{summary.get('deferred_wrapper_observed', 0)} wrapper_observed)"
            )
        if summary.get("static_candidates", 0):
            log(f"Static-only candidates: {summary.get('static_candidates', 0)}")

        new_kernels = [
            k["definition_name"]
            for k in inv.get("kernels", [])
            if is_new_kernel(k)
        ]
        if new_kernels:
            log("🆕 New kernels (definition will be generated):")
            for name in new_kernels:
                log(f"     {name}")

        needs_config_kernels = [
            k["definition_name"]
            for k in inv.get("kernels", [])
            if needs_config_kernel(k)
        ]
        if needs_config_kernels:
            log("⚠️  NEEDS_CONFIG kernels (skipped — re-run with --hf-config to resolve):")
            for name in needs_config_kernels:
                log(f"     {name}")

    log("")
    log("Next steps:")
    if run_output_root:
        log(f"  1. Review run artifacts in {run_output_root}")
        log("  2. Promote reviewed artifacts into the dataset root with tools/promote_run_to_dataset.py")
        log("  3. Open dataset PRs from the reviewed dataset root")
    elif validate_step:
        log("  1. Review official validation output above")
        if official_validate_strict:
            log("  2. This command was strict; validation failures would have failed the run")
        else:
            log("  2. Re-run with --official-validate-strict if you want validation failures to fail the command")
        return True
    else:
        log(f"  1. Review generated dataset artifacts under {definitions_dir.parent}")
        log("  2. Open dataset PRs from the reviewed dataset root")

    return False
