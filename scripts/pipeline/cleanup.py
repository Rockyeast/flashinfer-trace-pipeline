"""Cleanup helpers for pipeline run directories."""

from __future__ import annotations

import atexit
import shutil
from collections.abc import Callable
from pathlib import Path


Log = Callable[[str], None]


def run_root_has_reference_artifacts(run_root: Path) -> bool:
    """Return whether a run root has artifacts worth keeping for review/debug."""
    keep_file_names = {
        "kernel_inventory.json",
        "static_kernel_inventory.json",
        "aggregated_summary.json",
        "probe_run.json",
        # Backward-compatible with failed runs created before probe_run.json.
        "modal_function_call_id.txt",
        "modal_probe_run.json",
        "probe_error.json",
        "collector_diagnostics.json",
    }
    keep_suffixes = (
        ".jsonl",
        ".log",
        ".txt",
    )
    keep_path_parts = {
        "definitions",
        "workloads",
        "solutions",
        "tests",
        "reports",
        "logs",
        "probe",
    }

    for path in run_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "definition_index.json":
            continue
        if path.name in keep_file_names:
            return True
        if path.suffix in keep_suffixes:
            return True
        if keep_path_parts.intersection(path.relative_to(run_root).parts):
            return True
    return False


def remove_uninformative_run_root(run_root: Path, log: Log) -> bool:
    """Remove a run root that contains no review/debug-worthy artifacts."""
    if not run_root.exists() or run_root_has_reference_artifacts(run_root):
        return False
    try:
        shutil.rmtree(run_root)
        log(f"🧹 Removed uninformative run output: {run_root}")
        return True
    except OSError as exc:
        log(f"⚠️  Could not remove run output {run_root}: {exc}")
        return False


class FailedRunCleanup:
    """Remove an empty run root if the pipeline exits before completion."""

    def __init__(self, log: Log) -> None:
        self._log = log
        self._run_root: Path | None = None
        self._completed = False

    def register(self, run_root: Path | None) -> None:
        if run_root is None:
            return
        self._run_root = run_root
        atexit.register(self._cleanup_failed_run_root)

    def mark_completed(self) -> None:
        self._completed = True

    def prune_uninformative_run_root(self) -> None:
        if self._run_root is None:
            return
        remove_uninformative_run_root(self._run_root, self._log)

    def _cleanup_failed_run_root(self) -> None:
        if self._completed or self._run_root is None:
            return
        run_root = self._run_root
        remove_uninformative_run_root(run_root, self._log)
