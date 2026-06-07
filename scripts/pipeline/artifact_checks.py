"""Dataset artifact existence checks used by definition indexing."""

from __future__ import annotations

from pathlib import Path


def relpath_for_index(path: Path, root: Path) -> str:
    """Return a stable display path relative to the run root when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def has_any_file(path: Path) -> bool:
    """Return True if path is a file or contains any file."""
    return path.exists() and (path.is_file() or any(p.is_file() for p in path.rglob("*")))


def has_blob_for_definition(run_root: Path, op_type: str, name: str) -> bool:
    """Return True if a workload blob exists for one definition."""
    return any(
        has_any_file(path)
        for path in (
            run_root / "blob" / "workloads" / op_type / name,
            run_root / "blob" / op_type / name,
        )
    )


def has_baseline_solution(solutions_dir: Path, op_type: str, name: str) -> bool:
    """Return True if a baseline solution exists for one definition."""
    return any(
        has_any_file(path)
        for path in (
            solutions_dir / "baseline" / op_type / name,
            solutions_dir / "baseline" / op_type / f"{name}.json",
        )
    )


def has_eval_trace(run_root: Path, op_type: str, name: str) -> bool:
    """Return True if a baseline eval trace exists for one definition."""
    return any(
        path.exists()
        for path in (
            run_root / "traces" / "baseline" / op_type / f"{name}.jsonl",
            run_root / "traces" / op_type / f"{name}.jsonl",
        )
    )
