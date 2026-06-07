"""Definition index assembly for the pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from .artifact_checks import (
    has_baseline_solution,
    has_blob_for_definition,
    has_eval_trace,
    relpath_for_index,
)
from .definition_records import (
    LogFn,
    _noop_log,
    fi_api_from_definition,
    is_collectable_kernel,
)
from .paths import resolve_under_project


def load_fi_trace_staged_names(run_root: Path) -> set[str]:
    """Read staged official fi_trace definition names from a run root."""
    manifest_path = run_root / "probe" / "fi_trace_staged_definitions.json"
    if not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    names: set[str] = set()
    for key in ("staged", "skipped_same"):
        values = manifest.get(key, [])
        if isinstance(values, list):
            names.update(v for v in values if isinstance(v, str) and v)
    return names


def load_static_candidates_for_index(
    run_root: Path,
    definition_names: set[str],
) -> tuple[list[dict], int]:
    """Return static-only candidates that are not already materialized definitions."""
    inventory_path = run_root / "kernel_inventory.json"
    if not inventory_path.exists():
        return [], 0

    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], 0

    candidates: list[dict] = []
    already_defined = 0
    for item in inventory.get("static_candidates", []):
        if not isinstance(item, dict):
            continue
        name = item.get("definition_name")
        if not isinstance(name, str) or not name:
            continue
        if name in definition_names:
            already_defined += 1
            continue

        candidates.append(
            {
                "name": name,
                "op_type": item.get("op_type"),
                "variant": item.get("variant"),
                "fi_api": item.get("fi_api"),
                "target_api": item.get("target_api"),
                "inventory_status": item.get("inventory_status"),
                "runtime_status": item.get("runtime_status", "static_only"),
                "evidence": item.get("evidence"),
                "reason": "static_only_not_runtime_observed",
            }
        )
    return candidates, already_defined


def write_definition_index(
    run_root: Path | None,
    definitions_dir: Path,
    tests_output_dir: Path | None,
    *,
    project_root: Path,
    solutions_dir: Path | None = None,
    log: LogFn = _noop_log,
) -> Path | None:
    """Write a review-only index of definitions and generated artifacts."""
    if run_root is None:
        return None

    run_root = resolve_under_project(run_root, project_root)
    run_root.mkdir(parents=True, exist_ok=True)
    definitions_dir = resolve_under_project(definitions_dir, project_root)
    tests_dir = (
        resolve_under_project(tests_output_dir, project_root)
        if tests_output_dir
        else run_root / "tests" / "references"
    )
    solutions_dir = (
        resolve_under_project(solutions_dir, project_root)
        if solutions_dir
        else run_root / "solutions"
    )
    fi_trace_names = load_fi_trace_staged_names(run_root)

    entries: list[dict] = []
    invalid_definitions: list[dict] = []
    paths_by_name: dict[str, list[str]] = {}

    if definitions_dir.exists():
        for path in sorted(definitions_dir.rglob("*.json")):
            rel_path = relpath_for_index(path, run_root)
            try:
                defn = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                invalid_definitions.append({"path": rel_path, "error": str(exc)})
                continue

            name = defn.get("name")
            op_type = defn.get("op_type")
            if not isinstance(name, str) or not name or not isinstance(op_type, str) or not op_type:
                invalid_definitions.append({"path": rel_path, "error": "missing string name/op_type"})
                continue

            paths_by_name.setdefault(name, []).append(rel_path)
            fi_api = fi_api_from_definition(defn)
            kernel = {
                "definition_name": name,
                "op_type": op_type,
                "fi_api": fi_api or "",
            }

            workload_path = run_root / "workloads" / op_type / f"{name}.jsonl"
            reference_test_path = tests_dir / f"test_{name}.py"
            entries.append(
                {
                    "name": name,
                    "op_type": op_type,
                    "path": rel_path,
                    "source": "official_fi_trace" if name in fi_trace_names else "adapter_or_existing",
                    "fi_api": fi_api,
                    "collectable": is_collectable_kernel(kernel),
                    "has_workload": workload_path.exists(),
                    "has_blob": has_blob_for_definition(run_root, op_type, name),
                    "has_baseline_solution": has_baseline_solution(solutions_dir, op_type, name),
                    "has_eval_trace": has_eval_trace(run_root, op_type, name),
                    "has_reference_test": reference_test_path.exists(),
                }
            )

    duplicate_names = [
        {"name": name, "paths": paths}
        for name, paths in sorted(paths_by_name.items())
        if len(paths) > 1
    ]
    static_candidates, static_candidates_with_definition = load_static_candidates_for_index(
        run_root,
        set(paths_by_name),
    )
    summary = {
        "definitions": len(entries),
        "collectable": sum(1 for item in entries if item["collectable"]),
        "with_workloads": sum(1 for item in entries if item["has_workload"]),
        "with_blob": sum(1 for item in entries if item["has_blob"]),
        "with_baseline_solution": sum(1 for item in entries if item["has_baseline_solution"]),
        "with_eval_trace": sum(1 for item in entries if item["has_eval_trace"]),
        "with_reference_test": sum(1 for item in entries if item["has_reference_test"]),
        "duplicates": len(duplicate_names),
        "invalid_definitions": len(invalid_definitions),
        "static_candidates": len(static_candidates),
        "static_candidates_with_definition": static_candidates_with_definition,
    }

    index = {
        "run_root": str(run_root),
        "definitions_dir": str(definitions_dir),
        "summary": summary,
        "definitions": entries,
        "static_candidates": static_candidates,
        "duplicate_names": duplicate_names,
        "invalid_definitions": invalid_definitions,
    }
    out_path = run_root / "definition_index.json"
    out_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Definition index: {out_path}")
    return out_path
