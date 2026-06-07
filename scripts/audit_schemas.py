#!/usr/bin/env python3
"""Audit existing JSON artifacts against strict pipeline schemas."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from artifact_schemas import validate_definition, validate_solution, validate_workload_entry


Validator = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class AuditStats:
    checked: int = 0
    failed: int = 0


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {error.get('msg', 'validation failed')}")
    return "; ".join(parts)


def _print_failure(label: str, message: str, *, max_errors: int, stats: AuditStats) -> None:
    stats.failed += 1
    if stats.failed <= max_errors:
        print(f"[FAIL] {label}: {message}", file=sys.stderr)
    elif stats.failed == max_errors + 1:
        print("[FAIL] further errors suppressed", file=sys.stderr)


def _load_json(path: Path, label: str, stats: AuditStats, max_errors: int) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _print_failure(label, f"invalid JSON: {exc}", max_errors=max_errors, stats=stats)
        return None
    if not isinstance(data, dict):
        _print_failure(label, "top-level JSON value must be an object", max_errors=max_errors, stats=stats)
        return None
    return data


def _audit_json_files(root: Path, pattern: str, validator: Validator, max_errors: int) -> AuditStats:
    stats = AuditStats()
    for path in sorted(root.rglob(pattern)):
        label = str(path)
        data = _load_json(path, label, stats, max_errors)
        if data is None:
            continue
        stats.checked += 1
        try:
            validator(data)
        except ValidationError as exc:
            _print_failure(
                label,
                _format_validation_error(exc),
                max_errors=max_errors,
                stats=stats,
            )
        except Exception as exc:
            _print_failure(label, str(exc), max_errors=max_errors, stats=stats)
    return stats


def _audit_jsonl_files(root: Path, pattern: str, validator: Validator, max_errors: int) -> AuditStats:
    stats = AuditStats()
    for path in sorted(root.rglob(pattern)):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            _print_failure(str(path), f"could not read file: {exc}", max_errors=max_errors, stats=stats)
            continue
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            label = f"{path}:{line_no}"
            try:
                data = json.loads(line)
            except Exception as exc:
                _print_failure(label, f"invalid JSONL row: {exc}", max_errors=max_errors, stats=stats)
                continue
            if not isinstance(data, dict):
                _print_failure(
                    label,
                    "top-level JSONL row must be an object",
                    max_errors=max_errors,
                    stats=stats,
                )
                continue
            stats.checked += 1
            try:
                validator(data)
            except ValidationError as exc:
                _print_failure(
                    label,
                    _format_validation_error(exc),
                    max_errors=max_errors,
                    stats=stats,
                )
            except Exception as exc:
                _print_failure(label, str(exc), max_errors=max_errors, stats=stats)
    return stats


def _print_summary(name: str, stats: AuditStats) -> None:
    status = "PASS" if stats.failed == 0 else "FAIL"
    print(f"{name:12} {status:4} checked={stats.checked} failed={stats.failed}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit JSON artifacts against strict schemas.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository or dataset root to audit.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Maximum detailed failures to print before suppressing further details.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    max_errors = max(1, args.max_errors)

    definitions = _audit_json_files(root / "definitions", "*.json", validate_definition, max_errors)
    workloads = _audit_jsonl_files(root / "workloads", "*.jsonl", validate_workload_entry, max_errors)
    solutions = _audit_json_files(root / "solutions", "*.json", validate_solution, max_errors)

    _print_summary("definitions", definitions)
    _print_summary("workloads", workloads)
    _print_summary("solutions", solutions)

    total_failed = definitions.failed + workloads.failed + solutions.failed
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
