#!/usr/bin/env python3
"""Clean local pipeline/dev output directories.

The command is conservative by default: it prints what would be removed and
only deletes files when --apply is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline.cleanup import run_root_has_reference_artifacts  # noqa: E402


@dataclass(frozen=True)
class RemovalCandidate:
    path: Path
    reason: str


def _is_generated_artifact_check(path: Path) -> bool:
    return path.name.startswith("generated_artifact_check_")


def _is_known_throwaway_run(path: Path) -> bool:
    prefixes = (
        "dry_",
        "dummy_model_",
        "test_model_",
        "tests_",
        "solutions_",
    )
    return path.name.startswith(prefixes) or _is_generated_artifact_check(path)


def _run_root_file_count(path: Path) -> int:
    return sum(1 for child in path.rglob("*") if child.is_file())


def find_run_candidates(run_root: Path) -> list[RemovalCandidate]:
    candidates: list[RemovalCandidate] = []
    if not run_root.exists():
        return candidates

    for path in sorted(child for child in run_root.iterdir() if child.is_dir()):
        if _is_known_throwaway_run(path):
            candidates.append(RemovalCandidate(path, "known throwaway run name"))
            continue
        if _run_root_file_count(path) == 0:
            candidates.append(RemovalCandidate(path, "empty run directory"))
            continue
        if not run_root_has_reference_artifacts(path):
            candidates.append(RemovalCandidate(path, "no reference/debug artifacts"))
    return candidates


def find_empty_directories(root: Path) -> list[RemovalCandidate]:
    if not root.exists():
        return []
    return [
        RemovalCandidate(path, "empty directory")
        for path in sorted(root.rglob("*"), reverse=True)
        if path.is_dir() and not any(path.iterdir())
    ]


def find_dev_check_candidates(dev_checks_root: Path) -> list[RemovalCandidate]:
    if not dev_checks_root.exists():
        return []
    return [RemovalCandidate(dev_checks_root, "local dev check outputs")]


def find_cache_candidates(tmp_root: Path, include_caches: bool) -> list[RemovalCandidate]:
    if not include_caches:
        return []
    candidates: list[RemovalCandidate] = []
    official_cache = tmp_root / "official_validate_cache"
    if official_cache.exists():
        candidates.append(RemovalCandidate(official_cache, "official validation cache"))
    return candidates


def find_fi_smoke_candidates(tmp_root: Path, include_fi_smoke: bool) -> list[RemovalCandidate]:
    if not include_fi_smoke:
        return []
    fi_smoke = tmp_root / "fi_smoke"
    if fi_smoke.exists():
        return [RemovalCandidate(fi_smoke, "legacy fi_smoke output")]
    return []


def _dedupe(candidates: list[RemovalCandidate]) -> list[RemovalCandidate]:
    seen: set[Path] = set()
    result: list[RemovalCandidate] = []
    for candidate in candidates:
        path = candidate.path.resolve()
        if path in seen:
            continue
        if any(parent in seen for parent in path.parents):
            continue
        seen.add(path)
        result.append(candidate)
    return result


def _remove(candidate: RemovalCandidate) -> None:
    if candidate.path.is_dir():
        shutil.rmtree(candidate.path)
    else:
        candidate.path.unlink()


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmp-root", type=Path, default=PROJECT_ROOT / "tmp")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "tmp" / "run")
    parser.add_argument("--dev-checks-root", type=Path, default=PROJECT_ROOT / ".dev_checks")
    parser.add_argument("--include-caches", action="store_true", help="Also remove local caches.")
    parser.add_argument(
        "--include-fi-smoke",
        action="store_true",
        help="Also remove legacy tmp/fi_smoke outputs.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually remove candidates.")
    args = parser.parse_args()

    candidates = _dedupe(
        [
            *find_run_candidates(args.run_root),
            *find_empty_directories(args.tmp_root),
            *find_dev_check_candidates(args.dev_checks_root),
            *find_cache_candidates(args.tmp_root, args.include_caches),
            *find_fi_smoke_candidates(args.tmp_root, args.include_fi_smoke),
        ]
    )

    if not candidates:
        print("No local output cleanup candidates found.")
        return 0

    action = "Removing" if args.apply else "Would remove"
    for candidate in candidates:
        rel = _display_path(candidate.path)
        print(f"{action}: {rel}  ({candidate.reason})")
        if args.apply:
            _remove(candidate)

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to remove these paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
