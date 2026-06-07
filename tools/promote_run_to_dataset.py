#!/usr/bin/env python3
"""Promote one pipeline run into a canonical dataset root.

The pipeline isolates artifacts by model and timestamp under tmp/run/. This
tool is the explicit review step that merges one approved run into an
official-style dataset layout:

    definitions/
    workloads/
    blob/workloads/

Definitions are compared structurally. Workloads are deduplicated by their
semantic content, with safetensors inputs represented by file-content hashes
instead of local UUID-based paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_RUN_DIRS = ("definitions", "workloads")
OPTIONAL_RUN_PATHS = ("blob", "kernel_inventory.json", "collector_diagnostics.json")


@dataclass
class PromoteReport:
    source_model: str | None
    run_dir: str
    dataset_dir: str
    baseline_dir: str | None
    dry_run: bool
    definitions_added: list[str] = field(default_factory=list)
    definitions_skipped_existing: list[str] = field(default_factory=list)
    workloads_added: dict[str, int] = field(default_factory=dict)
    workloads_skipped_duplicate: dict[str, int] = field(default_factory=dict)
    blobs_added: list[str] = field(default_factory=list)
    blobs_skipped_existing: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    baseline_definitions_new: list[str] = field(default_factory=list)
    baseline_definitions_same: list[str] = field(default_factory=list)
    baseline_definition_conflicts: list[str] = field(default_factory=list)
    baseline_workload_new: dict[str, int] = field(default_factory=dict)
    baseline_workload_duplicate: dict[str, int] = field(default_factory=dict)
    skipped_uncollected_definitions: list[str] = field(default_factory=list)
    skipped_target_api_definitions: list[str] = field(default_factory=list)
    skipped_target_api_workloads: list[str] = field(default_factory=list)
    skipped_gemm_definitions: list[str] = field(default_factory=list)
    skipped_gemm_workloads: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_model": self.source_model,
            "run_dir": self.run_dir,
            "dataset_dir": self.dataset_dir,
            "baseline_dir": self.baseline_dir,
            "dry_run": self.dry_run,
            "definitions_added": self.definitions_added,
            "definitions_skipped_existing": self.definitions_skipped_existing,
            "workloads_added": self.workloads_added,
            "workloads_skipped_duplicate": self.workloads_skipped_duplicate,
            "blobs_added": self.blobs_added,
            "blobs_skipped_existing": self.blobs_skipped_existing,
            "conflicts": self.conflicts,
            "warnings": self.warnings,
            "baseline_definitions_new": self.baseline_definitions_new,
            "baseline_definitions_same": self.baseline_definitions_same,
            "baseline_definition_conflicts": self.baseline_definition_conflicts,
            "baseline_workload_new": self.baseline_workload_new,
            "baseline_workload_duplicate": self.baseline_workload_duplicate,
            "skipped_uncollected_definitions": self.skipped_uncollected_definitions,
            "skipped_target_api_definitions": self.skipped_target_api_definitions,
            "skipped_target_api_workloads": self.skipped_target_api_workloads,
            "skipped_gemm_definitions": self.skipped_gemm_definitions,
            "skipped_gemm_workloads": self.skipped_gemm_workloads,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_official_fi_definition(path: Path) -> bool:
    """Return True when a definition declares a real flashinfer.* fi_api tag."""
    try:
        defn = _load_json(path)
    except Exception:
        return False
    for tag in defn.get("tags", []):
        if isinstance(tag, str) and tag.startswith("fi_api:flashinfer."):
            return True
    return False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_blob_ref(raw_path: str) -> Path:
    path = raw_path
    if path.startswith("./"):
        path = path[2:]
    return Path(path)


def _iter_safetensor_refs(value: Any) -> list[Path]:
    refs: list[Path] = []
    if isinstance(value, dict):
        if value.get("type") == "safetensors" and isinstance(value.get("path"), str):
            refs.append(_normalize_blob_ref(value["path"]))
        for child in value.values():
            refs.extend(_iter_safetensor_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(_iter_safetensor_refs(child))
    return refs


def _semantic_workload_key(record: dict[str, Any], run_dir: Path, report: PromoteReport) -> str:
    """Build a dedupe key independent of UUIDs and local blob path names."""

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            if value.get("type") == "safetensors" and isinstance(value.get("path"), str):
                rel = _normalize_blob_ref(value["path"])
                src = run_dir / rel
                if src.exists():
                    digest = _sha256_file(src)
                else:
                    digest = f"missing:{rel.as_posix()}"
                    warning = f"Referenced blob does not exist: {rel.as_posix()}"
                    if warning not in report.warnings:
                        report.warnings.append(warning)
                return {
                    "type": "safetensors",
                    "tensor_key": value.get("tensor_key"),
                    "sha256": digest,
                }
            return {k: normalize(v) for k, v in sorted(value.items()) if k not in {"uuid"}}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value

    return hashlib.sha256(_canonical_json(normalize(record)).encode("utf-8")).hexdigest()


def _load_existing_workload_keys(path: Path, dataset_dir: Path, report: PromoteReport) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                report.conflicts.append(f"Invalid JSONL in existing workload {path}:{lineno}: {exc}")
                continue
            keys.add(_semantic_workload_key(record, dataset_dir, report))
    return keys


def _copy_file_with_conflict_check(
    src: Path,
    dst: Path,
    rel_label: str,
    report: PromoteReport,
    dry_run: bool,
) -> bool:
    """Copy src to dst if needed. Return True when dst becomes available."""
    if not src.exists():
        report.conflicts.append(f"Missing source file: {src}")
        return False
    if dst.exists():
        if _sha256_file(src) == _sha256_file(dst):
            if rel_label not in report.blobs_skipped_existing:
                report.blobs_skipped_existing.append(rel_label)
            return True
        report.conflicts.append(f"Blob path conflict with different content: {rel_label}")
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if rel_label not in report.blobs_added:
        report.blobs_added.append(rel_label)
    return True


def validate_run_dir(run_dir: Path) -> list[str]:
    errors = []
    if not run_dir.exists():
        return [f"Run dir does not exist: {run_dir}"]
    for dirname in REQUIRED_RUN_DIRS:
        if not (run_dir / dirname).is_dir():
            errors.append(f"Run dir missing required directory: {dirname}/")
    if not any((run_dir / optional).exists() for optional in OPTIONAL_RUN_PATHS):
        errors.append("Run dir does not look like a pipeline output: no blob/, kernel_inventory.json, or diagnostics")
    return errors


def _collected_definition_names(run_dir: Path) -> set[str]:
    """Return definition names that have at least one collected workload row."""
    names: set[str] = set()
    workloads_dir = run_dir / "workloads"
    if not workloads_dir.is_dir():
        return names
    for path in sorted(workloads_dir.rglob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as f:
                if any(line.strip() for line in f):
                    names.add(path.stem)
        except OSError:
            continue
    return names


def promote_definitions(
    run_dir: Path,
    dataset_dir: Path,
    report: PromoteReport,
    dry_run: bool,
    *,
    include_gemm: bool = False,
    allowed_definition_names: set[str] | None = None,
) -> set[str]:
    promoted_names: set[str] = set()
    for src in sorted((run_dir / "definitions").rglob("*.json")):
        rel = src.relative_to(run_dir)
        dst = dataset_dir / rel
        rel_label = rel.as_posix()
        if src.parent.name == "gemm" and not include_gemm:
            report.skipped_gemm_definitions.append(rel_label)
            continue
        if allowed_definition_names is not None and src.stem not in allowed_definition_names:
            report.skipped_uncollected_definitions.append(rel_label)
            continue
        if not _is_official_fi_definition(src):
            report.skipped_target_api_definitions.append(rel_label)
            continue
        if dst.exists():
            src_obj = _load_json(src)
            dst_obj = _load_json(dst)
            if _canonical_json(src_obj) == _canonical_json(dst_obj):
                report.definitions_skipped_existing.append(rel_label)
                promoted_names.add(src.stem)
            else:
                report.conflicts.append(f"Definition conflict: {rel_label}")
            continue
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        report.definitions_added.append(rel_label)
        promoted_names.add(src.stem)
    return promoted_names


def promote_workloads(
    run_dir: Path,
    dataset_dir: Path,
    report: PromoteReport,
    dry_run: bool,
    *,
    include_gemm: bool = False,
    allowed_definition_names: set[str] | None = None,
) -> None:
    for src in sorted((run_dir / "workloads").rglob("*.jsonl")):
        rel = src.relative_to(run_dir)
        dst = dataset_dir / rel
        rel_label = rel.as_posix()
        if src.parent.name == "gemm" and not include_gemm:
            report.skipped_gemm_workloads.append(rel_label)
            continue
        if allowed_definition_names is not None and src.stem not in allowed_definition_names:
            report.skipped_target_api_workloads.append(rel_label)
            continue
        existing_keys = _load_existing_workload_keys(dst, dataset_dir, report)
        new_lines: list[str] = []
        added = 0
        skipped = 0

        with src.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    report.conflicts.append(f"Invalid JSONL in run workload {src}:{lineno}: {exc}")
                    continue

                key = _semantic_workload_key(record, run_dir, report)
                if key in existing_keys:
                    skipped += 1
                    continue

                refs = _iter_safetensor_refs(record)
                refs_ok = True
                for rel_blob in refs:
                    src_blob = run_dir / rel_blob
                    dst_blob = dataset_dir / rel_blob
                    refs_ok = _copy_file_with_conflict_check(
                        src_blob,
                        dst_blob,
                        rel_blob.as_posix(),
                        report,
                        dry_run,
                    ) and refs_ok
                if not refs_ok:
                    continue

                existing_keys.add(key)
                new_lines.append(_canonical_json(record))
                added += 1

        if added and not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("a", encoding="utf-8") as out:
                if dst.exists() and dst.stat().st_size > 0:
                    out.write("\n")
                out.write("\n".join(new_lines))
                out.write("\n")

        report.workloads_added[rel_label] = added
        report.workloads_skipped_duplicate[rel_label] = skipped


def compare_baseline(
    run_dir: Path,
    baseline_dir: Path,
    report: PromoteReport,
    *,
    include_gemm: bool = False,
) -> None:
    """Compare run artifacts against a read-only baseline dataset root.

    This is intentionally report-only. It does not affect promotion into
    ``dataset_dir`` and does not read from the destination dataset.
    """

    if not baseline_dir.exists():
        report.warnings.append(f"Baseline dir does not exist: {baseline_dir}")
        return

    for src in sorted((run_dir / "definitions").rglob("*.json")):
        rel = src.relative_to(run_dir)
        rel_label = rel.as_posix()
        if src.parent.name == "gemm" and not include_gemm:
            continue
        if not _is_official_fi_definition(src):
            continue
        baseline = baseline_dir / rel
        if not baseline.exists():
            report.baseline_definitions_new.append(rel_label)
            continue
        src_obj = _load_json(src)
        baseline_obj = _load_json(baseline)
        if _canonical_json(src_obj) == _canonical_json(baseline_obj):
            report.baseline_definitions_same.append(rel_label)
        else:
            report.baseline_definition_conflicts.append(rel_label)

    for src in sorted((run_dir / "workloads").rglob("*.jsonl")):
        rel = src.relative_to(run_dir)
        rel_label = rel.as_posix()
        if src.parent.name == "gemm" and not include_gemm:
            continue
        def_src = run_dir / "definitions" / rel.parent.relative_to("workloads") / f"{src.stem}.json"
        if def_src.exists() and not _is_official_fi_definition(def_src):
            continue
        baseline = baseline_dir / rel
        baseline_keys = _load_existing_workload_keys(baseline, baseline_dir, report)
        new = 0
        duplicate = 0

        with src.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    report.warnings.append(f"Invalid JSONL in run workload {src}:{lineno}: {exc}")
                    continue
                key = _semantic_workload_key(record, run_dir, report)
                if key in baseline_keys:
                    duplicate += 1
                else:
                    new += 1

        report.baseline_workload_new[rel_label] = new
        report.baseline_workload_duplicate[rel_label] = duplicate


def write_report(report: PromoteReport, dataset_dir: Path, run_dir: Path, dry_run: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_dir = dataset_dir / "promote_reports"
    report_path = report_dir / f"{run_dir.name}-{stamp}.json"
    if not dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def run_validation(dataset_dir: Path, bench_dir: Path | None) -> int:
    if bench_dir:
        cmd = [
            sys.executable,
            "-c",
            "from flashinfer_bench.cli.main import cli; cli()",
            "validate",
            "--dataset",
            str(dataset_dir),
            "--checks",
            "layout,definition,workload",
            "--outputs",
            "stdout,json,text",
            "--disable-gpu",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(bench_dir)
        return subprocess.run(cmd, env=env).returncode
    cmd = [
        "flashinfer-bench",
        "validate",
        "--dataset",
        str(dataset_dir),
        "--checks",
        "layout,definition,workload",
        "--outputs",
        "stdout,json,text",
        "--disable-gpu",
    ]
    return subprocess.run(cmd).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote one pipeline run into an official-style dataset root.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Pipeline run output directory.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Destination dataset root.")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Optional read-only dataset root used only for duplicate/conflict reporting.",
    )
    parser.add_argument("--source-model", default=None, help="Optional model name recorded in the promote report.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be promoted without writing files.")
    parser.add_argument("--allow-conflicts", action="store_true", help="Write non-conflicting files even if conflicts are found.")
    parser.add_argument("--validate", action="store_true", help="Run flashinfer-bench dataset validation after promotion.")
    parser.add_argument("--bench-dir", type=Path, default=None, help="Optional local flashinfer-bench checkout for validation.")
    parser.add_argument(
        "--include-uncollected-definitions",
        action="store_true",
        help=(
            "Also promote definitions that do not have a collected workload JSONL. "
            "By default, promotion only copies definitions with at least one workload row."
        ),
    )
    parser.add_argument(
        "--include-gemm",
        action="store_true",
        help="Also promote GEMM definitions/workloads. Disabled by default because GEMM has no FlashInfer fi_api.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    dry_run = bool(args.dry_run)

    errors = validate_run_dir(run_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    report = PromoteReport(
        source_model=args.source_model,
        run_dir=str(run_dir),
        dataset_dir=str(dataset_dir),
        baseline_dir=str(args.baseline_dir.resolve()) if args.baseline_dir else None,
        dry_run=dry_run,
    )

    if args.baseline_dir:
        compare_baseline(
            run_dir,
            args.baseline_dir.resolve(),
            report,
            include_gemm=args.include_gemm,
        )
    collected_definition_names = _collected_definition_names(run_dir)
    promoted_names = promote_definitions(
        run_dir,
        dataset_dir,
        report,
        dry_run=dry_run,
        include_gemm=args.include_gemm,
        allowed_definition_names=(
            None if args.include_uncollected_definitions else collected_definition_names
        ),
    )
    promote_workloads(
        run_dir,
        dataset_dir,
        report,
        dry_run=dry_run,
        include_gemm=args.include_gemm,
        allowed_definition_names=promoted_names,
    )

    report_path = write_report(report, dataset_dir, run_dir, dry_run=dry_run)
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    if dry_run:
        print(f"Dry run: report not written. Would write: {report_path}")
    else:
        print(f"Wrote promote report: {report_path}")

    if report.conflicts and not args.allow_conflicts:
        print(f"ERROR: {len(report.conflicts)} conflict(s) found; promotion requires review.", file=sys.stderr)
        return 1

    if args.validate and not dry_run:
        return run_validation(dataset_dir, args.bench_dir.resolve() if args.bench_dir else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
