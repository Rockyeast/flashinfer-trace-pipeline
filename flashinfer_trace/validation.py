"""Validation and review reporting for one trace run.

Covers both the lightweight local consistency checks and the wrapper around
upstream flashinfer-bench dataset validation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from flashinfer_trace.core.planning import (
    build_probe_plan,
    load_approved_targets,
    load_definitions,
)
from flashinfer_trace.core.schemas import ApprovedTarget


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"run report must be an object: {path}")
    return payload


def _collect_definition_names(collect: dict[str, Any]) -> set[str]:
    plan = collect.get("plan") if isinstance(collect.get("plan"), dict) else {}
    targets = plan.get("targets") if isinstance(plan.get("targets"), list) else []
    names: set[str] = set()
    for target in targets:
        if isinstance(target, dict) and isinstance(target.get("definition_name"), str):
            names.add(target["definition_name"])
    return names


def _audited_definition_items(audit: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind in ("passed", "repaired"):
        raw_items = audit.get(kind)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            name = item["name"]
            if name in seen:
                continue
            seen.add(name)
            copied = {
                "name": name,
                "kind": kind,
                "reason": "audited definition was not selected by collect_plan",
            }
            for key in ("source_name", "path", "source", "hints_path", "repair"):
                if isinstance(item.get(key), str):
                    copied[key] = item[key]
            items.append(copied)
    return sorted(items, key=lambda item: item["name"])


def _uncollected_definitions(audit: dict[str, Any], collect: dict[str, Any]) -> list[dict[str, Any]]:
    collect_names = _collect_definition_names(collect)
    if not collect_names:
        return []
    return [
        item
        for item in _audited_definition_items(audit)
        if item["name"] not in collect_names
    ]


def update_run_report(run_dir: Path, **sections: Any) -> dict[str, Any]:
    """Update the canonical machine-readable report for one run."""
    report_path = run_dir / "reports" / "run_report.json"
    report = _load_json_if_exists(report_path)
    report.setdefault("version", 1)
    report["run_dir"] = str(run_dir)
    for key, value in sections.items():
        if value is not None:
            report[key] = value

    internal = report.get("internal_validation") if isinstance(report.get("internal_validation"), dict) else {}
    official = report.get("official_validation") if isinstance(report.get("official_validation"), dict) else {}
    export = report.get("export") if isinstance(report.get("export"), dict) else {}
    collect = report.get("collect") if isinstance(report.get("collect"), dict) else {}
    audit = report.get("definition_audit") if isinstance(report.get("definition_audit"), dict) else {}
    remote = report.get("remote") if isinstance(report.get("remote"), dict) else {}

    internal_summary = internal.get("summary") if isinstance(internal.get("summary"), dict) else {}
    export_summary = export.get("summary") if isinstance(export.get("summary"), dict) else {}
    collect_manifest = collect.get("manifest") if isinstance(collect.get("manifest"), dict) else {}
    collect_summary = collect_manifest.get("summary") if isinstance(collect_manifest.get("summary"), dict) else {}
    audit_summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    remote_summary = remote.get("summary") if isinstance(remote.get("summary"), dict) else {}
    uncollected = _uncollected_definitions(audit, collect)
    diagnostics = report.get("diagnostics") if isinstance(report.get("diagnostics"), dict) else {}
    diagnostics = dict(diagnostics)
    if uncollected:
        diagnostics["uncollected_definitions"] = uncollected
    else:
        diagnostics.pop("uncollected_definitions", None)
    if diagnostics:
        report["diagnostics"] = diagnostics
    else:
        report.pop("diagnostics", None)

    internal_ok = internal_summary.get("ok")
    official_ok = official.get("ok")
    export_ok = export_summary.get("ok")
    accepted = bool(internal_ok) and (official_ok is not False) and (export_ok is not False)
    if official_ok is None:
        accepted = False

    report["summary"] = {
        "accepted": accepted,
        "internal_ok": internal_ok,
        "official_ok": official_ok,
        "export_ok": export_ok,
        "errors": internal_summary.get("errors", 0),
        "warnings": internal_summary.get("warnings", 0),
        "workloads": collect_summary.get("workloads", 0),
        "captures": collect_summary.get("captures", 0),
        "sanitized": collect_summary.get("sanitized", 0),
        "definition_audit_repaired": audit_summary.get("repaired", 0),
        "definition_audit_rejected": audit_summary.get("rejected", 0),
        "uncollected_definitions": len(uncollected),
        "early_stopped": bool(remote_summary.get("early_stopped")),
    }
    if remote_summary.get("early_stop_reason"):
        report["summary"]["early_stop_reason"] = remote_summary["early_stop_reason"]
    _write_json(report_path, report)
    (run_dir / "reports" / "review.md").write_text(render_run_review_markdown(report), encoding="utf-8")
    return report


def render_run_review_markdown(report: dict[str, Any]) -> str:
    """Render the human-facing review view from ``run_report.json``."""
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    internal = report.get("internal_validation") if isinstance(report.get("internal_validation"), dict) else {}
    official = report.get("official_validation") if isinstance(report.get("official_validation"), dict) else {}
    collect = report.get("collect") if isinstance(report.get("collect"), dict) else {}
    audit = report.get("definition_audit") if isinstance(report.get("definition_audit"), dict) else {}
    export = report.get("export") if isinstance(report.get("export"), dict) else {}
    remote = report.get("remote") if isinstance(report.get("remote"), dict) else {}
    diagnostics = report.get("diagnostics") if isinstance(report.get("diagnostics"), dict) else {}

    internal_summary = internal.get("summary") if isinstance(internal.get("summary"), dict) else {}
    collect_manifest = collect.get("manifest") if isinstance(collect.get("manifest"), dict) else {}
    collect_summary = collect_manifest.get("summary") if isinstance(collect_manifest.get("summary"), dict) else {}
    audit_summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    export_summary = export.get("summary") if isinstance(export.get("summary"), dict) else {}
    findings = internal.get("findings") if isinstance(internal.get("findings"), list) else []
    remote_summary = remote.get("summary") if isinstance(remote.get("summary"), dict) else {}

    lines = [
        "# Trace Run Review",
        "",
        "## Acceptance",
        "",
        f"- accepted: {summary.get('accepted')}",
        f"- internal validation: {summary.get('internal_ok')}",
        f"- official validation: {summary.get('official_ok')}",
        f"- export: {summary.get('export_ok')}",
        f"- early stopped: {summary.get('early_stopped')}",
        "",
        "## Collect",
        "",
        f"- workloads: {collect_summary.get('workloads', 0)}",
        f"- captures: {collect_summary.get('captures', 0)}",
        f"- workload files: {collect_summary.get('workload_files', 0)}",
        f"- sanitized entries: {collect_summary.get('sanitized', 0)}",
        "",
        "## Definition Audit",
        "",
        f"- raw: {audit_summary.get('raw', 0)}",
        f"- passed: {audit_summary.get('passed', 0)}",
        f"- repaired: {audit_summary.get('repaired', 0)}",
        f"- rejected: {audit_summary.get('rejected', 0)}",
        "",
    ]
    if remote_summary.get("early_stop_reason"):
        lines.extend([
            "## Remote",
            "",
            f"- early stop reason: {remote_summary.get('early_stop_reason')}",
            "",
        ])
    repaired_items = audit.get("repaired") if isinstance(audit.get("repaired"), list) else []
    if repaired_items:
        lines.extend(["### Repaired Definitions", ""])
        for item in repaired_items:
            if not isinstance(item, dict):
                continue
            source_name = item.get("source_name", "<unknown>")
            name = item.get("name", "<unknown>")
            reason = item.get("reason", "")
            repair = item.get("repair", "")
            detail = f"{source_name} -> {name}"
            suffix = ", ".join(str(x) for x in [repair, reason] if x)
            if suffix:
                detail = f"{detail} ({suffix})"
            lines.append(f"- {detail}")
        lines.append("")
    uncollected_items = (
        diagnostics.get("uncollected_definitions")
        if isinstance(diagnostics.get("uncollected_definitions"), list)
        else []
    )
    if uncollected_items:
        lines.extend(["### Uncollected Definitions", ""])
        lines.append(f"- count: {len(uncollected_items)}")
        lines.append("")
        for item in uncollected_items:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "<unknown>")
            kind = item.get("kind", "unknown")
            reason = item.get("reason", "")
            lines.append(f"- {name} ({kind}) - {reason}")
        lines.append("")
    reviewed_overwrites = audit.get("reviewed_overwrites") if isinstance(audit.get("reviewed_overwrites"), dict) else {}
    overwritten_definitions = reviewed_overwrites.get("overwritten_definitions") or []
    overwritten_hints = reviewed_overwrites.get("overwritten_hints") or []
    if overwritten_definitions or overwritten_hints:
        lines.extend(["### Reviewed Overwrites", ""])
        if overwritten_definitions:
            lines.append(f"- overwritten definitions: {len(overwritten_definitions)}")
            for p in overwritten_definitions:
                lines.append(f"  - {p}")
        if overwritten_hints:
            lines.append(f"- overwritten hints: {len(overwritten_hints)}")
            for p in overwritten_hints:
                lines.append(f"  - {p}")
        lines.append("")
    lines.extend([
        "## Dataset View",
        "",
        f"- definitions: {export_summary.get('definitions', 0)}",
        f"- missing definitions: {export_summary.get('missing_definitions', 0)}",
        "",
        "## Internal Findings",
        "",
        f"- errors: {internal_summary.get('errors', 0)}",
        f"- warnings: {internal_summary.get('warnings', 0)}",
        "",
    ])
    if not findings:
        lines.append("- none")
    else:
        for finding in findings:
            if isinstance(finding, dict):
                lines.append(
                    f"- {finding.get('severity', 'unknown')}: "
                    f"{finding.get('name', '<unknown>')} - {finding.get('reason', '')}"
                )
    lines.extend([
        "",
        "## Official Validation",
        "",
        f"- ok: {official.get('ok')}",
        f"- returncode: {official.get('returncode')}",
        "",
    ])
    return "\n".join(lines)


def export_run_dataset(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare an official-style dataset root from one reviewed run.

    By default the run root's ``output`` directory is the official-style staging root:
    ``definitions/``, ``workloads/``, and ``blob/`` live directly under that
    directory. Passing a different ``output_dir`` copies those files out for
    promotion or ad-hoc validation.
    """
    output_root = run_dir / "output"
    output_dir = output_dir or output_root
    same_root = output_dir.resolve() == output_root.resolve()
    if output_dir.exists() and not same_root:
        if not overwrite:
            raise FileExistsError(f"dataset output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _load_json_if_exists(run_dir / "reports" / "run_report.json")
    collect_section = report.get("collect") if isinstance(report.get("collect"), dict) else {}
    collect_plan = collect_section.get("plan") if isinstance(collect_section.get("plan"), dict) else None
    if collect_plan is None:
        raise FileNotFoundError(f"collect plan not found in run report: {run_dir / 'reports' / 'run_report.json'}")
    if not isinstance(collect_plan, dict):
        raise ValueError("collect plan must be an object")

    copied_definitions: list[dict[str, str]] = []
    missing_definitions: list[dict[str, str]] = []
    referenced_definition_paths: set[Path] = set()
    for target in collect_plan.get("targets", []):
        if not isinstance(target, dict):
            continue
        definition_path = target.get("definition_path")
        op_type = target.get("op_type")
        definition_name = target.get("definition_name")
        if not all(isinstance(item, str) and item for item in (definition_path, op_type, definition_name)):
            continue
        src = Path(definition_path)
        dst = output_dir / "definitions" / op_type / f"{definition_name}.json"
        if not src.exists():
            missing_definitions.append({"name": definition_name, "path": str(src)})
            continue
        referenced_definition_paths.add(dst.resolve())
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            if dst.exists() and not overwrite:
                pass
            else:
                shutil.copyfile(src, dst)
        copied_definitions.append({"name": definition_name, "source": str(src), "destination": str(dst)})

    pruned_definitions: list[str] = []
    definitions_root = output_dir / "definitions"
    if same_root and definitions_root.exists():
        for path in sorted(definitions_root.rglob("*.json")):
            if path.resolve() in referenced_definition_paths:
                continue
            pruned_definitions.append(str(path))
            path.unlink()
        for path in sorted(definitions_root.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

    copied_dirs: list[dict[str, str]] = []
    for dirname in ("workloads", "blob"):
        src_dir = output_root / dirname
        dst_dir = output_dir / dirname
        if not src_dir.exists():
            continue
        if src_dir.resolve() != dst_dir.resolve():
            if dst_dir.exists():
                if not overwrite:
                    raise FileExistsError(f"dataset output already exists: {dst_dir}")
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
        copied_dirs.append({"source": str(src_dir), "destination": str(dst_dir)})

    manifest = {
        "summary": {
            "definitions": len(copied_definitions),
            "missing_definitions": len(missing_definitions),
            "pruned_definitions": len(pruned_definitions),
            "copied_dirs": len(copied_dirs),
            "ok": len(missing_definitions) == 0,
        },
        "run_dir": str(run_dir),
        "dataset_dir": str(output_dir),
        "definitions": copied_definitions,
        "missing_definitions": missing_definitions,
        "pruned_definitions": pruned_definitions,
        "copied_dirs": copied_dirs,
    }
    return manifest


def _names_from_items(items: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(items, list):
        return names
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(item["name"])
    return names


def _collectable_approved_names(targets: list[ApprovedTarget]) -> set[str]:
    return {
        target.name
        for target in targets
        if target.role != "warmup" and target.collect and target.backend == "flashinfer"
    }


def _validate_workloads(
    manifest: dict[str, Any],
    definition_names: set[str],
    findings: list[dict[str, str]],
) -> None:
    workloads = manifest.get("workloads", [])
    if isinstance(workloads, list):
        for workload in workloads:
            if not isinstance(workload, dict):
                continue
            name = str(workload.get("name") or "<unknown>")
            definition_name = workload.get("definition_name") or workload.get("name")
            if isinstance(definition_name, str) and definition_name not in definition_names:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": f"workload has no matching definition: {definition_name}",
                })
            workload_paths = workload.get("workload_paths", [])
            if not isinstance(workload_paths, list) or not workload_paths:
                findings.append({"severity": "warning", "name": name, "reason": "workload has no workload JSONL files"})
                continue
            for raw_path in workload_paths:
                if isinstance(raw_path, str) and not Path(raw_path).exists():
                    findings.append({"severity": "error", "name": name, "reason": f"workload JSONL missing: {raw_path}"})
            blob_paths = workload.get("blob_paths", [])
            real_inputs = workload.get("real_inputs", [])
            if not isinstance(blob_paths, list) or (not blob_paths and real_inputs):
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "workload requested real inputs but has no blob files",
                })
                continue
            for raw_path in blob_paths:
                if isinstance(raw_path, str) and not Path(raw_path).exists():
                    findings.append({"severity": "error", "name": name, "reason": f"workload blob missing: {raw_path}"})


def validate_run(
    *,
    approved_targets_path: Path,
    definitions_dir: Path,
    parse_report: dict[str, Any] | None = None,
    parse_report_path: Path | None = None,
    workload_manifest_path: Path | None = None,
    workload_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate consistency between reviewed targets, definitions, and runtime outputs."""
    approved_targets = load_approved_targets(approved_targets_path)
    definitions = load_definitions(definitions_dir)

    approved_names = {
        target.name
        for target in approved_targets
        if target.role != "warmup"
    }
    definition_names = set(definitions)
    findings: list[dict[str, str]] = []

    for target in approved_targets:
        if target.role == "warmup":
            continue
        if not target.target:
            findings.append({"severity": "error", "name": target.name, "reason": "approved target has no target API"})
            continue
        if target.definition_source == "fitrace" and target.backend == "flashinfer" and target.collect:
            probe_plan = build_probe_plan([target])
            if not probe_plan.targets:
                findings.append({
                    "severity": "error",
                    "name": target.name,
                    "reason": "approved fitrace target is not probeable",
                })
            continue
        if not target.collect:
            continue
        definition_name = target.definition_name or target.name
        if definition_name not in definition_names:
            findings.append({
                "severity": "error",
                "name": target.name,
                "reason": f"approved target has no matching definition: {definition_name}",
            })

    parse_summary: dict[str, Any] | None = None
    if parse_report is None and parse_report_path:
        parse_report = _load_json(parse_report_path)
    if parse_report is not None:
        if not isinstance(parse_report, dict):
            findings.append({"severity": "error", "name": "parse_report", "reason": "parse report is not an object"})
        else:
            parse_summary = parse_report.get("summary") if isinstance(parse_report.get("summary"), dict) else None
            observed_names = _names_from_items(parse_report.get("observed_targets"))
            missing_names = set(parse_report.get("missing_targets", [])) if isinstance(parse_report.get("missing_targets"), list) else set()
            for name in sorted(approved_names - observed_names):
                findings.append({"severity": "warning", "name": name, "reason": "approved target not observed in non-warmup events"})
            for name in sorted(missing_names & approved_names):
                findings.append({"severity": "warning", "name": name, "reason": "parse report marked target missing"})

    workload_summary: dict[str, Any] | None = None
    if workload_manifest is not None:
        manifest = workload_manifest
        if not isinstance(manifest, dict):
            findings.append({"severity": "error", "name": "workload_manifest", "reason": "workload manifest is not an object"})
        else:
            workload_summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else None
            workload_names = _names_from_items(manifest.get("workloads"))
            for name in sorted(workload_names - approved_names):
                findings.append({"severity": "error", "name": name, "reason": "workload generated for non-approved target"})
            expected_workload_names = _collectable_approved_names(approved_targets)
            for name in sorted(expected_workload_names - workload_names):
                findings.append({"severity": "error", "name": name, "reason": "expected collect target has no workload"})
            _validate_workloads(manifest, definition_names, findings)
    elif workload_manifest_path:
        manifest = _load_json(workload_manifest_path)
        if not isinstance(manifest, dict):
            findings.append({"severity": "error", "name": str(workload_manifest_path), "reason": "workload manifest is not an object"})
        else:
            workload_summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else None
            workload_names = _names_from_items(manifest.get("workloads"))
            for name in sorted(workload_names - approved_names):
                findings.append({"severity": "error", "name": name, "reason": "workload generated for non-approved target"})
            expected_workload_names = _collectable_approved_names(approved_targets)
            for name in sorted(expected_workload_names - workload_names):
                findings.append({"severity": "error", "name": name, "reason": "expected collect target has no workload"})
            _validate_workloads(manifest, definition_names, findings)

    error_count = sum(1 for finding in findings if finding["severity"] == "error")
    warning_count = sum(1 for finding in findings if finding["severity"] == "warning")
    return {
        "summary": {
            "approved_targets": len(approved_names),
            "definitions": len(definitions),
            "collectable_targets": len(_collectable_approved_names(approved_targets)),
            "errors": error_count,
            "warnings": warning_count,
            "ok": error_count == 0,
        },
        "parse_summary": parse_summary,
        "workload_summary": workload_summary,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Upstream flashinfer-bench dataset validation
# ---------------------------------------------------------------------------


def _build_official_validate_command(
    *,
    dataset_dir: Path,
    checks: str,
    outputs: str,
    output_folder: Path | None,
    disable_gpu: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-c",
        "from flashinfer_bench.cli.main import cli; cli()",
        "validate",
        "--dataset",
        str(dataset_dir),
        "--checks",
        checks,
        "--outputs",
        outputs,
    ]
    if disable_gpu:
        command.append("--disable-gpu")
    if output_folder is not None:
        command.extend(["--output-folder", str(output_folder)])
    return command


def run_official_validate(
    *,
    dataset_dir: Path,
    output_dir: Path,
    checks: str = "layout,definition,workload",
    outputs: str = "stdout,json,text",
    output_folder: Path | None = None,
    disable_gpu: bool = True,
) -> dict[str, Any]:
    """Run flashinfer-bench validation and return a machine-readable report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "official_validate_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    effective_output_folder = output_folder or (output_dir / "official_validate_outputs")

    command = _build_official_validate_command(
        dataset_dir=dataset_dir,
        checks=checks,
        outputs=outputs,
        output_folder=effective_output_folder,
        disable_gpu=disable_gpu,
    )
    report: dict[str, Any] = {
        "command": command,
        "dataset_dir": str(dataset_dir),
        "checks": checks,
        "outputs": outputs,
        "output_folder": str(effective_output_folder),
        "disable_gpu": disable_gpu,
    }

    try:
        import flashinfer_bench  # noqa: F401
    except ImportError as exc:
        report.update({
            "returncode": 1,
            "ok": False,
            "error": "flashinfer-bench is not installed",
            "detail": str(exc),
        })
        return report

    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache_root)
    env["FLASHINFER_WORKSPACE_BASE"] = str(cache_root)
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    output_text = f"{result.stdout}\n{result.stderr}".lower()
    output_has_errors = bool(re.search(r"\b[1-9]\d*\s+error\b", output_text)) or any(
        marker in output_text
        for marker in ("[error]", "parse error", "validation error", "cannot validate workloads")
    )
    report.update({
        "returncode": result.returncode,
        "ok": result.returncode == 0 and not output_has_errors,
        "stdout": result.stdout,
        "stderr": result.stderr,
    })
    if output_has_errors:
        report["error"] = "official validator output contains errors"
    (output_dir / "official_validate.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "official_validate.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return report
