#!/usr/bin/env python3
"""Smoke-validate one reviewed adapter draft.

This tool validates candidate adapter code without registering it. It imports
``scripts/adapters/_draft_*.py`` directly, builds a synthetic matched-kernel
bucket from the LLM proposal input, asks the draft to build kernels and
definitions, then schema-validates any generated definitions.

Passing this smoke check means the draft is structurally runnable. It does not
prove the adapter is semantically correct, and it never renames or registers the
draft module.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import py_compile
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@dataclass
class StepResult:
    name: str
    status: str
    detail: str = ""


@dataclass
class DraftReport:
    draft: str
    trace_id: str | None
    out_dir: str
    steps: list[StepResult] = field(default_factory=list)
    kernels: list[dict[str, Any]] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.steps.append(StepResult(name, status, detail))

    def failed(self) -> bool:
        return any(step.status == "fail" for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft": self.draft,
            "trace_id": self.trace_id,
            "out_dir": self.out_dir,
            "steps": [step.__dict__ for step in self.steps],
            "kernels": self.kernels,
            "definitions": self.definitions,
            "status": "fail" if self.failed() else "pass",
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_manifest_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        for name in ("apply_manifest.json", "parse_rule_proposals.json"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise ValueError(f"cannot find apply_manifest.json or parse_rule_proposals.json in {path}")
    return path


def _select_proposal(manifest: dict[str, Any], trace_id: str | None) -> dict[str, Any]:
    proposals = manifest.get("proposals") or []
    if trace_id:
        matches = [p for p in proposals if p.get("trace_id") == trace_id]
    else:
        promoted = [p for p in proposals if p.get("review_decision") == "promote"]
        matches = promoted if promoted else proposals
    if not matches:
        raise ValueError("no proposal matched")
    if len(matches) > 1 and not trace_id:
        raise ValueError("multiple proposals matched; pass --trace-id")
    return matches[0]


def _proposal_input_path(manifest_path: Path | None) -> Path | None:
    if manifest_path is None:
        return None
    candidate = manifest_path.parent / "proposal_input.json"
    return candidate if candidate.exists() else None


def _proposal_item(proposal_input: dict[str, Any] | None, trace_id: str | None) -> dict[str, Any]:
    if not proposal_input or not trace_id:
        return {}
    for item in proposal_input.get("items") or []:
        if item.get("trace_id") == trace_id:
            return item
    return {}


def _import_module(path: Path):
    adapters_dir = SCRIPTS_DIR / "adapters"
    if path.parent.resolve() == adapters_dir.resolve():
        module_name = f"adapters.{path.stem}"
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)

    module_name = f"_adapter_draft_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_method(trace_id: str) -> str:
    method = trace_id.rsplit(".", 1)[-1]
    if method in {"run", "forward", "apply", "plan", "begin_forward"}:
        return method
    return "function_call"


def _synthetic_matched_bucket(
    *,
    op_type: str,
    variant: str,
    trace_id: str,
    fi_api: str | None,
    signatures: list[dict[str, Any]],
    count: int,
) -> dict[str, dict[str, Any]]:
    method = _runtime_method(trace_id)
    key = f"{op_type}:{variant}"
    return {
        key: {
            "op_type": op_type,
            "variant": variant,
            "fi_api": fi_api,
            "target_api": None,
            "trace_ids": [trace_id],
            "trace_counts": {trace_id: count},
            "total_count": count,
            "signatures": signatures,
            "runtime_evidence": {method: count},
            "source": "adapter_draft_validation",
            "classification_sources": ["adapter_draft_validation"],
        }
    }


def _default_out_dir(path: Path) -> Path:
    return REPO_ROOT / ".dev_checks" / "adapter_drafts" / path.stem


def _step(report: DraftReport, name: str, func):
    try:
        value = func()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        report.add(name, "fail", detail)
        return None
    report.add(name, "pass")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-validate one adapter draft.")
    parser.add_argument("draft", type=Path, help="Path to scripts/adapters/_draft_<op_type>.py")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Proposal directory, apply_manifest.json, or parse_rule_proposals.json",
    )
    parser.add_argument("--trace-id", default=None, help="Trace ID to validate")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for report/artifacts")
    parser.add_argument("--model-tag", default="draft", help="Model tag passed to generate_definition")
    parser.add_argument("--tp", type=int, default=1, help="TP value passed to generate_definition")
    args = parser.parse_args()

    draft_path = args.draft.resolve()
    out_dir = (args.out_dir or _default_out_dir(draft_path)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = DraftReport(
        draft=str(draft_path.relative_to(REPO_ROOT) if draft_path.is_relative_to(REPO_ROOT) else draft_path),
        trace_id=args.trace_id,
        out_dir=str(out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir),
    )

    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.path.insert(0, str(REPO_ROOT))

    _step(report, "py_compile", lambda: py_compile.compile(str(draft_path), doraise=True))
    module = _step(report, "import_draft", lambda: _import_module(draft_path))
    if module is None:
        _write_json(out_dir / "adapter_draft_report.json", report.to_dict())
        return 1

    manifest_path = _resolve_manifest_path(args.manifest)
    proposal = None
    proposal_input = None
    if manifest_path is not None:
        proposal = _step(report, "load_proposal", lambda: _select_proposal(_load_json(manifest_path), args.trace_id))
        input_path = _proposal_input_path(manifest_path)
        if input_path is not None:
            proposal_input = _step(report, "load_proposal_input", lambda: _load_json(input_path))

    metadata = _step(report, "load_draft_metadata", lambda: getattr(module, "draft_metadata", lambda: {})()) or {}
    trace_id = args.trace_id or metadata.get("trace_id") or (proposal or {}).get("trace_id")
    op_type = getattr(module, "OP_TYPE", metadata.get("op_type") or (proposal or {}).get("proposed_op_type"))
    variant = getattr(module, "_VARIANT", metadata.get("variant") or (proposal or {}).get("proposed_variant") or "default")
    fi_api = getattr(module, "_FI_API", metadata.get("fi_api") or (proposal or {}).get("proposed_fi_api"))

    report.trace_id = trace_id
    _write_json(out_dir / "draft_metadata.json", metadata)
    if proposal is not None:
        _write_json(out_dir / "proposal.json", proposal)

    if not trace_id:
        report.add("resolve_trace_id", "fail", "missing trace_id; pass --trace-id or --manifest")
        _write_json(out_dir / "adapter_draft_report.json", report.to_dict())
        return 1

    def _classify() -> tuple[str, str] | None:
        return module.classify_trace_id(trace_id)

    classified = _step(report, "classify_trace_id", _classify)
    if classified != (op_type, variant):
        report.add(
            "classify_trace_id_expected",
            "fail",
            f"expected {(op_type, variant)!r}, got {classified!r}",
        )

    item = _proposal_item(proposal_input, trace_id)
    signatures = item.get("signatures") if isinstance(item.get("signatures"), list) else []
    count = int(item.get("count") or 1)
    matched = _synthetic_matched_bucket(
        op_type=op_type,
        variant=variant,
        trace_id=trace_id,
        fi_api=fi_api,
        signatures=signatures,
        count=max(count, 1),
    )
    _write_json(out_dir / "matched_kernel_input.json", matched)

    kernels = _step(report, "build_kernels", lambda: module.build_kernels(matched))
    if not kernels:
        report.add(
            "build_kernels_nonempty",
            "fail",
            "draft produced no kernel entries; implement parameter extraction before registering",
        )
        kernels = []
    if not isinstance(kernels, list):
        report.add("build_kernels_type", "fail", f"expected list, got {type(kernels).__name__}")
        kernels = []
    report.kernels = kernels
    _write_json(out_dir / "kernels.json", kernels)

    from artifact_schemas import validate_definition

    definitions_dir = out_dir / "definitions"
    for idx, kernel in enumerate(kernels):
        definition = _step(
            report,
            f"generate_definition[{idx}]",
            lambda kernel=kernel: module.generate_definition(kernel, args.model_tag, args.tp),
        )
        if not isinstance(definition, dict):
            report.add(
                f"generate_definition[{idx}]_type",
                "fail",
                f"expected dict, got {type(definition).__name__}",
            )
            continue
        _step(report, f"validate_definition[{idx}]", lambda definition=definition: validate_definition(definition))
        op = str(definition.get("op_type") or op_type)
        name = str(definition.get("name") or kernel.get("definition_name") or f"definition_{idx}")
        out_path = definitions_dir / op / f"{name}.json"
        _write_json(out_path, definition)
        report.definitions.append(str(out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path))

    if not report.definitions:
        report.add(
            "generated_definitions_nonempty",
            "fail",
            "no schema-valid definitions were generated",
        )

    report_path = out_dir / "adapter_draft_report.json"
    _write_json(report_path, report.to_dict())
    print(f"adapter draft smoke report: {report_path}")
    for step in report.steps:
        detail = f" - {step.detail}" if step.detail else ""
        print(f"{step.status.upper():4} {step.name}{detail}")
    return 1 if report.failed() else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
