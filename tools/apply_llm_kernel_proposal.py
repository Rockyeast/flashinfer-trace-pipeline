#!/usr/bin/env python3
"""Write a reviewed LLM kernel proposal as an adapter draft.

This tool is intentionally conservative. It never edits parse rules and never
creates a live adapter module directly. Instead, it writes
``scripts/adapters/_draft_<op_type>.py`` so the draft is visible for review but
skipped by adapter auto-registration.

Usage:
  python tools/apply_llm_kernel_proposal.py \
    tmp/run/<run>/llm_diagnostics/parse_rules \
    --trace-id flashinfer.foo.bar \
    --promote \
    --apply
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS_DIR = REPO_ROOT / "scripts" / "adapters"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path(path: Path) -> Path:
    if path.is_dir():
        for name in ("apply_manifest.json", "parse_rule_proposals.json"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise ValueError(f"cannot find apply_manifest.json or parse_rule_proposals.json in {path}")
    return path


def _snake(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("empty snake_case value")
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        raise ValueError(f"not a safe snake_case identifier: {value!r}")
    return value


def _py(value: Any) -> str:
    return repr(value)


def _select_proposal(
    manifest: dict[str, Any],
    trace_id: str | None,
    *,
    promote: bool,
) -> dict[str, Any]:
    proposals = manifest.get("proposals") or []
    if promote:
        if not trace_id:
            raise ValueError("--promote requires --trace-id")
        matches = [p for p in proposals if p.get("trace_id") == trace_id]
        if matches:
            matches = [dict(matches[0], review_decision="promote")]
    else:
        matches = [p for p in proposals if p.get("review_decision") == "promote"]
        if trace_id:
            matches = [p for p in matches if p.get("trace_id") == trace_id]

    if not matches:
        raise ValueError("no promoted proposal matched; pass --promote or set review_decision=promote first")
    if len(matches) > 1:
        names = ", ".join(str(p.get("trace_id")) for p in matches[:5])
        raise ValueError(f"multiple promoted proposals matched ({names}); pass --trace-id")
    return matches[0]


def _adapter_draft_template(proposal: dict[str, Any], op_type: str, variant: str) -> str:
    trace_id = str(proposal.get("trace_id") or "")
    fi_api = proposal.get("proposed_fi_api")
    target_api = proposal.get("proposed_target_api")
    definition_name_template = proposal.get("definition_name_template")
    params_to_extract = proposal.get("params_to_extract") or {}
    collector_notes = proposal.get("collector_notes")
    test_generator_notes = proposal.get("test_generator_notes")
    risks = proposal.get("risks") or []
    reasoning = proposal.get("reasoning") or ""
    proposal_payload = dict(proposal)

    return dedent(f'''\
        """Draft adapter for {op_type}.

        Generated from an LLM proposal for human review. This file is named
        ``_draft_{op_type}.py`` so scripts/adapters/__init__.py will skip it.
        Rename it to ``{op_type}.py`` only after implementing and testing the
        TODOs below.
        """

        from __future__ import annotations

        from parse.inventory_helpers import observation_fields

        OP_TYPE = {op_type!r}
        _TRACE_ID = {trace_id!r}
        _VARIANT = {variant!r}
        _FI_API = {fi_api!r}
        _TARGET_API = {target_api!r}
        _DEFINITION_NAME_TEMPLATE = {definition_name_template!r}
        _PARAMS_TO_EXTRACT = {_py(params_to_extract)}
        _COLLECTOR_NOTES = {collector_notes!r}
        _TEST_GENERATOR_NOTES = {test_generator_notes!r}
        _RISKS = {_py(risks)}
        _LLM_REASONING = {reasoning!r}
        _LLM_PROPOSAL = {_py(proposal_payload)}


        def draft_metadata() -> dict:
            """Return the reviewed LLM proposal that produced this draft."""
            return {{
                "op_type": OP_TYPE,
                "variant": _VARIANT,
                "trace_id": _TRACE_ID,
                "fi_api": _FI_API,
                "target_api": _TARGET_API,
                "definition_name_template": _DEFINITION_NAME_TEMPLATE,
                "params_to_extract": _PARAMS_TO_EXTRACT,
                "collector_notes": _COLLECTOR_NOTES,
                "test_generator_notes": _TEST_GENERATOR_NOTES,
                "risks": _RISKS,
                "reasoning": _LLM_REASONING,
                "proposal": _LLM_PROPOSAL,
            }}


        def classify_trace_id(trace_id: str) -> tuple[str, str] | None:
            """Return (op_type, variant) after this draft is reviewed.

            TODO:
            - confirm this is a real benchmark kernel, not helper/noise
            - confirm all valid trace_id variants, not just the observed sample
            - then enable the exact or regex match below
            """
            if trace_id == _TRACE_ID:
                return (OP_TYPE, _VARIANT)
            return None


        def definition_name(variant: str, params: dict) -> str:
            """Return the canonical definition name for this adapter.

            TODO: replace with reviewed naming logic.
            """
            _ = (variant, params)
            raise NotImplementedError("review and implement definition_name")


        def build_kernels(matched_kernels: dict[str, dict]) -> list[dict]:
            """Build kernel inventory entries from matched trace buckets.

            TODO:
            - extract canonical params from signatures
            - set fi_api only when observed trace_id starts with flashinfer.
            - include observation_fields(info, observed_kwargs, observed_param_names)

            Until reviewed, this returns [] so the draft cannot affect inventory.
            """
            _ = (matched_kernels, observation_fields)
            return []


        def generate_definition(kernel: dict, model_tag: str, tp: int) -> dict:
            """Render the official definition JSON.

            TODO: implement via adapters.official_templates.render_definition
            when possible, otherwise write the reference/axes/inputs/outputs
            explicitly and validate with scripts/artifact_schemas.py.
            """
            _ = (kernel, model_tag, tp)
            raise NotImplementedError("review and implement generate_definition")


        # Optional baseline-solution skeleton. It is intentionally commented out
        # so the draft cannot advertise solution support before human review.
        #
        # Review checklist:
        # - confirm the FlashInfer call signature from the official API/source
        # - map definition input names to the wrapper function arguments
        # - preserve destination-passing style when required by the official API
        # - validate the returned payload with schemas.validate_solution()
        #
        # LLM collector notes:
        #   {_py(collector_notes)}
        #
        # def generate_baseline_solution(definition: dict) -> dict | None:
        #     from adapters._solution_utils import input_names, solution_payload
        #
        #     inputs = input_names(definition)
        #     source = (
        #         "import flashinfer\\n\\n"
        #         f"def run({{', '.join(inputs)}}):\\n"
        #         "    # TODO: call {fi_api} with reviewed argument mapping.\\n"
        #         '    raise NotImplementedError("review and implement baseline solution")\\n'
        #     )
        #     return solution_payload(
        #         definition,
        #         name="flashinfer_{op_type}_baseline",
        #         description="TODO: reviewed FlashInfer baseline for {op_type}.",
        #         source=source,
        #     )


        # Optional reference-test generator skeleton. Keep this as guidance until
        # the new op_type has reviewed test support under scripts/test_generators/.
        #
        # LLM test-generator notes:
        #   {_py(test_generator_notes)}
        #
        # Suggested follow-up:
        # - add or extend scripts/test_generators/<module>.py
        # - load definition.reference as ground truth
        # - compare the generated baseline solution against reference outputs
        # - run tools/run_reference_tests_modal.py for GPU-backed correctness
        ''')


def _write_adapter_draft(
    proposal: dict[str, Any],
    *,
    apply: bool,
    force: bool,
) -> Path:
    trace_id = str(proposal.get("trace_id") or "")
    op_type = _snake(str(proposal.get("proposed_op_type") or ""))
    variant = _snake(str(proposal.get("proposed_variant") or "default"))
    fi_api = proposal.get("proposed_fi_api")
    target_api = proposal.get("proposed_target_api")

    if not trace_id:
        raise ValueError("proposal missing trace_id")
    if not isinstance(fi_api, str) or not fi_api.startswith("flashinfer."):
        if target_api:
            raise ValueError(
                "proposal is target_api-only; adapter drafts for non-flashinfer APIs are not dataset-ready"
            )
        raise ValueError("proposal has no flashinfer.* fi_api")

    path = ADAPTERS_DIR / f"_draft_{op_type}.py"
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")

    content = _adapter_draft_template(proposal, op_type, variant)
    if apply:
        path.write_text(content, encoding="utf-8")
    return path


def _run_py_compile(path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        cwd=REPO_ROOT,
        check=True,
    )


def _run_draft_validation(
    path: Path,
    *,
    manifest_path: Path,
    trace_id: str | None,
) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "validate_adapter_draft.py"),
        str(path),
        "--manifest",
        str(manifest_path),
    ]
    if trace_id:
        cmd.extend(["--trace-id", trace_id])
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Write one reviewed LLM proposal as an adapter draft.")
    parser.add_argument("manifest", type=Path, help="proposal directory, apply_manifest.json, or parse_rule_proposals.json")
    parser.add_argument("--trace-id", default=None, help="Trace ID to apply when multiple proposals are promoted")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Treat the selected --trace-id as reviewed/promoted without editing apply_manifest.json",
    )
    parser.add_argument("--apply", action="store_true", help="Actually write the draft; default is dry-run")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing _draft_<op_type>.py")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After writing the draft, run tools/validate_adapter_draft.py smoke checks",
    )
    args = parser.parse_args()

    manifest_path = _resolve_manifest_path(args.manifest)
    manifest = _load_json(manifest_path)
    proposal = _select_proposal(manifest, args.trace_id, promote=args.promote)

    path = _write_adapter_draft(proposal, apply=args.apply, force=args.force)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: adapter draft for {proposal.get('trace_id')}", flush=True)
    print(
        f"  - {'write' if args.apply else 'would write'} {path.relative_to(REPO_ROOT)}",
        flush=True,
    )

    if args.apply:
        _run_py_compile(path)
        print("py_compile: ok", flush=True)
        if args.validate:
            return _run_draft_validation(path, manifest_path=manifest_path, trace_id=args.trace_id)
    else:
        print("No files written. Re-run with --apply to create the draft.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
