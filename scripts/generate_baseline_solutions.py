#!/usr/bin/env python3
"""Generate baseline solution JSON files.

This stage first asks the op adapter for a FlashInfer wrapper baseline. If the
adapter has no op-specific solution generator, it falls back to a conservative
correctness baseline by copying definition.reference into main.py.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

from adapters import (
    eval_validation_policy,
    generate_baseline_solution as generate_adapter_baseline_solution,
)
from artifact_schemas import validate_definition, validate_solution


def _load_definition(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: skipping invalid definition {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"WARNING: skipping non-object definition {path}", file=sys.stderr)
        return None
    try:
        validate_definition(data)
    except Exception as exc:
        print(f"WARNING: skipping invalid definition {path}: {exc}", file=sys.stderr)
        return None
    return data


def _reference_has_run(reference: str) -> bool:
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
        for node in tree.body
    )


def _definition_paths(definitions_dir: Path, requested: set[str]) -> list[Path]:
    paths = sorted(definitions_dir.rglob("*.json")) if definitions_dir.exists() else []
    if not requested:
        return paths
    return [path for path in paths if path.stem in requested]


def _solution_payload(definition: dict, reference: str, solution_name: str) -> dict:
    def_name = definition["name"]
    return {
        "name": solution_name,
        "definition": def_name,
        "author": "baseline",
        "spec": {
            "language": "python",
            "target_hardware": ["NVIDIA_H100", "NVIDIA_A100", "CPU"],
            "entry_point": "main.py::run",
            "dependencies": [],
            "validation_policy": eval_validation_policy(definition),
            "destination_passing_style": False,
            "binding": None,
        },
        "sources": [
            {
                "path": "main.py",
                "content": reference,
            }
        ],
        "description": (
            "Baseline solution copied from definition.reference. "
            "This is a correctness-first PyTorch reference baseline."
        ),
    }


def _reference_solution_payload(definition: dict, reference: str) -> dict:
    digest = hashlib.sha256(reference.encode("utf-8")).hexdigest()[:8]
    return _solution_payload(definition, reference, f"torch_reference_{digest}")


def _solution_source_hash(payload: dict) -> str:
    sources = payload.get("sources", [])
    content = ""
    if isinstance(sources, list):
        content = "\n".join(
            source.get("content", "")
            for source in sources
            if isinstance(source, dict) and source.get("path") == "main.py"
        )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]


def _build_solution_payload(definition: dict, reference: str) -> tuple[dict, str]:
    adapter_payload = generate_adapter_baseline_solution(definition)
    if adapter_payload is not None:
        adapter_payload.setdefault("spec", {}).setdefault(
            "validation_policy",
            eval_validation_policy(definition),
        )
        return adapter_payload, "adapter"
    return _reference_solution_payload(definition, reference), "reference"


def generate_baseline_solutions(
    *,
    definitions_dir: Path,
    solutions_dir: Path,
    requested: set[str],
    replace: bool,
    dry_run: bool,
) -> int:
    """Generate missing baseline solution JSON files.

    Existing solution directories are skipped by default so manually written or
    previously generated baselines are not overwritten.
    """
    written = 0
    skipped_existing = 0
    skipped_invalid = 0
    skipped_requested_missing = set(requested)

    for path in _definition_paths(definitions_dir, requested):
        definition = _load_definition(path)
        if definition is None:
            skipped_invalid += 1
            continue

        def_name = definition.get("name")
        op_type = definition.get("op_type")
        reference = definition.get("reference")
        if isinstance(def_name, str):
            skipped_requested_missing.discard(def_name)
        if not isinstance(def_name, str) or not def_name:
            print(f"WARNING: skipping {path}: missing string name", file=sys.stderr)
            skipped_invalid += 1
            continue
        if not isinstance(op_type, str) or not op_type:
            print(f"WARNING: skipping {path}: missing string op_type", file=sys.stderr)
            skipped_invalid += 1
            continue
        if not isinstance(reference, str) or not reference.strip():
            print(f"WARNING: skipping {def_name}: missing definition.reference", file=sys.stderr)
            skipped_invalid += 1
            continue
        if not _reference_has_run(reference):
            print(f"WARNING: skipping {def_name}: reference has no run() function", file=sys.stderr)
            skipped_invalid += 1
            continue

        payload, source_kind = _build_solution_payload(definition, reference)
        solution_name = payload.get("name")
        if not isinstance(solution_name, str) or not solution_name:
            solution_name = f"{source_kind}_baseline_{_solution_source_hash(payload)}"
            payload["name"] = solution_name

        sol_dir = solutions_dir / "baseline" / op_type / def_name
        existing = sorted(sol_dir.glob("*.json")) if sol_dir.exists() else []
        if existing and not replace:
            skipped_existing += 1
            print(f"skip existing baseline solution: {sol_dir}")
            continue

        payload.setdefault("definition", def_name)
        validate_solution(payload)
        out_path = sol_dir / f"{solution_name}.json"

        if dry_run:
            print(f"(dry-run) would write {source_kind} baseline {out_path}")
        else:
            sol_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {source_kind} baseline {out_path}")
        written += 1

    if skipped_requested_missing:
        for name in sorted(skipped_requested_missing):
            print(f"WARNING: requested definition not found: {name}", file=sys.stderr)
        skipped_invalid += len(skipped_requested_missing)

    print(
        "baseline solutions: "
        f"written={written} skipped_existing={skipped_existing} "
        f"skipped_invalid={skipped_invalid}"
    )
    return 1 if skipped_invalid and requested else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate baseline solution JSONs from adapters or definition.reference."
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=Path("definitions"),
        help="Root directory containing definition JSON files.",
    )
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=Path("solutions"),
        help="Root solutions directory; writes under solutions/baseline/...",
    )
    parser.add_argument(
        "--definitions",
        nargs="*",
        default=[],
        help="Optional definition names to generate. Defaults to all definitions.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Write the generated baseline JSON even if the "
            "solution directory already contains JSON files. Existing unrelated "
            "solution files are not deleted."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    return generate_baseline_solutions(
        definitions_dir=args.definitions_dir,
        solutions_dir=args.solutions_dir,
        requested=set(args.definitions),
        replace=args.replace,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
