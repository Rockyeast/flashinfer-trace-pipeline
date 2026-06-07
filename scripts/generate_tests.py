#!/usr/bin/env python3
"""Generate reference test files from definition JSONs.

This is Step 5 of the v2 pipeline. For each definition, it generates a
pytest-compatible test file at tests/references/test_{def_name}.py.

The generated test contains:
  - The reference implementation from definition.reference
  - A generate_random_inputs() helper from definition.inputs
  - A test_correctness() body backed by the adapter baseline solution
  - A main() function for standalone execution

Definitions use adapter-backed baseline solutions in scripts/adapters/.
Definitions without an adapter-backed baseline solution fail explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from test_generators import generate_test_file as generate_modular_test_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference test files from definitions",
    )
    parser.add_argument(
        "--definitions",
        nargs="+",
        help=(
            "Optional definition-name filter. Defaults to every JSON under "
            "--definitions-dir."
        ),
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=Path("definitions"),
        help="Root definitions directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/references"),
        help="Output directory for test files",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing test files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated",
    )
    args = parser.parse_args()

    requested_definitions = set(args.definitions or [])
    if not args.definitions_dir.exists():
        print(f"ERROR: {args.definitions_dir} not found", file=sys.stderr)
        sys.exit(1)

    definition_paths = sorted(args.definitions_dir.rglob("*.json"))
    if requested_definitions:
        definition_paths = [
            path for path in definition_paths if path.stem in requested_definitions
        ]
        missing = requested_definitions - {path.stem for path in definition_paths}
        for name in sorted(missing):
            print(f"  ⚠️  {name}: definition not found, skipping")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    errors = 0

    for def_path in definition_paths:
        def_name = def_path.stem
        with open(def_path) as f:
            definition = json.load(f)

        test_path = args.output_dir / f"test_{def_name}.py"

        if test_path.exists() and not args.replace:
            if args.dry_run:
                print(f"  ⏭️  {def_name}: test already exists")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  🆕 {def_name} → {test_path}")
            generated += 1
            continue

        try:
            content = generate_modular_test_file(definition)
            test_path.write_text(content, encoding="utf-8")
            generated += 1
            print(f"  ✅ {def_name} → {test_path}")
        except Exception as e:
            print(f"  ❌ {def_name}: {e}", file=sys.stderr)
            errors += 1

    print(f"\n{'='*60}")
    print("Generate Tests Summary:")
    print(f"  ✅ Generated: {generated}")
    print(f"  ⏭️  Skipped:   {skipped}")
    print(f"  ❌ Errors:    {errors}")
    print(f"{'='*60}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
