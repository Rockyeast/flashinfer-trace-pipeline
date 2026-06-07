"""Definition, baseline-solution, and reference-test generation steps."""

from __future__ import annotations

from collections.abc import Callable

from pipeline.configs import BaselineSolutionsConfig, DefinitionConfig, TestConfig


RunScript = Callable[[str, list[str], bool], int]
Log = Callable[[str], None]


def step_definitions(
    config: DefinitionConfig,
    *,
    run_script: RunScript,
    log_step: Log,
) -> int:
    """Generate definition JSON files from kernel_inventory.json."""
    log_step("Step 3: Generate Definitions")

    args = [
        str(config.inventory_path),
        "--model-tag", config.model_tag,
        "--tp", str(config.tp),
        "--definitions-dir", str(config.definitions_dir),
    ]
    if config.fi_trace_manifest is not None:
        args.extend(["--fi-trace-manifest", str(config.fi_trace_manifest)])
    if config.dry_run:
        args.append("--dry-run")

    return run_script("generate_definitions.py", args, dry_run=config.dry_run)


def step_baseline_solutions(
    config: BaselineSolutionsConfig,
    *,
    run_script: RunScript,
    log_step: Log,
) -> int:
    """Generate baseline solution JSON files from definitions."""
    log_step("Step 4b: Generate Baseline Solutions")

    args = [
        "--definitions-dir", str(config.definitions_dir),
        "--solutions-dir", str(config.solutions_dir),
    ]
    if config.dry_run:
        args.append("--dry-run")

    return run_script("generate_baseline_solutions.py", args, dry_run=config.dry_run)


def step_tests(
    config: TestConfig,
    *,
    run_script: RunScript,
    log: Log,
    log_step: Log,
) -> int:
    """Generate reference pytest files from definition JSON files."""
    log_step("Step 5: Generate Reference Tests")
    definitions_dir = config.definitions_dir
    if not any(definitions_dir.rglob("*.json")):
        log(f"⚠️  No definition JSONs found under {definitions_dir}; skipping tests")
        return 0

    args = ["--definitions-dir", str(definitions_dir)]
    if config.output_dir:
        args.extend(["--output-dir", str(config.output_dir)])
    if config.dry_run:
        args.append("--dry-run")

    return run_script("generate_tests.py", args, dry_run=config.dry_run)
