"""Official flashinfer-bench validation step."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from pipeline.configs import ValidationConfig
from pipeline.paths import resolve_under_project
from pipeline.subprocess_logging import command_label, run_streamed


Log = Callable[[str], None]


def _logs_dir(config: ValidationConfig) -> Path:
    return config.dataset_root / "logs"


def step_schema_audit(
    config: ValidationConfig,
    *,
    scripts_dir: Path,
    log: Log,
    log_step: Log,
) -> int:
    """Run this repo's strict schema audit for dataset JSON artifacts."""
    log_step("Step 6a: Schema Audit")
    args = [
        sys.executable,
        str(scripts_dir / "audit_schemas.py"),
        "--root",
        str(config.dataset_root),
    ]
    log(f"Running: {' '.join(args)}")
    if config.dry_run:
        log("(dry-run, skipping)")
        return 0
    return run_streamed(
        args,
        log_path=_logs_dir(config) / "schema_audit.log",
        prefix="schema",
    )


def step_official_validate(
    config: ValidationConfig,
    *,
    project_root: Path,
    log: Log,
    log_step: Log,
) -> int:
    """Run upstream flashinfer-bench dataset validation for generated artifacts."""
    log_step("Step 6b: Official Dataset Validation")

    env = os.environ.copy()
    # Keep validator/import side effects out of ~/.cache and inside this repo's tmp/.
    # flashinfer_bench imports FlashInfer modules before the subprocess is launched,
    # so the current process environment must be updated before the import below.
    cache_root = project_root / "tmp" / "official_validate_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env["XDG_CACHE_HOME"] = str(cache_root)
    env["FLASHINFER_WORKSPACE_BASE"] = str(cache_root)
    os.environ.update(
        {
            "XDG_CACHE_HOME": env["XDG_CACHE_HOME"],
            "FLASHINFER_WORKSPACE_BASE": env["FLASHINFER_WORKSPACE_BASE"],
        }
    )

    try:
        import flashinfer_bench  # noqa: F401
    except ImportError:
        log("❌ flashinfer-bench is not installed. Run: pip install flashinfer-bench")
        return 1

    args = [
        sys.executable,
        "-c",
        "from flashinfer_bench.cli.main import cli; cli()",
        "validate",
        "--dataset", str(config.dataset_root),
        "--checks", config.checks,
        "--outputs", config.outputs,
    ]
    if config.disable_gpu:
        args.append("--disable-gpu")
    if config.output_folder:
        args.extend(["--output-folder", str(resolve_under_project(config.output_folder, project_root))])

    log(f"Running: {' '.join(args)}")
    gpu_mode = "GPU disabled" if config.disable_gpu else "GPU enabled"
    log(f"Official validation checks: {config.checks} ({gpu_mode})")
    if config.dry_run:
        log("(dry-run, skipping)")
        return 0

    return run_streamed(
        args,
        env=env,
        log_path=_logs_dir(config) / "official_validate.log",
        prefix="official-validate",
    )


def step_generated_artifacts_check(
    config: ValidationConfig,
    *,
    scripts_dir: Path,
    log: Log,
    log_step: Log,
) -> int:
    """Generate and syntax-check non-GPU artifacts from dataset definitions."""
    log_step("Step 6c: Generated Artifact Check")
    definitions_dir = config.dataset_root / "definitions"
    if not definitions_dir.exists() or not any(definitions_dir.rglob("*.json")):
        log(f"❌ no definition JSONs found under {definitions_dir}")
        return 1

    solutions_dir = config.generated_artifact_root / "solutions"
    tests_dir = config.generated_artifact_root / "tests" / "references"
    commands = [
        [
            sys.executable,
            str(scripts_dir / "generate_baseline_solutions.py"),
            "--definitions-dir",
            str(definitions_dir),
            "--solutions-dir",
            str(solutions_dir),
        ],
        [
            sys.executable,
            str(scripts_dir / "generate_tests.py"),
            "--definitions-dir",
            str(definitions_dir),
            "--output-dir",
            str(tests_dir),
        ],
        [
            sys.executable,
            str(scripts_dir / "audit_schemas.py"),
            "--root",
            str(config.generated_artifact_root),
        ],
    ]

    for cmd in commands:
        log(f"Running: {' '.join(cmd)}")
        if config.dry_run:
            log("(dry-run, skipping)")
            continue
        rc = run_streamed(
            cmd,
            log_path=_logs_dir(config) / f"generated_artifact_check_{command_label(cmd)}.log",
            prefix="artifact-check",
        )
        if rc != 0:
            return rc

    if config.dry_run:
        log(
            "Would compile generated tests under "
            f"{tests_dir} after generation completes."
        )
        return 0

    test_files = sorted(tests_dir.glob("*.py"))
    if not test_files:
        log(f"❌ no generated test files found under {tests_dir}")
        return 1

    cmd = [sys.executable, "-m", "py_compile", *[str(path) for path in test_files]]
    log(f"Running: {' '.join(cmd)}")
    return run_streamed(
        cmd,
        log_path=_logs_dir(config) / "generated_tests_py_compile.log",
        prefix="artifact-check",
    )
