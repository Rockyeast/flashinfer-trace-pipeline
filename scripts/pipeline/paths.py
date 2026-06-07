"""Output path planning for the top-level pipeline entrypoint."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .mode import arg_was_provided


@dataclass(frozen=True)
class PipelinePaths:
    definitions_dir: Path
    collect_output_dir: str
    solutions_dir: Path
    probe_output_dir: Path | None
    tests_output_dir: Path | None
    inventory_output_path: Path | None
    run_output_root: Path | None


def _slugify_path_component(value: str) -> str:
    """Convert a model name like Qwen/Qwen3-0.6B into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "model"


def resolve_under_project(path: Path, project_root: Path) -> Path:
    """Resolve relative paths from the repo root instead of the caller's cwd."""
    return path if path.is_absolute() else project_root / path


_MODEL_BOUND_STEPS = {
    "all",
    "static",
    "probe",
    "parse",
    "definitions",
    "collect",
}


def _run_label(args, argv: list[str]) -> str:
    """Return the label used for isolated run output directories."""
    if not arg_was_provided("--model-name", argv) and args.model_tag:
        return args.model_tag
    if args.step not in _MODEL_BOUND_STEPS and not arg_was_provided("--model-name", argv):
        return args.step
    return args.model_name


def _make_run_output_root(base_dir: Path, label: str, project_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return resolve_under_project(base_dir, project_root) / f"{_slugify_path_component(label)}_{ts}"


def resolve_pipeline_paths(
    args,
    argv: list[str],
    *,
    project_root: Path,
) -> PipelinePaths:
    """Normalize all output paths before the step runner starts."""
    definitions_dir = args.definitions_dir
    collect_output_dir = args.collect_output_dir
    solutions_dir = args.solutions_dir
    probe_output_dir = args.probe_output_dir
    tests_output_dir = args.tests_output_dir
    inventory_output_path: Path | None = None
    run_output_root: Path | None = None

    if args.step != "validate":
        reuse_collect_output = args.step == "collect" and arg_was_provided("--collect-output-dir", argv)
        if reuse_collect_output:
            run_output_root = resolve_under_project(Path(collect_output_dir), project_root)
        else:
            run_output_root = _make_run_output_root(
                args.output_root,
                _run_label(args, argv),
                project_root,
            )
        if not arg_was_provided("--definitions-dir", argv):
            definitions_dir = run_output_root / "definitions"
        if not arg_was_provided("--collect-output-dir", argv):
            collect_output_dir = str(run_output_root)
        if not arg_was_provided("--solutions-dir", argv):
            solutions_dir = run_output_root / "solutions"
        if not arg_was_provided("--probe-output-dir", argv):
            probe_output_dir = run_output_root / "probe"
        if not arg_was_provided("--tests-output-dir", argv):
            tests_output_dir = run_output_root / "tests" / "references"
        inventory_output_path = run_output_root / "kernel_inventory.json"
        run_output_root.mkdir(parents=True, exist_ok=True)

    return PipelinePaths(
        definitions_dir=definitions_dir,
        collect_output_dir=collect_output_dir,
        solutions_dir=solutions_dir,
        probe_output_dir=probe_output_dir,
        tests_output_dir=tests_output_dir,
        inventory_output_path=inventory_output_path,
        run_output_root=run_output_root,
    )


def log_run_paths(paths: PipelinePaths, log) -> None:
    """Print the normalized output paths that the pipeline will use."""
    if paths.run_output_root is None:
        return
    log(f"Run output root: {paths.run_output_root}")
    log(f"   definitions: {paths.definitions_dir}")
    log(f"   workloads/blob: {paths.collect_output_dir}")
    log(f"   solutions: {paths.solutions_dir}")
    log(f"   probe: {paths.probe_output_dir}")
    log(f"   tests: {paths.tests_output_dir}")
    log(f"   inventory: {paths.inventory_output_path}")
