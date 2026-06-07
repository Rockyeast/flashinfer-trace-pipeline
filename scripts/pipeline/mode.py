"""Pipeline profile and step-mode resolution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


PIPELINE_PROFILES: dict[str, dict[str, object]] = {
    "fast": {
        "probe_coverage": "fast",
        "probe_prefill_path": "both",
        "probe_page_sizes": [64],
    },
    "full": {
        "probe_coverage": "full",
        "probe_prefill_path": "both",
        "probe_page_sizes": [64],
    },
    "smoke": {
        "step": "validate",
        "official_validate_checks": "layout,definition,workload",
        "official_validate_strict": True,
        "official_validate_disable_gpu": True,
    },
}

PROFILE_OPTION_BY_ATTR = {
    "probe_coverage": "--probe-coverage",
    "probe_page_sizes": "--probe-page-sizes",
    "skip_official_validate": "--skip-official-validate",
    "official_validate_checks": "--official-validate-checks",
    "official_validate_strict": "--official-validate-strict",
    "official_validate_disable_gpu": "--official-validate-disable-gpu",
}

PROFILE_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("--fast", "fast", "Use fast probe scenarios."),
    ("--full", "full", "Use full probe scenarios."),
    ("--smoke", "smoke", "Run strict dataset validation."),
)

PROFILE_FLAG_DESTS = {
    flag.lstrip("-").replace("-", "_"): (flag, profile)
    for flag, profile, _ in PROFILE_FLAGS
}

STEP_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("--static", "static", "Run static candidate generation."),
    ("--probe", "probe", "Run the Modal probe step."),
    ("--parse", "parse", "Parse probe output."),
    ("--definitions", "definitions", "Generate definitions."),
    ("--collect", "collect", "Collect workloads."),
    ("--solutions", "solutions", "Generate baseline solutions."),
    ("--tests", "tests", "Generate reference tests."),
    ("--validate", "validate", "Run dataset validation."),
)

STEP_FLAG_DESTS = {
    flag.lstrip("-").replace("-", "_"): (flag, step)
    for flag, step, _ in STEP_FLAGS
}


@dataclass(frozen=True)
class PipelineMode:
    step: str
    runtime_probe_requested: bool
    static_sidecar_requested: bool

    @property
    def run_all(self) -> bool:
        return self.step == "all"

    def runs_step(self, step: str) -> bool:
        return self.run_all or self.step == step

    @property
    def runs_runtime_parse(self) -> bool:
        return (self.run_all and self.runtime_probe_requested) or self.step == "parse"


def arg_was_provided(option: str, argv: list[str]) -> bool:
    """Return True for both '--flag value' and '--flag=value' spellings."""
    return option in argv or any(arg.startswith(f"{option}=") for arg in argv)


def resolve_cli_mode_flags(
    args: argparse.Namespace,
    argv: list[str],
    parser: argparse.ArgumentParser,
) -> None:
    """Resolve mutually-exclusive profile and step flags into args.profile/args.step."""
    selected_profile_flags = [
        (flag, profile)
        for dest, (flag, profile) in PROFILE_FLAG_DESTS.items()
        if getattr(args, dest)
    ]
    if len(selected_profile_flags) > 1:
        flags = ", ".join(flag for flag, _profile in selected_profile_flags)
        parser.error(f"profile flags are mutually exclusive: {flags}")
    if selected_profile_flags:
        _flag, profile = selected_profile_flags[0]
        args.profile = profile

    selected_step_flags = [
        (flag, step)
        for dest, (flag, step) in STEP_FLAG_DESTS.items()
        if getattr(args, dest)
    ]
    if len(selected_step_flags) > 1:
        flags = ", ".join(flag for flag, _step in selected_step_flags)
        parser.error(f"step flags are mutually exclusive: {flags}")
    if selected_step_flags:
        step_flag, step = selected_step_flags[0]
        profile_step = PIPELINE_PROFILES.get(args.profile or "", {}).get("step")
        if profile_step and profile_step != step:
            parser.error(f"{step_flag} cannot be combined with --{args.profile}")
        args.step = step
        if step == "collect" and not arg_was_provided("--skip-official-validate", argv):
            args.skip_official_validate = True


def resolve_pipeline_mode(args: argparse.Namespace, probe_output) -> PipelineMode:
    """Resolve the effective pipeline mode after profile/step flags are applied."""
    runtime_step_requested = args.step in ("all", "probe", "parse")
    return PipelineMode(
        step=args.step,
        runtime_probe_requested=runtime_step_requested or probe_output is not None,
        static_sidecar_requested=args.with_static and runtime_step_requested,
    )


def requires_model_runtime_plan(mode: PipelineMode) -> bool:
    """Return whether this invocation needs HF config / TP / Modal GPU planning."""
    return (
        mode.runs_step("static")
        or mode.runs_step("probe")
        or mode.runs_step("definitions")
        or mode.runs_step("collect")
    )


def apply_pipeline_profile(args: argparse.Namespace, argv: list[str]) -> None:
    """Apply profile defaults without overriding explicit CLI arguments."""
    if not args.profile:
        return

    for attr, value in PIPELINE_PROFILES[args.profile].items():
        if attr == "step":
            if args.step == "all":
                setattr(args, attr, value)
            continue
        if attr == "probe_prefill_path":
            if not (
                arg_was_provided("--paged", argv)
                or arg_was_provided("--both", argv)
            ):
                setattr(args, attr, value)
            continue
        option = PROFILE_OPTION_BY_ATTR[attr]
        if option and not arg_was_provided(option, argv):
            setattr(args, attr, value)
