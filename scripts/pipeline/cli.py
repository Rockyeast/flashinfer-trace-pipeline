"""CLI parser for the top-level FlashInfer trace pipeline."""

from __future__ import annotations

import argparse
import sys

from .cli_sections import (
    add_collect_options,
    add_dataset_io_options,
    add_generation_options,
    add_mode_flags,
    add_probe_options,
    add_validation_options,
)


_ADVANCED_HELP = (
    "--probe-output-dir",
    "--probe-resume-function-call-id",
    "--no-probe-detach",
    "--model-tag",
    "--hf-config",
    "--allow-global-definitions",
    "--skip-probe",
    "--skip-collect",
    "--solutions-dir",
    "--skip-baseline-solutions",
    "--probe-prompt",
    "--probe-coverage",
    "--probe-page-sizes",
    "--force-flashinfer-backends",
    "--collect-prompt-source",
    "--collect-debug-hooks",
    "--collect-page-size",
    "--collect-cuda-graph-max-bs",
    "--collect-dtype",
    "--collect-mem-fraction-static",
    "--collect-trust-remote-code",
    "--collect-streaming",
    "--no-collect-streaming",
    "--collect-batch-sizes",
    "--collect-workloads-per-batch",
    "--collect-max-dups-per-axes",
    "--collect-output-dir",
    "--tests-output-dir",
    "--skip-official-validate",
    "--official-validate-dataset",
    "--official-validate-bench-dir",
    "--official-validate-checks",
    "--official-validate-outputs",
    "--official-validate-output-folder",
    "--official-validate-strict",
    "--official-validate-disable-gpu",
    "--validate-generated-artifacts",
    "--llm-classify",
    "--llm-model",
    "--llm-base-url",
)


class _PipelineArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that hides advanced options from normal --help."""

    def format_help(self) -> str:
        show_all = "--help-all" in sys.argv
        hidden = set() if show_all else set(_ADVANCED_HELP)
        original_help: list[tuple[argparse.Action, str | None]] = []
        if hidden:
            for action in self._actions:
                if any(opt in hidden for opt in action.option_strings):
                    original_help.append((action, action.help))
                    action.help = argparse.SUPPRESS
        try:
            help_text = super().format_help()
        finally:
            for action, help_value in original_help:
                action.help = help_value
        if "--help-all" not in sys.argv:
            help_text += (
                "\nAdvanced/debug options are hidden. "
                "Use --help-all to show every flag.\n"
            )
        return help_text


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level pipeline CLI parser."""
    parser = _PipelineArgumentParser(
        description="Run the FlashInfer trace pipeline",
    )
    parser.add_argument(
        "--help-all",
        action="help",
        help="Show all options, including advanced/debug flags.",
    )
    add_mode_flags(parser)
    add_dataset_io_options(parser)
    add_probe_options(parser)
    add_collect_options(parser)
    add_generation_options(parser)
    add_validation_options(parser)
    return parser
