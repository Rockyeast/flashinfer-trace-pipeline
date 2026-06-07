"""Argument groups for the top-level pipeline CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .mode import PROFILE_FLAGS, STEP_FLAGS


def add_mode_flags(parser: argparse.ArgumentParser) -> None:
    """Add high-level profile and step-selection flags."""
    parser.set_defaults(profile="fast", step="all")
    for flag, _profile, help_text in PROFILE_FLAGS:
        parser.add_argument(flag, action="store_true", help=help_text)
    for flag, _step, help_text in STEP_FLAGS:
        parser.add_argument(flag, action="store_true", help=help_text)


def add_dataset_io_options(parser: argparse.ArgumentParser) -> None:
    """Add model, artifact, and dataset path options."""
    parser.add_argument(
        "--probe-output",
        type=Path,
        help="Path to existing aggregated_summary.json (skip probe step)",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help="Path to existing kernel_inventory.json (skip probe + parse steps)",
    )
    parser.add_argument(
        "--with-static",
        action="store_true",
        default=True,
        help=(
            "After runtime probe/parse, also run static candidates and attach "
            "static-only entries to the inventory (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-static",
        dest="with_static",
        action="store_false",
        help="Skip static candidate generation and inventory attachment.",
    )
    parser.add_argument(
        "--sglang-root",
        type=Path,
        default=Path("sglang"),
        help="Local SGLang checkout used by static candidates (default: ./sglang).",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.5-35B-A3B",
        help="Model name (HuggingFace repo ID)",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Model tag for definitions (auto-detected if not set)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=0,
        help=(
            "Tensor parallelism degree (0 = auto-detect from model name). "
            "Use --tp 1 when you need official global-head definition names."
        ),
    )
    parser.add_argument(
        "--hf-config",
        type=Path,
        default=None,
        help="Path to HuggingFace config.json for dimension resolution",
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=Path("definitions"),
        help="Root definitions directory",
    )
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=Path("solutions"),
        help="Root solutions directory; baseline solutions are written under solutions/baseline/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("tmp/run"),
        help=(
            "Base directory for isolated pipeline run outputs "
            "(default: tmp/run)."
        ),
    )
    parser.add_argument(
        "--allow-global-definitions",
        action="store_true",
        help=(
            "Allow collect to scan the repo-root definitions/ directory. "
            "By default collect rejects this to avoid collecting historical definitions."
        ),
    )


def add_probe_options(parser: argparse.ArgumentParser) -> None:
    """Add runtime probe options."""
    parser.add_argument(
        "--probe-output-dir",
        type=Path,
        default=None,
        help="Write a newly generated probe aggregated_summary.json to this directory.",
    )
    parser.add_argument(
        "--probe-resume-function-call-id",
        default="",
        help=(
            "Reattach to an existing Modal probe FunctionCall instead of launching "
            "a new probe job."
        ),
    )
    parser.add_argument(
        "--no-probe-detach",
        action="store_true",
        help=(
            "Run Modal probe without `modal run --detach`. By default probe uses "
            "--detach so a local client disconnect does not stop the remote app."
        ),
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip the probe step (requires --probe-output or --inventory)",
    )
    parser.add_argument(
        "--probe-prompt",
        default="__SHAREGPT__",
        help="Prompt passed to probe; __SHAREGPT__ loads scripts/fixtures/sharegpt_100.json (default).",
    )
    parser.add_argument(
        "--probe-coverage",
        choices=["fast", "full"],
        default="fast",
        help=(
            "Runtime probe scenario profile. fast is cheap coverage; full runs broader scenarios."
        ),
    )
    parser.add_argument(
        "--probe-page-sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Explicit paged-prefill probe page_size values. --fast/--full "
            "default to page_size=64 for paged coverage. Used only with "
            "--paged/--both; pass values like `1 64` to force extra ps1/ps64 "
            "coverage."
        ),
    )
    parser.add_argument(
        "--probe-mem-fraction-static",
        type=float,
        default=0.7,
        help=(
            "SGLang Engine mem_fraction_static used by probe. Increase this for "
            "large-model probes when SGLang reports insufficient static memory."
        ),
    )
    prefill_group = parser.add_mutually_exclusive_group()
    prefill_group.add_argument(
        "--paged",
        dest="probe_prefill_path",
        action="store_const",
        const="paged",
        default="default",
        help="Run probe through the paged-prefill path only.",
    )
    prefill_group.add_argument(
        "--both",
        dest="probe_prefill_path",
        action="store_const",
        const="both",
        help=(
            "Run probe through both default/ragged and paged-prefill paths. "
            "This is the --fast/--full default."
        ),
    )
    parser.add_argument(
        "--force-flashinfer-backends",
        action="store_true",
        help=(
            "Force supported SGLang attention/sampling backend knobs to flashinfer "
            "during probe and collect."
        ),
    )


def add_collect_options(parser: argparse.ArgumentParser) -> None:
    """Add workload collection options."""
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip the Modal collection step (4a)",
    )
    parser.add_argument(
        "--collect-prompt-source",
        choices=["sharegpt", "synthetic"],
        default="sharegpt",
        help="Prompt source for workload collection (default: sharegpt).",
    )
    parser.add_argument(
        "--collect-debug-hooks",
        action="store_true",
        help=(
            "Enable collector hook debug logging. This runs only two "
            "inference rounds and is intended for debugging missing captures."
        ),
    )
    parser.add_argument(
        "--collect-page-size",
        type=int,
        default=0,
        help=(
            "FlashInfer collect page_size. Default 0 omits page_size from "
            "SGLang Engine kwargs and lets SGLang choose its runtime default."
        ),
    )
    parser.add_argument(
        "--collect-cuda-graph-max-bs",
        type=int,
        default=-1,
        help=(
            "FlashInfer collect cuda_graph_max_bs. Use -1 to omit it from "
            "SGLang Engine kwargs for probe-compatible A/B runs."
        ),
    )
    parser.add_argument(
        "--collect-dtype",
        default="",
        help="Optional dtype passed to collect SGLang Engine, e.g. bfloat16.",
    )
    parser.add_argument(
        "--collect-mem-fraction-static",
        type=float,
        default=-1.0,
        help=(
            "Optional mem_fraction_static passed to collect SGLang Engine. "
            "Use a negative value to omit it."
        ),
    )
    parser.add_argument(
        "--collect-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the collect SGLang Engine.",
    )
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--collect-streaming",
        dest="collect_streaming",
        action="store_true",
        default=True,
        help=(
            "Run collection in per-batch streaming mode: one SGLang "
            "run per outer batch size, sanitize that pass, then append workloads "
            "(default)."
        ),
    )
    stream_group.add_argument(
        "--no-collect-streaming",
        dest="collect_streaming",
        action="store_false",
        help="Run single-pass collection instead of streaming by batch size.",
    )
    parser.add_argument(
        "--collect-batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Outer inference batch sizes for streaming collection (default: 1 2 4 8 16 32 64).",
    )
    parser.add_argument(
        "--collect-workloads-per-batch",
        type=int,
        default=4,
        help="Maximum new workloads per definition appended after each streaming batch-size pass (default: 4).",
    )
    parser.add_argument(
        "--collect-max-dups-per-axes",
        type=int,
        default=2,
        help="Maximum duplicate candidates with identical axes kept within one sanitize pass (default: 2).",
    )
    parser.add_argument(
        "--collect-output-dir",
        default="",
        help="Write collect output (workloads/ and blob/) to this directory instead of the repo default. Useful for testing without polluting workloads/.",
    )


def add_generation_options(parser: argparse.ArgumentParser) -> None:
    """Add definition, solution, test, and LLM generation options."""
    parser.add_argument(
        "--skip-baseline-solutions",
        action="store_true",
        help="Skip generation of baseline solution JSON files (Step 4b).",
    )
    parser.add_argument(
        "--tests-output-dir",
        type=Path,
        default=None,
        help="Write generated reference tests to this directory instead of tests/references.",
    )
    parser.add_argument(
        "--llm-classify",
        action="store_true",
        help=(
            "For unknown trace_ids in step 2 (parse): call LLM for diagnostic "
            "classification suggestions. Suggestions are cached and not promoted "
            "automatically. Requires ANTHROPIC_API_KEY."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default="claude-sonnet-4-6",
        help="Anthropic model ID for LLM classification diagnostics (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Proxy base URL for Anthropic-compatible API (e.g. https://api.aipaibox.com).",
    )


def add_validation_options(parser: argparse.ArgumentParser) -> None:
    """Add validation and smoke-check options."""
    parser.add_argument(
        "--skip-official-validate",
        action="store_true",
        help=(
            "Skip upstream flashinfer-bench validation. By default the pipeline "
            "runs dataset-only checks: layout,definition,workload."
        ),
    )
    parser.add_argument(
        "--official-validate-dataset",
        type=Path,
        default=None,
        help=(
            "Dataset root to pass to flashinfer-bench validate. "
            "Defaults to run output root, then --collect-output-dir, then repo root."
        ),
    )
    parser.add_argument(
        "--official-validate-checks",
        default="layout,definition,workload",
        help=(
            "Comma-separated official validation checks. Keep the default for this "
            "pipeline stage; solution/trace/baseline require later artifacts."
        ),
    )
    parser.add_argument(
        "--official-validate-outputs",
        default="stdout,json,text",
        help="Comma-separated official validation outputs: stdout,json,text.",
    )
    parser.add_argument(
        "--official-validate-output-folder",
        type=Path,
        default=None,
        help="Where official validation reports are written. Defaults to <dataset>/reports/.",
    )
    parser.add_argument(
        "--official-validate-strict",
        action="store_true",
        help="Exit nonzero if official validation fails. By default, warn and continue.",
    )
    parser.add_argument(
        "--official-validate-disable-gpu",
        action="store_true",
        help=(
            "Pass --disable-gpu to flashinfer-bench validate. Use this only for "
            "CPU-only structural checks; GPU validation is the default."
        ),
    )
    parser.add_argument(
        "--validate-generated-artifacts",
        action="store_true",
        help=(
            "During validate/smoke, generate baseline solutions and reference tests "
            "from dataset definitions into tmp/run, then compile generated tests."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
