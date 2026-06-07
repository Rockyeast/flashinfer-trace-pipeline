"""Typed config objects passed between pipeline orchestration steps."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.paths import resolve_under_project


def _copy_prefixed_args(
    args: Namespace,
    *,
    prefix: str,
    field_names: set[str],
    values: dict[str, Any],
) -> None:
    """Copy argparse values named like ``<prefix>_<field>`` into config kwargs."""
    for name in field_names:
        if name in values:
            continue
        arg_name = f"{prefix}_{name}"
        if hasattr(args, arg_name):
            values[name] = getattr(args, arg_name)


def _field_names(config_cls: type) -> set[str]:
    """Return public dataclass field names for config auto-mapping."""
    return {field.name for field in fields(config_cls)}


@dataclass(frozen=True)
class ProbeConfig:
    """Runtime probe options shared by the pipeline entrypoint and probe step."""

    model_name: str
    tp: int
    prompt: str
    output_dir: Path | None
    coverage: str
    prefill_path: str
    page_sizes: list[int] | None
    mem_fraction_static: float
    force_flashinfer_backends: bool
    resume_function_call_id: str = ""
    detach: bool = True
    dry_run: bool = False

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        *,
        tp: int,
        output_dir: Path | None,
    ) -> "ProbeConfig":
        """Build probe config from parsed CLI args plus runtime-derived values."""
        values: dict[str, Any] = {
            "model_name": args.model_name,
            "tp": tp,
            "output_dir": output_dir,
            "force_flashinfer_backends": args.force_flashinfer_backends,
            "detach": not args.no_probe_detach,
            "dry_run": args.dry_run,
        }
        _copy_prefixed_args(args, prefix="probe", field_names=_field_names(cls), values=values)
        return cls(**values)


@dataclass(frozen=True)
class CollectConfig:
    """Workload collection options shared by pipeline collection helpers."""

    model_path: str
    tp: int
    definitions_dir: Path
    output_dir: Path | str | None = ""
    prompt_source: str = "sharegpt"
    debug_hooks: bool = False
    page_size: int = 0
    cuda_graph_max_bs: int = -1
    dtype: str = ""
    mem_fraction_static: float = -1.0
    trust_remote_code: bool = False
    force_flashinfer_backends: bool = False
    streaming: bool = True
    batch_sizes: list[int] | None = None
    workloads_per_batch: int = 4
    max_dups_per_axes: int = 2

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        *,
        tp: int,
        definitions_dir: Path,
        output_dir: Path | str | None,
    ) -> "CollectConfig":
        """Build collect config from parsed CLI args plus runtime-derived values."""
        values: dict[str, Any] = {
            "model_path": args.model_name,
            "tp": tp,
            "definitions_dir": definitions_dir,
            "output_dir": output_dir,
            "force_flashinfer_backends": args.force_flashinfer_backends,
        }
        _copy_prefixed_args(args, prefix="collect", field_names=_field_names(cls), values=values)
        return cls(**values)


@dataclass(frozen=True)
class ParseConfig:
    """Runtime parse options shared by the pipeline entrypoint and parse step."""

    probe_output: Path
    model_name: str
    definitions_dir: Path
    hf_config: Path | None
    output_path: Path | None
    llm_classify: bool = False
    llm_model: str = "claude-sonnet-4-6"
    llm_base_url: str | None = None
    dry_run: bool = False

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        *,
        probe_output: Path,
        definitions_dir: Path,
        output_path: Path | None,
    ) -> "ParseConfig":
        """Build parse config from parsed CLI args plus runtime-derived values."""
        return cls(
            probe_output=probe_output,
            model_name=args.model_name,
            definitions_dir=definitions_dir,
            hf_config=args.hf_config,
            output_path=output_path,
            llm_classify=args.llm_classify,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            dry_run=args.dry_run,
        )


@dataclass(frozen=True)
class StaticCandidatesConfig:
    """Static inventory generation options."""

    model_name: str
    tp: int
    definitions_dir: Path
    hf_config: Path | None
    sglang_root: Path
    output_path: Path | None
    page_sizes: list[int] | None
    dry_run: bool = False

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        *,
        tp: int,
        definitions_dir: Path,
        output_path: Path | None,
    ) -> "StaticCandidatesConfig":
        """Build static-candidate config from parsed CLI args plus runtime-derived values."""
        return cls(
            model_name=args.model_name,
            tp=tp,
            definitions_dir=definitions_dir,
            hf_config=args.hf_config,
            sglang_root=args.sglang_root,
            output_path=output_path,
            page_sizes=args.probe_page_sizes,
            dry_run=args.dry_run,
        )


@dataclass(frozen=True)
class DefinitionConfig:
    """Definition generation options."""

    inventory_path: Path
    model_tag: str
    tp: int
    definitions_dir: Path
    fi_trace_manifest: Path | None = None
    dry_run: bool = False

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        *,
        inventory_path: Path,
        model_tag: str,
        tp: int,
        definitions_dir: Path,
        fi_trace_manifest: Path | None,
    ) -> "DefinitionConfig":
        """Build definition config from parsed CLI args plus runtime-derived values."""
        return cls(
            inventory_path=inventory_path,
            model_tag=model_tag,
            tp=tp,
            definitions_dir=definitions_dir,
            fi_trace_manifest=fi_trace_manifest,
            dry_run=args.dry_run,
        )


@dataclass(frozen=True)
class TestConfig:
    """Reference test generation options."""

    definitions_dir: Path
    output_dir: Path | None
    dry_run: bool = False

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        *,
        definitions_dir: Path,
        output_dir: Path | None,
    ) -> "TestConfig":
        """Build reference-test config from parsed CLI args plus runtime-derived values."""
        return cls(
            definitions_dir=definitions_dir,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )


@dataclass(frozen=True)
class BaselineSolutionsConfig:
    """Baseline solution generation options."""

    definitions_dir: Path
    solutions_dir: Path
    dry_run: bool = False

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        *,
        definitions_dir: Path,
        solutions_dir: Path,
    ) -> "BaselineSolutionsConfig":
        """Build baseline-solution config from parsed CLI args plus paths."""
        return cls(
            definitions_dir=definitions_dir,
            solutions_dir=solutions_dir,
            dry_run=args.dry_run,
        )


@dataclass(frozen=True)
class ValidationConfig:
    """Dataset validation and generated-artifact check options."""

    dataset_root: Path
    checks: str
    outputs: str
    output_folder: Path | None
    strict: bool
    disable_gpu: bool
    generated_artifacts: bool
    generated_artifact_root: Path
    dry_run: bool = False

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        *,
        run_output_root: Path | None,
        collect_output_dir: str,
        project_root: Path,
    ) -> "ValidationConfig":
        """Build validation config from parsed CLI args plus run paths."""
        if args.official_validate_dataset:
            dataset_root = resolve_under_project(args.official_validate_dataset, project_root)
        elif run_output_root:
            dataset_root = run_output_root
        elif collect_output_dir:
            dataset_root = resolve_under_project(Path(collect_output_dir), project_root)
        else:
            dataset_root = project_root

        if run_output_root is not None:
            generated_artifact_root = run_output_root / "generated_artifact_check"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            generated_artifact_root = project_root / "tmp" / "run" / f"generated_artifact_check_{ts}"

        return cls(
            dataset_root=dataset_root,
            checks=args.official_validate_checks,
            outputs=args.official_validate_outputs,
            output_folder=args.official_validate_output_folder,
            strict=args.official_validate_strict,
            disable_gpu=args.official_validate_disable_gpu,
            generated_artifacts=args.validate_generated_artifacts,
            generated_artifact_root=generated_artifact_root,
            dry_run=args.dry_run,
        )
