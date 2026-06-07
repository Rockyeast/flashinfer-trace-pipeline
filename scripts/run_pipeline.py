#!/usr/bin/env python3
"""Run the FlashInfer trace pipeline.

This is the top-level entrypoint for static candidates, runtime probing,
definition generation, workload collection, reference-test generation, and
dataset validation. For the stable user-facing guide, see docs/README_ZH.md
or docs/README.md.

Usage:
    # Recommended daily run: fast probe scenarios, default/ragged prefill path
    python scripts/run_pipeline.py --fast --model-name Qwen/Qwen3.5-35B-A3B

    # Final wider run: full probe scenarios plus both default and paged prefill paths
    python scripts/run_pipeline.py --full --both --model-name Qwen/Qwen3.5-35B-A3B

    # Reuse an existing runtime probe output
    python scripts/run_pipeline.py \
        --fast \
        --probe-output tmp/run/Qwen_Qwen3.5-35B-A3B_*/probe/aggregated_summary.json \
        --model-name Qwen/Qwen3.5-35B-A3B

    # Common profile flags
    python scripts/run_pipeline.py --smoke
    python scripts/run_pipeline.py --collect --definitions-dir ...
    python scripts/run_pipeline.py --validate --official-validate-dataset ...

    # Dry run
    python scripts/run_pipeline.py --model-name ... --dry-run
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from pipeline.cli import build_parser
from pipeline.cleanup import FailedRunCleanup
from pipeline.collect import step_collect
from pipeline.configs import (
    BaselineSolutionsConfig,
    CollectConfig,
    DefinitionConfig,
    ParseConfig,
    ProbeConfig,
    StaticCandidatesConfig,
    TestConfig,
    ValidationConfig,
)
from pipeline.definition_index import write_definition_index
from pipeline.definition_records import is_repo_global_definitions_dir
from pipeline.discovery_steps import step_parse, step_probe, step_static_candidates
from pipeline.fi_trace import stage_runtime_fi_trace_if_needed
from pipeline.generation_steps import step_baseline_solutions, step_definitions, step_tests
from pipeline.mode import (
    apply_pipeline_profile,
    resolve_cli_mode_flags,
    resolve_pipeline_mode,
)
from pipeline.paths import log_run_paths, resolve_pipeline_paths
from pipeline.runtime_plan import prepare_runtime_plan
from pipeline.static_inventory import merge_static_candidates_into_inventory
from pipeline.subprocess_logging import command_label, prepend_pythonpath, run_streamed
from pipeline.summary import log_pipeline_summary
from pipeline.validation import (
    step_generated_artifacts_check,
    step_official_validate,
    step_schema_audit,
)


SCRIPTS_DIR = Path(__file__).parent          # scripts/ 目录
PROJECT_ROOT = SCRIPTS_DIR.parent            # flashinfer-trace 仓库根目录
SCRIPT1_PATH = SCRIPTS_DIR / "probe" / "scheduler.py"  # Modal probe 调度脚本路径
PIPELINE_LOG_DIR: Path | None = None
PIPELINE_LOG_PATH: Path | None = None


def _set_pipeline_log_dir(run_output_root: Path | None) -> None:
    """Enable run-local pipeline logs when this command owns a run directory."""
    global PIPELINE_LOG_DIR, PIPELINE_LOG_PATH
    if run_output_root is None:
        PIPELINE_LOG_DIR = None
        PIPELINE_LOG_PATH = None
        return
    PIPELINE_LOG_DIR = run_output_root / "logs"
    PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_LOG_PATH = PIPELINE_LOG_DIR / "pipeline.log"
    PIPELINE_LOG_PATH.write_text("", encoding="utf-8")


def _log(msg: str) -> None:
    """打印带时间戳的日志，格式：[pipeline HH:MM:SS] msg"""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[pipeline {ts}] {msg}"
    print(line, flush=True)
    if PIPELINE_LOG_PATH is not None:
        with PIPELINE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _log_step(name: str) -> None:
    """打印步骤标题（带上下分隔线），让日志更易读"""
    _log("=" * 60)
    _log(name)
    _log("=" * 60)


def _run_script(script: str, args: list[str], dry_run: bool = False) -> int:
    """统一调用 pipeline 子脚本的工具函数。

    所有子步骤（parse、definitions、collect、tests）都通过这个函数调用。
    dry_run=True 时只打印命令、不实际执行，返回 0（用于调试/预览）。
    返回子进程的 exit code（0=成功，非0=失败）。
    """
    # 构造完整命令行列表：
    #   sys.executable → 当前 Python 解释器路径（如 /usr/bin/python3）
    #   SCRIPTS_DIR / script → 子脚本的完整路径（如 /...flashinfer-trace/scripts/parse/parse_probe.py）
    #   + args → 传给子脚本的额外参数（如 ["--model-name", "Qwen/Qwen2.5-7B"]）
    # 例如最终 cmd = ["/usr/bin/python3", "/...scripts/parse/parse_probe.py", "--model-name", "Qwen/Qwen2.5-7B"]
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + args
    _log(f"Running: {' '.join(cmd)}")   # 打印将要执行的完整命令，方便调试

    if dry_run:
        # dry-run 模式：只打印命令不执行，直接返回 0（视为成功）
        _log("(dry-run, skipping)")
        return 0

    env = os.environ.copy()
    env = prepend_pythonpath(env, SCRIPTS_DIR)
    log_path = None
    if PIPELINE_LOG_DIR is not None:
        log_path = PIPELINE_LOG_DIR / f"{command_label(cmd)}.log"

    # 用 run_streamed 启动子进程，同时把 stdout/stderr 写入 run-local log。
    # 子进程完成后，返回值是它的退出码：
    #   0  → 正常退出（脚本运行成功）
    #   非0 → 出错（脚本内部报错、assert 失败等）
    return run_streamed(cmd, env=env, log_path=log_path)


def main():
    parser = build_parser()
    args = parser.parse_args()
    resolve_cli_mode_flags(args, sys.argv[1:], parser)
    apply_pipeline_profile(args, sys.argv[1:])

    paths = resolve_pipeline_paths(args, sys.argv[1:], project_root=PROJECT_ROOT)
    _set_pipeline_log_dir(paths.run_output_root)
    cleanup = FailedRunCleanup(_log)
    cleanup.register(paths.run_output_root)
    definitions_dir = paths.definitions_dir
    collect_output_dir = paths.collect_output_dir
    solutions_dir = paths.solutions_dir
    probe_output_dir = paths.probe_output_dir
    tests_output_dir = paths.tests_output_dir
    inventory_output_path = paths.inventory_output_path
    run_output_root = paths.run_output_root

    log_run_paths(paths, _log)

    def refresh_definition_index() -> None:
        write_definition_index(
            run_output_root,
            definitions_dir,
            tests_output_dir,
            project_root=PROJECT_ROOT,
            solutions_dir=solutions_dir,
            log=_log,
        )

    probe_output = args.probe_output     # 用户指定的已有 probe 输出路径（跳过 Step 1）
    inventory_path = args.inventory       # 用户指定的已有 inventory 路径（跳过 Step 1+2）
    user_provided_inventory = inventory_path is not None
    mode = resolve_pipeline_mode(args, probe_output)

    runtime_plan = prepare_runtime_plan(
        args=args,
        mode=mode,
        probe_output=probe_output,
        inventory_path=inventory_path,
        project_root=PROJECT_ROOT,
        log=_log,
    )
    tp = runtime_plan.tp
    validation_config = ValidationConfig.from_args(
        args,
        run_output_root=run_output_root,
        collect_output_dir=collect_output_dir,
        project_root=PROJECT_ROOT,
    )

    # Derive model_tag from model_name if not set
    model_tag = args.model_tag
    if not model_tag:
        model_tag = args.model_name.split("/")[-1].lower()

    # Step 1a: Static candidates（显式 --static 时使用，不跑 inference）
    if (
        mode.runs_step("static")
        and not inventory_path
        and not mode.runtime_probe_requested
    ):
        static_config = StaticCandidatesConfig.from_args(
            args,
            tp=tp,
            definitions_dir=definitions_dir,
            output_path=inventory_output_path,
        )
        inventory_path = step_static_candidates(
            static_config,
            project_root=PROJECT_ROOT,
            run_script=_run_script,
            log=_log,
            log_step=_log_step,
        )
        if inventory_path is None and not args.dry_run:
            sys.exit(1)

    # Step 1b: Probe（默认 runtime probe，在 Modal GPU 上采集 kernel 调用）
    # 跳过条件：--skip-probe，或用户已提供 --probe-output / --inventory
    if mode.runs_step("probe") and mode.runtime_probe_requested and not args.skip_probe:
        # 只有在没有任何现有输出的情况下才实际跑 probe
        if not probe_output and not inventory_path:
            probe_config = ProbeConfig.from_args(
                args,
                tp=tp,
                output_dir=probe_output_dir,
            )
            probe_output = step_probe(
                probe_config,
                project_root=PROJECT_ROOT,
                scheduler_path=SCRIPT1_PATH,
                log=_log,
                log_step=_log_step,
            )
            if probe_output is None and not args.dry_run:
                # probe 返回 None 表示子进程失败，后续步骤无法继续
                _log("❌ Probe step failed, cannot continue")
                sys.exit(1)
        elif probe_output:
            # 用户传了 --probe-output，直接用现有结果，跳过 probe
            _log(f"⏭️  Skipping probe (using provided --probe-output: {probe_output})")
        elif inventory_path:
            # 用户传了 --inventory，连 parse 也不用跑，直接跳过 probe
            _log(f"⏭️  Skipping probe (using provided --inventory: {inventory_path})")

    if mode.runtime_probe_requested and not probe_output and not inventory_path:
        if not args.dry_run:
            _log("❌ No probe output available. Run --probe or provide --probe-output/--inventory.")
            sys.exit(1)

    # Stage official fi_trace before parse so inventory can see those definitions.
    try:
        fi_trace_manifest_path = stage_runtime_fi_trace_if_needed(
            runtime_requested=mode.runtime_probe_requested,
            probe_output=probe_output,
            definitions_dir=definitions_dir,
            dry_run=args.dry_run,
            project_root=PROJECT_ROOT,
            log=_log,
            log_step=_log_step,
        )
    except RuntimeError as exc:
        _log(f"❌ {exc}")
        if not args.dry_run:
            sys.exit(1)
        fi_trace_manifest_path = (
            probe_output.parent / "fi_trace_staged_definitions.json"
            if probe_output is not None
            else None
        )

    # Step 2: Runtime parse（解析 probe 输出 → kernel_inventory.json）
    # 如果用户直接提供了 --inventory，跳过这步
    if mode.runs_runtime_parse:
        if probe_output and not inventory_path:
            parse_config = ParseConfig.from_args(
                args,
                probe_output=probe_output,
                definitions_dir=definitions_dir,
                output_path=inventory_output_path,
            )
            inventory_path = step_parse(
                parse_config,
                run_script=_run_script,
                log=_log,
                log_step=_log_step,
            )
            if inventory_path is None and not args.dry_run:
                sys.exit(1)
        elif inventory_path:
            _log(f"⏭️  Skipping parse (using provided --inventory: {inventory_path})")

    if (
        mode.static_sidecar_requested
        and inventory_path
        and not user_provided_inventory
        and not args.dry_run
    ):
        static_inventory_path = Path(inventory_path).with_name("static_kernel_inventory.json")
        static_config = StaticCandidatesConfig.from_args(
            args,
            tp=tp,
            definitions_dir=definitions_dir,
            output_path=static_inventory_path,
        )
        sidecar_path = step_static_candidates(
            static_config,
            project_root=PROJECT_ROOT,
            run_script=_run_script,
            log=_log,
            log_step=_log_step,
        )
        if sidecar_path is not None:
            merge_static_candidates_into_inventory(Path(inventory_path), sidecar_path, log=_log)
        else:
            _log("⚠️  Static sidecar candidate generation failed; continuing with runtime inventory only")

    inventory_required = mode.runs_step("definitions")

    if inventory_required and not inventory_path and args.dry_run:
        inventory_path = Path("(dry-run-placeholder)")

    if inventory_required and not inventory_path:
        print("ERROR: No inventory path available. Run parse step first or use --inventory.")
        sys.exit(1)

    # Step 3: Generate Definitions（根据 inventory 生成 definition JSON）
    if mode.runs_step("definitions"):
        definition_config = DefinitionConfig.from_args(
            args,
            inventory_path=inventory_path,
            model_tag=model_tag,
            tp=tp,
            definitions_dir=definitions_dir,
            fi_trace_manifest=fi_trace_manifest_path,
        )
        rc = step_definitions(
            definition_config,
            run_script=_run_script,
            log_step=_log_step,
        )
        if rc != 0 and not args.dry_run:
            _log("❌ Step 3 failed")
            sys.exit(1)
        refresh_definition_index()

    # Step 4a: Collect（Modal GPU 上收集真实 workload，可用 --skip-collect 跳过）
    if mode.runs_step("collect") and not args.skip_collect:
        if is_repo_global_definitions_dir(definitions_dir, project_root=PROJECT_ROOT) and not args.allow_global_definitions:
            _log(
                "❌ Refusing to collect from repo-root definitions/. "
                "Use an isolated tmp/run/.../definitions directory, or pass "
                "--allow-global-definitions if you intentionally want all repo definitions."
            )
            sys.exit(1)
        collect_config = CollectConfig.from_args(
            args,
            tp=tp,
            definitions_dir=definitions_dir,
            output_dir=collect_output_dir,
        )
        rc = step_collect(
            collect_config,
            dry_run=args.dry_run,
            project_root=PROJECT_ROOT,
            scripts_dir=SCRIPTS_DIR,
            log=_log,
            log_step=_log_step,
        )
        if rc != 0 and not args.dry_run:
            _log("❌ Step 4a (collect) failed")
            sys.exit(rc)
        refresh_definition_index()

    # Step 4b: Baseline Solutions（adapter wrapper 优先，reference fallback）
    if mode.runs_step("solutions") and not args.skip_baseline_solutions:
        baseline_config = BaselineSolutionsConfig.from_args(
            args,
            definitions_dir=definitions_dir,
            solutions_dir=solutions_dir,
        )
        rc = step_baseline_solutions(
            baseline_config,
            run_script=_run_script,
            log_step=_log_step,
        )
        if rc != 0 and not args.dry_run:
            _log("❌ Step 4b (baseline solutions) failed")
            sys.exit(rc)
        refresh_definition_index()

    # Step 5: Generate Tests（生成 reference test 文件，已有的跳过）
    if mode.runs_step("tests"):
        test_config = TestConfig.from_args(
            args,
            definitions_dir=definitions_dir,
            output_dir=tests_output_dir,
        )
        rc = step_tests(
            test_config,
            run_script=_run_script,
            log=_log,
            log_step=_log_step,
        )
        if rc != 0 and not args.dry_run:
            _log("❌ Step 5 failed")
            sys.exit(1)
        refresh_definition_index()

    if mode.runs_step("validate"):
        rc = step_schema_audit(
            validation_config,
            scripts_dir=SCRIPTS_DIR,
            log=_log,
            log_step=_log_step,
        )
        if rc != 0:
            _log(f"❌ schema audit failed with exit code {rc}")
            if not args.dry_run:
                sys.exit(rc)
        if not args.skip_official_validate:
            rc = step_official_validate(
                validation_config,
                project_root=PROJECT_ROOT,
                log=_log,
                log_step=_log_step,
            )
            if rc != 0:
                msg = f"official validation failed with exit code {rc}"
                if validation_config.strict and not args.dry_run:
                    _log(f"❌ {msg}")
                    sys.exit(rc)
                _log(f"⚠️  {msg}; continuing because --official-validate-strict is not set")
        if validation_config.generated_artifacts:
            rc = step_generated_artifacts_check(
                validation_config,
                scripts_dir=SCRIPTS_DIR,
                log=_log,
                log_step=_log_step,
            )
            if rc != 0:
                _log(f"❌ generated artifact check failed with exit code {rc}")
                if not args.dry_run:
                    sys.exit(rc)
        refresh_definition_index()

    should_return = log_pipeline_summary(
        inventory_path=inventory_path,
        run_output_root=run_output_root,
        definitions_dir=definitions_dir,
        validate_step=mode.runs_step("validate"),
        official_validate_strict=validation_config.strict,
        log=_log,
    )
    cleanup.prune_uninformative_run_root()
    cleanup.mark_completed()
    if should_return:
        return


if __name__ == "__main__":
    main()
