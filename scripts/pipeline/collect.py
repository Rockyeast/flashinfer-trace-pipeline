"""Workload collection orchestration for the top-level pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from pipeline.configs import CollectConfig
from pipeline.definition_records import collect_definition_groups
from pipeline.paths import resolve_under_project


LogFn = Callable[[str], None]


def _build_collect_command(
    *,
    modal_bin: str,
    collect_script: str,
    scripts_dir: Path,
    config: CollectConfig,
    definition_names: list[str] | None,
    output_dir: Path | str | None,
    collect_enable_piecewise_cuda_graph: bool,
) -> list[str]:
    """Build one modal command for the collector."""
    cmd = [
        modal_bin,
        "run",
        str(scripts_dir / collect_script),
        "--model-path",
        config.model_path,
        "--tp",
        str(config.tp),
        "--definitions-dir",
        str(config.definitions_dir),
    ]

    if not definition_names:
        raise ValueError("definition_names must be non-empty")
    cmd += ["--definitions", ",".join(definition_names)]

    if output_dir:
        cmd += ["--output-dir", str(output_dir)]
    cmd += ["--prompt-source", config.prompt_source]
    if config.page_size > 0:
        cmd += ["--page-size", str(config.page_size)]
    cmd += ["--cuda-graph-max-bs", str(config.cuda_graph_max_bs)]
    if config.dtype:
        cmd += ["--dtype", config.dtype]
    if config.mem_fraction_static >= 0:
        cmd += ["--mem-fraction-static", str(config.mem_fraction_static)]
    if config.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if config.force_flashinfer_backends:
        cmd += ["--force-flashinfer-backends"]
    if collect_enable_piecewise_cuda_graph:
        cmd += ["--enable-piecewise-cuda-graph"]
    if config.debug_hooks:
        cmd += ["--debug-hooks"]
    if config.streaming:
        batch_sizes = config.batch_sizes or [1, 2, 4, 8, 16, 32, 64]
        cmd += ["--streaming-batch-sizes", ",".join(str(bs) for bs in batch_sizes)]
        cmd += ["--workloads-per-batch", str(config.workloads_per_batch)]
    else:
        cmd += ["--no-streaming-collect"]
    cmd += ["--max-dups-per-axes", str(config.max_dups_per_axes)]
    return cmd


def _merge_diagnostic_entry(base: dict, incoming: dict) -> dict:
    """Merge per-definition collector diagnostics from multiple coverage passes."""
    merged = dict(base)
    for key in ("candidates", "selected", "discarded", "written"):
        if isinstance(incoming.get(key), int):
            merged[key] = int(merged.get(key, 0) or 0) + int(incoming[key])

    reasons = dict(merged.get("discard_reasons") or {})
    for reason, count in (incoming.get("discard_reasons") or {}).items():
        reasons[reason] = int(reasons.get(reason, 0) or 0) + int(count)
    if reasons:
        merged["discard_reasons"] = reasons

    by_batch = list(merged.get("by_batch") or [])
    by_batch.extend(incoming.get("by_batch") or [])
    if by_batch:
        merged["by_batch"] = by_batch

    return merged


def _merge_collect_pass_outputs(
    pass_dirs: list[tuple[str, Path]],
    final_output_dir: Path,
    *,
    log: LogFn,
) -> None:
    """Merge workloads/blob/diagnostics from isolated collect pass dirs."""
    final_output_dir.mkdir(parents=True, exist_ok=True)
    merged_diag: dict = {
        "_coverage": "auto",
        "_pass_dirs": {label: str(path) for label, path in pass_dirs},
        "_passes": {},
    }
    replaced_workloads: set[Path] = set()

    for label, pass_dir in pass_dirs:
        workloads_dir = pass_dir / "workloads"
        if workloads_dir.exists():
            for src in sorted(workloads_dir.rglob("*.jsonl")):
                rel = src.relative_to(pass_dir)
                dst = final_output_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                mode = "a"
                if rel not in replaced_workloads:
                    mode = "w"
                    replaced_workloads.add(rel)
                with open(src, "r") as fin, open(dst, mode) as fout:
                    for line in fin:
                        fout.write(line)

        blob_dir = pass_dir / "blob"
        if blob_dir.exists():
            for src in sorted(blob_dir.rglob("*")):
                if not src.is_file():
                    continue
                rel = src.relative_to(pass_dir)
                dst = final_output_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)

        diag_path = pass_dir / "collector_diagnostics.json"
        if diag_path.exists():
            try:
                diag = json.loads(diag_path.read_text())
            except json.JSONDecodeError as exc:
                log(f"WARNING: failed to read diagnostics from {diag_path}: {exc}")
                continue
            merged_diag["_passes"][label] = diag
            for name, entry in diag.items():
                if name.startswith("_") or not isinstance(entry, dict):
                    continue
                if name in merged_diag and isinstance(merged_diag[name], dict):
                    merged_diag[name] = _merge_diagnostic_entry(merged_diag[name], entry)
                else:
                    merged_diag[name] = entry

    diag_out = final_output_dir / "collector_diagnostics.json"
    diag_out.write_text(json.dumps(merged_diag, indent=2, ensure_ascii=False))
    log(f"Merged collect diagnostics: {diag_out}")


def _run_collect_with_log(label: str, args: list[str], log_path: Path) -> int:
    """Run a collect subprocess while streaming output to both terminal and log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(args)}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"[collect {label}] {line}")
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return proc.wait()


def step_collect(
    config: CollectConfig,
    *,
    dry_run: bool,
    project_root: Path,
    scripts_dir: Path,
    log: LogFn,
    log_step: LogFn,
) -> int:
    """Run Modal workload collection for real FlashInfer definitions."""
    collect_script = "collect_workloads_modal.py"
    log_step("Step 4a: Collect Workloads")
    if config.tp <= 0:
        log("ERROR: invalid TP for collect. Pass --tp explicitly or fix TP auto-detection.")
        return 1
    final_output_dir = (
        resolve_under_project(Path(config.output_dir), project_root)
        if config.output_dir
        else project_root
    )
    logs_dir = final_output_dir / "logs"

    modal_bin = shutil.which("modal") or str(Path.home() / ".local" / "bin" / "modal")

    normal_defs, paged_prefill_defs = collect_definition_groups(config.definitions_dir, log=log)
    active_groups = [
        ("ragged", normal_defs, False),
        ("paged", paged_prefill_defs, True),
    ]
    active_groups = [(label, defs, piecewise) for label, defs, piecewise in active_groups if defs]
    if len(active_groups) > 1:
        pass_specs = [
            (label, defs, piecewise, final_output_dir / f".collect_pass_{label}")
            for label, defs, piecewise in active_groups
        ]
        merge_after_collect = True
    else:
        pass_specs = [
            (label, defs, piecewise, config.output_dir)
            for label, defs, piecewise in active_groups
        ]
        merge_after_collect = False

    commands: list[tuple[str, Path | str | None, list[str]]] = []
    for label, def_names, enable_piecewise, output_dir in pass_specs:
        if not def_names:
            log(f"Skipping {label} collect pass: no matching definitions")
            continue
        cmd = _build_collect_command(
            modal_bin=modal_bin,
            collect_script=collect_script,
            scripts_dir=scripts_dir,
            config=config,
            definition_names=def_names,
            output_dir=output_dir,
            collect_enable_piecewise_cuda_graph=enable_piecewise,
        )
        commands.append((label, output_dir, cmd))

    if not commands:
        log("No collectable definitions found.")
        return 0

    for label, _, cmd in commands:
        log(f"Running ({label}): {' '.join(cmd)}")

    if dry_run:
        log("(dry-run, skipping Modal collection)")
        return 0

    logs_dir.mkdir(parents=True, exist_ok=True)

    if merge_after_collect:
        for _, output_dir, _ in commands:
            if output_dir:
                pass_dir = Path(output_dir)
                if pass_dir.name.startswith(".collect_pass_") and pass_dir.exists():
                    shutil.rmtree(pass_dir)

    threads: list[tuple[str, Path, threading.Thread, dict[str, int]]] = []
    for label, _, cmd in commands:
        log_path = logs_dir / f"collect_{label}.log"
        log(f"Collect log ({label}): {log_path}")
        result: dict[str, int] = {}
        thread = threading.Thread(
            target=lambda label_value=label, command=cmd, path=log_path, r=result: r.update(
                rc=_run_collect_with_log(label_value, command, path)
            ),
            daemon=False,
        )
        thread.start()
        threads.append((label, log_path, thread, result))

    return_codes: dict[str, int] = {}
    for label, log_path, thread, result in threads:
        thread.join()
        return_codes[label] = result.get("rc", 1)
        if return_codes[label] != 0:
            log(f"ERROR: collect pass {label} failed with rc={return_codes[label]} (log: {log_path})")

    failures = {label: rc for label, rc in return_codes.items() if rc != 0}
    if failures:
        log(f"ERROR: collect pass failures: {failures}")
        return next(iter(failures.values()))

    if merge_after_collect:
        merge_dirs = [
            (label, Path(output_dir))
            for label, output_dir, _ in commands
            if output_dir is not None
        ]
        _merge_collect_pass_outputs(merge_dirs, final_output_dir, log=log)

    return 0
