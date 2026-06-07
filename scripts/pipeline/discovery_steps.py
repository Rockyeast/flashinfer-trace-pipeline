"""Probe, parse, and static-candidate steps for the top-level pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from .configs import ParseConfig, ProbeConfig, StaticCandidatesConfig
from .paths import resolve_under_project
from .runtime_plan import auto_download_hf_config
from .subprocess_logging import popen_streamed, run_streamed, wait_streamed


RunScript = Callable[[str, list[str], bool], int]
Log = Callable[[str], None]


def _probe_logs_dir(output_dir: Path) -> Path:
    """Place probe logs under the run root when output_dir is run/probe."""
    if output_dir.name == "probe":
        return output_dir.parent / "logs"
    return output_dir / "logs"


def _safe_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name).strip("_")


def _default_probe_output_dir(model_name: str, project_root: Path) -> Path:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "tmp" / "fi_probe" / f"{_safe_model_name(model_name)}_{ts}" / "probe"


def _log_probe_resume_hint(
    output_dir: Path,
    log: Log,
    *,
    probe_prefill_path: str = "default",
    probe_page_size: int | None = None,
) -> None:
    """Print resumability hints for a probe run that did not finish locally."""
    probe_run_path = output_dir / "probe_run.json"
    function_call_id = ""
    if probe_run_path.exists():
        try:
            probe_run = json.loads(probe_run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            probe_run = {}
        function_call_id = str(probe_run.get("function_call_id") or "").strip()
    else:
        # Backward-compatible with failed runs created by the previous local format.
        call_id_path = output_dir / "modal_function_call_id.txt"
        if call_id_path.exists():
            function_call_id = call_id_path.read_text(encoding="utf-8").strip()

    if function_call_id:
        prefill_flag = "--paged " if probe_prefill_path == "paged" else ""
        page_size_flag = (
            f"--probe-page-sizes {probe_page_size} "
            if probe_prefill_path == "paged" and probe_page_size
            else ""
        )
        log(f"📌 Probe Modal function_call_id: {function_call_id}")
        log(
            "   Resume this probe without launching a new cloud job: "
            f"python3 scripts/run_pipeline.py --probe "
            f"--probe-output-dir {output_dir} "
            f"--probe-resume-function-call-id {function_call_id} "
            f"{prefill_flag}"
            f"{page_size_flag}"
            "--skip-official-validate"
        )
    if probe_run_path.exists():
        log(f"   Probe run metadata/error marker: {probe_run_path}")


def _probe_prefill_passes(
    probe_prefill_path: str,
    resume_function_call_id: str,
    page_sizes: list[int] | None,
) -> list[tuple[str, str, int | None]]:
    """Return internal probe prefill passes for the requested coverage."""
    paged_sizes = [size for size in (page_sizes or []) if size > 0]

    def paged_passes() -> list[tuple[str, str, int | None]]:
        if not paged_sizes:
            return [("paged", "paged", None)]
        return [(f"paged_ps{size}", "paged", size) for size in paged_sizes]

    if probe_prefill_path == "paged":
        return paged_passes()
    if probe_prefill_path == "both" and not resume_function_call_id:
        return [("default", "default", None), *paged_passes()]
    return [("default", "default", None)]


def _probe_pass_output_dir(base_output_dir: Path, label: str, multi_pass: bool) -> Path:
    """Return the output dir for one probe pass."""
    return base_output_dir / f".probe_pass_{label}" if multi_pass else base_output_dir


def _merge_probe_pass_outputs(
    output_dir: Path,
    pass_dirs: list[tuple[str, Path]],
    *,
    log: Log,
) -> Path:
    """Merge multiple probe aggregated_summary.json files into one summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[tuple[str, Path, dict]] = []
    for label, pass_dir in pass_dirs:
        summary_path = pass_dir / "aggregated_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append((label, summary_path, summary))

    trace_counts: Counter[str] = Counter()
    signatures_by_key: dict[str, dict] = {}
    processes: list[dict] = []
    param_name_conflicts: list[dict] = []
    result_text_by_pass: dict[str, str | None] = {}
    error_by_pass: dict[str, str | None] = {}

    totals = {
        "raw_file_count": 0,
        "meta_file_count": 0,
        "total_events": 0,
        "warmup_events": 0,
    }
    for label, _summary_path, summary in summaries:
        pass_probe_page_size = summary.get("probe_page_size")
        for key in totals:
            totals[key] += int(summary.get(key, 0) or 0)
        result_text_by_pass[label] = summary.get("result_text")
        error_by_pass[label] = summary.get("error")
        processes.extend(summary.get("processes") or [])
        param_name_conflicts.extend(summary.get("param_name_conflicts") or [])

        for item in summary.get("top_trace_ids") or []:
            trace_id = item.get("trace_id")
            if isinstance(trace_id, str):
                trace_counts[trace_id] += int(item.get("count", 0) or 0)

        for item in summary.get("top_signatures") or []:
            item_for_merge = dict(item)
            item_for_merge["probe_pass_label"] = label
            if pass_probe_page_size:
                item_for_merge["probe_page_size"] = int(pass_probe_page_size)
            trace_id = item.get("trace_id", "")
            key = json.dumps(
                {
                    "trace_id": trace_id,
                    "category": item.get("category", ""),
                    "signature": item.get("signature", {}),
                    "probe_page_size": item_for_merge.get("probe_page_size"),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if key not in signatures_by_key:
                signatures_by_key[key] = item_for_merge
                signatures_by_key[key]["count"] = 0
            signatures_by_key[key]["count"] += int(item.get("count", 0) or 0)
            for optional_key in ("stack", "param_names"):
                if optional_key not in signatures_by_key[key] and optional_key in item:
                    signatures_by_key[key][optional_key] = item[optional_key]

    fi_trace_out_dir = output_dir / "fi_trace_out"
    written_fi_trace: set[str] = set()
    for _label, pass_dir in pass_dirs:
        src_dir = pass_dir / "fi_trace_out"
        if not src_dir.exists():
            continue
        fi_trace_out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob("*.json")):
            dst = fi_trace_out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            written_fi_trace.add(src.name)

    merged = dict(summaries[0][2])
    merged.update(totals)
    merged["trace_dir"] = str(output_dir)
    merged["probe_prefill_mode"] = "both"
    merged["probe_passes"] = [
        {
            "label": label,
            "aggregated_summary": str(summary_path),
            "probe_page_size": summary.get("probe_page_size"),
        }
        for label, summary_path, summary in summaries
    ]
    merged["probe_page_sizes"] = sorted(
        {
            int(summary.get("probe_page_size"))
            for _label, _summary_path, summary in summaries
            if summary.get("probe_page_size")
        }
    )
    merged["result_text_by_pass"] = result_text_by_pass
    merged["error_by_pass"] = error_by_pass
    merged["unique_trace_ids"] = len(trace_counts)
    merged["unique_signatures"] = len(signatures_by_key)
    merged["param_name_conflicts"] = param_name_conflicts
    merged["top_trace_ids"] = [
        {"trace_id": trace_id, "count": count}
        for trace_id, count in trace_counts.most_common()
    ]
    merged["top_signatures"] = sorted(
        signatures_by_key.values(),
        key=lambda item: (-int(item.get("count", 0) or 0), item.get("trace_id", "")),
    )
    merged["processes"] = processes
    merged["fi_trace_definition_count"] = len(written_fi_trace)
    if written_fi_trace:
        merged["fi_trace_out_dir"] = str(fi_trace_out_dir)
    else:
        merged.pop("fi_trace_out_dir", None)

    merged_path = output_dir / "aggregated_summary.json"
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"✅ Merged probe output: {merged_path}")
    return merged_path


def _run_llm_parse_rule_proposals(
    *,
    inventory_path: Path,
    summary_path: Path,
    dry_run: bool,
    log: Log,
) -> None:
    """Write review-only proposal files for LLM-classified trace IDs."""
    if dry_run and not inventory_path.exists():
        log("Would generate LLM parse-rule proposals if inventory contains llm_classified_trace_ids")
        return

    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"⚠️  Could not inspect inventory for LLM proposals: {exc}")
        return

    classified = inventory.get("llm_classified_trace_ids") or []
    if not classified:
        log("ℹ️  No LLM-classified trace IDs to write parse-rule proposals for")
        return

    project_root = Path(__file__).resolve().parents[2]
    tool_path = project_root / "tools" / "propose_parse_rules.py"
    out_dir = inventory_path.parent / "llm_diagnostics" / "parse_rules"
    args = [
        sys.executable,
        str(tool_path),
        "--inventory",
        str(inventory_path),
        "--summary",
        str(summary_path),
        "--out-dir",
        str(out_dir),
    ]
    log(f"Running: {' '.join(args)}")
    if dry_run:
        log("(dry-run, skipping)")
        return

    scripts_dir = project_root / "scripts"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(scripts_dir) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(args, env=env)
    if result.returncode != 0:
        log(f"⚠️  propose_parse_rules.py failed with exit code {result.returncode}")
        return
    log(f"✅ LLM parse-rule proposals: {out_dir / 'parse_rule_proposals.md'}")


def step_probe(
    config: ProbeConfig,
    *,
    project_root: Path,
    scheduler_path: Path,
    log: Log,
    log_step: Log,
) -> Path | None:
    """Run the Modal SGLang probe and return aggregated_summary.json."""
    log_step("Step 1: Probe (Modal GPU)")

    model_name = config.model_name
    tp = config.tp
    dry_run = config.dry_run
    prompt = config.prompt
    output_dir = config.output_dir
    probe_coverage = config.coverage
    probe_prefill_path = config.prefill_path
    probe_page_sizes = config.page_sizes
    probe_mem_fraction_static = config.mem_fraction_static
    force_flashinfer_backends = config.force_flashinfer_backends
    resume_function_call_id = config.resume_function_call_id
    detach = config.detach

    if tp <= 0:
        log(f"❌ Invalid TP for probe: {tp}")
        return None

    if not scheduler_path.exists():
        log(f"❌ Script 1 not found: {scheduler_path}")
        return None

    output_dir = output_dir or _default_probe_output_dir(model_name, project_root)
    pass_specs = _probe_prefill_passes(
        probe_prefill_path,
        resume_function_call_id,
        probe_page_sizes,
    )
    multi_pass = len(pass_specs) > 1
    if resume_function_call_id and probe_prefill_path == "both":
        log(
            "Probe resume uses one Modal function_call_id, so only the default "
            "prefill pass will be reattached. Use --paged to resume a paged pass."
        )

    if dry_run:
        detach_part = " --detach" if detach else ""
        force_part = " --force-flashinfer-backends" if force_flashinfer_backends else ""
        resume_part = (
            f" --resume-function-call-id {resume_function_call_id}"
            if resume_function_call_id
            else ""
        )
        for label, prefill_mode, probe_page_size in pass_specs:
            pass_output_dir = _probe_pass_output_dir(output_dir, label, multi_pass)
            page_size_part = (
                f" --probe-page-size {probe_page_size}"
                if prefill_mode == "paged" and probe_page_size is not None
                else ""
            )
            log(
                f"(dry-run) Would run ({label}): modal run{detach_part} {scheduler_path} "
                f"--model-name {model_name} --tp-size {tp} --prompt {prompt} "
                f"--probe-coverage {probe_coverage} --probe-prefill-mode {prefill_mode}"
                f" --probe-mem-fraction-static {probe_mem_fraction_static}"
                f"{page_size_part}{force_part}{resume_part} --output-dir {pass_output_dir}"
            )
        return output_dir / "aggregated_summary.json"

    probe_jobs: list[
        tuple[str, str, int | None, Path, object]
    ] = []
    logs_dir = _probe_logs_dir(output_dir)
    for label, prefill_mode, probe_page_size in pass_specs:
        pass_output_dir = _probe_pass_output_dir(output_dir, label, multi_pass)
        cmd = ["modal", "run"]
        if detach:
            cmd.append("--detach")
        cmd.extend([
            str(scheduler_path),
            "--model-name", model_name,
            "--tp-size", str(tp),
            "--prompt", prompt,
            "--probe-coverage", probe_coverage,
            "--probe-prefill-mode", prefill_mode,
            "--probe-mem-fraction-static", str(probe_mem_fraction_static),
        ])
        if prefill_mode == "paged" and probe_page_size is not None:
            cmd.extend(["--probe-page-size", str(probe_page_size)])
        if force_flashinfer_backends:
            cmd.append("--force-flashinfer-backends")
        if resume_function_call_id:
            cmd.extend(["--resume-function-call-id", resume_function_call_id])
        cmd.extend(["--output-dir", str(pass_output_dir)])

        log(f"Running ({label}): {' '.join(cmd)}")
        log_path = logs_dir / f"probe_{label}.log"
        if multi_pass:
            try:
                process = popen_streamed(
                    cmd,
                    cwd=project_root,
                    log_path=log_path,
                    prefix=f"probe {label}",
                )
            except OSError:
                failed = subprocess.CompletedProcess(cmd, 127)
                probe_jobs.append((label, prefill_mode, probe_page_size, pass_output_dir, failed))
                continue
            probe_jobs.append((label, prefill_mode, probe_page_size, pass_output_dir, process))
        else:
            result = subprocess.CompletedProcess(
                cmd,
                run_streamed(
                    cmd,
                    cwd=project_root,
                    log_path=log_path,
                    prefix=f"probe {label}",
                ),
            )
            probe_jobs.append((label, prefill_mode, probe_page_size, pass_output_dir, result))

    completed_pass_dirs: list[tuple[str, Path]] = []
    failed = False
    for label, prefill_mode, probe_page_size, pass_output_dir, process in probe_jobs:
        if hasattr(process, "process"):
            returncode = wait_streamed(process)
        else:
            returncode = process.returncode

        if returncode != 0:
            log(f"❌ Probe pass failed: {label}")
            _log_probe_resume_hint(
                pass_output_dir,
                log,
                probe_prefill_path=prefill_mode,
                probe_page_size=probe_page_size,
            )
            failed = True
            continue

        probe_output = pass_output_dir / "aggregated_summary.json"
        if not probe_output.exists():
            log(f"❌ Probe completed but expected output not found: {probe_output}")
            _log_probe_resume_hint(
                pass_output_dir,
                log,
                probe_prefill_path=prefill_mode,
                probe_page_size=probe_page_size,
            )
            failed = True
            continue
        log(f"✅ Probe output ({label}): {probe_output}")
        completed_pass_dirs.append((label, pass_output_dir))

    if failed:
        return None

    if multi_pass:
        return _merge_probe_pass_outputs(output_dir, completed_pass_dirs, log=log)
    return completed_pass_dirs[0][1] / "aggregated_summary.json"


def step_parse(
    config: ParseConfig,
    *,
    run_script: RunScript,
    log: Log,
    log_step: Log,
) -> Path | None:
    """Parse probe output into kernel_inventory.json."""
    log_step("Step 2: Parse Probe Results")

    probe_output = config.probe_output
    model_name = config.model_name
    definitions_dir = config.definitions_dir
    hf_config = config.hf_config
    output_path = config.output_path
    dry_run = config.dry_run

    if hf_config is None and not dry_run:
        hf_config = auto_download_hf_config(model_name, log=log)
        if hf_config is None:
            log("[HF] Proceeding without HF config — GQA/MLA kernels may be NEEDS_CONFIG")

    args = [
        str(probe_output),
        "--model-name", model_name,
        "--definitions-dir", str(definitions_dir),
    ]
    if hf_config:
        args.extend(["--hf-config", str(hf_config)])
    if output_path:
        args.extend(["--output", str(output_path)])
    if config.llm_classify:
        args.append("--llm-classify")
    args.extend(["--llm-model", config.llm_model])
    if config.llm_base_url:
        args.extend(["--llm-base-url", config.llm_base_url])

    rc = run_script("parse/parse_probe.py", args, dry_run)
    if rc != 0:
        log("❌ parse_probe.py failed")
        return None

    inventory_path = output_path or probe_output.parent / "kernel_inventory.json"
    if inventory_path.exists() or dry_run:
        log(f"✅ Kernel inventory: {inventory_path}")
        if config.llm_classify:
            _run_llm_parse_rule_proposals(
                inventory_path=inventory_path,
                summary_path=probe_output,
                dry_run=dry_run,
                log=log,
            )
        return inventory_path
    log("❌ kernel_inventory.json not found")
    return None


def step_static_candidates(
    config: StaticCandidatesConfig,
    *,
    project_root: Path,
    run_script: RunScript,
    log: Log,
    log_step: Log,
) -> Path | None:
    """Generate a candidate kernel_inventory.json from static facts."""
    log_step("Step 1: Static Candidates")

    model_name = config.model_name
    hf_config = config.hf_config
    dry_run = config.dry_run
    output_path = config.output_path

    if hf_config is None and not dry_run:
        hf_config = auto_download_hf_config(model_name, log=log)
        if hf_config is None:
            log("❌ Static candidates require HF config. Pass --hf-config or install/configure huggingface_hub.")
            return None

    if output_path is None:
        output_path = project_root / "kernel_inventory.json"

    args = [
        "--model-name", model_name,
        "--hf-config", str(hf_config or Path("(auto-hf-config)")),
        "--definitions-dir", str(config.definitions_dir),
        "--sglang-root", str(resolve_under_project(config.sglang_root, project_root)),
        "--tp", str(config.tp),
        "--output", str(output_path),
    ]
    page_sizes = [size for size in (config.page_sizes or []) if size > 0]
    if page_sizes:
        args.extend(["--page-sizes", *[str(size) for size in page_sizes]])
    rc = run_script("static/main.py", args, dry_run=dry_run)
    if rc != 0:
        log("❌ static candidate generation failed")
        return None
    if output_path.exists() or dry_run:
        log(f"✅ Static kernel inventory: {output_path}")
        return output_path
    log("❌ static kernel_inventory.json not found")
    return None
