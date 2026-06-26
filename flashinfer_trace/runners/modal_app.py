"""Modal CLI entrypoints for streaming probe runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal


def _modal_image_name() -> str:
    return os.environ.get("FLASHINFER_TRACE_MODAL_IMAGE", "lmsysorg/sglang:v0.5.12.post1")


def _modal_gpu() -> str | None:
    value = os.environ.get("FLASHINFER_TRACE_MODAL_GPU", "").strip()
    return value or None


def _modal_timeout() -> int:
    value = os.environ.get("FLASHINFER_TRACE_MODAL_TIMEOUT", "3600")
    try:
        return max(1, int(value))
    except ValueError:
        return 3600


app = modal.App("flashinfer-trace-probe")
image = modal.Image.from_registry(_modal_image_name()).add_local_python_source("flashinfer_trace", copy=True)
secrets = [modal.Secret.from_name("huggingface-secret")]


@app.function(
    image=image,
    gpu=_modal_gpu(),
    timeout=_modal_timeout(),
    name="run_sglang_probe",
    serialized=True,
    secrets=secrets,
)
def run_sglang_probe(plan: dict[str, Any]) -> dict[str, Any]:
    from flashinfer_trace.runners.modal_runner import run_remote_sglang_probe

    return run_remote_sglang_probe(plan)


@app.local_entrypoint()
def probe(plan_path: str, output_dir: str, resume_call_id: str = "") -> None:
    """Run a Modal probe from a local modal_probe_plan.json file."""
    from flashinfer_trace.runners.modal_runner import materialize_modal_result

    plan_file = Path(plan_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))

    print(f"[flashinfer_trace] Modal CLI plan: {plan_file}", flush=True)
    print(f"[flashinfer_trace] Modal CLI output: {out_dir}", flush=True)
    print(
        "[flashinfer_trace] Modal CLI runtime: "
        f"image={_modal_image_name()} gpu={_modal_gpu() or 'none'} timeout={_modal_timeout()}s",
        flush=True,
    )
    if resume_call_id:
        print(f"[flashinfer_trace] Reattaching Modal call: {resume_call_id}", flush=True)
        call = modal.FunctionCall.from_id(resume_call_id)
        function_call_id = resume_call_id
    else:
        call = run_sglang_probe.spawn(plan)
        function_call_id = str(call.object_id)
        print(f"[flashinfer_trace] Modal function_call_id: {function_call_id}", flush=True)
        print(
            "[flashinfer_trace] Resume command: "
            f"modal run -m flashinfer_trace.runners.modal_app::probe "
            f"--plan-path {plan_file} --output-dir {out_dir} --resume-call-id {function_call_id}",
            flush=True,
        )
    metadata = {
        "status": "submitted",
        "function_call_id": function_call_id,
        "plan_path": str(plan_file),
        "output_dir": str(out_dir),
        "resume_command": (
            f"modal run -m flashinfer_trace.runners.modal_app::probe "
            f"--plan-path {plan_file} --output-dir {out_dir} --resume-call-id {function_call_id}"
        ),
    }
    (out_dir / "probe_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = call.get()
    metadata["status"] = "completed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    materialize_modal_result(result, out_dir)
