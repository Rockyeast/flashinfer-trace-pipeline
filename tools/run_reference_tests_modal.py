"""Run generated reference tests on Modal GPU.

Usage:
  REFERENCE_TESTS_DIR=tmp/run/Qwen_Qwen3-0.6B_20260502_194411/tests/references \\
  DEFINITIONS_DIR=tmp/run/Qwen_Qwen3-0.6B_20260502_194411/definitions \\
    modal run tools/run_reference_tests_modal.py --all

  REFERENCE_TESTS_DIR=tmp/run/Qwen_Qwen3-0.6B_20260502_194411/tests/references \\
  DEFINITIONS_DIR=tmp/run/Qwen_Qwen3-0.6B_20260502_194411/definitions \\
    modal run tools/run_reference_tests_modal.py --tests test_rmsnorm_h1024.py,test_silu_and_mul_i3072.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal


LOCAL_TESTS_DIR = Path(os.environ.get("REFERENCE_TESTS_DIR", "tests/references"))
LOCAL_DEFINITIONS_DIR = Path(os.environ.get("DEFINITIONS_DIR", "definitions"))
REMOTE_TESTS_DIR = Path("/root/tests/references")
REMOTE_DEFINITIONS_DIR = Path("/root/definitions")

app = modal.App("flashinfer-reference-test-runner")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("libnuma1", "build-essential", "git")
    .pip_install("pytest>=7.0")
    .pip_install("sglang[all]>=0.5.9")
    .pip_install("flashinfer-bench>=0.1.2")
    .pip_install("nvidia-cudnn-cu12==9.16.0.29")
    .pip_install("safetensors", "numpy")
    .env(
        {
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(LOCAL_TESTS_DIR, remote_path=str(REMOTE_TESTS_DIR))
    .add_local_dir(LOCAL_DEFINITIONS_DIR, remote_path=str(REMOTE_DEFINITIONS_DIR))
)


def _discover_local_tests() -> list[str]:
    if not LOCAL_TESTS_DIR.exists():
        raise FileNotFoundError(f"REFERENCE_TESTS_DIR does not exist: {LOCAL_TESTS_DIR}")
    if not LOCAL_DEFINITIONS_DIR.exists():
        raise FileNotFoundError(f"DEFINITIONS_DIR does not exist: {LOCAL_DEFINITIONS_DIR}")
    return sorted(p.name for p in LOCAL_TESTS_DIR.glob("test_*.py"))


@app.function(image=image, gpu=os.environ.get("MODAL_GPU", "A10G"), timeout=1800, retries=0)
def run_tests_remote(test_files: list[str]) -> dict:
    import importlib
    import torch

    versions = {}
    for name in ("torch", "flashinfer", "sglang", "sgl_kernel"):
        try:
            mod = importlib.import_module(name)
            versions[name] = {
                "version": getattr(mod, "__version__", "?"),
                "path": getattr(mod, "__file__", "?"),
            }
        except Exception as exc:
            versions[name] = {"error": str(exc)}

    results = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": versions,
        "tests": {},
    }

    for test_file in test_files:
        path = REMOTE_TESTS_DIR / test_file
        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", str(path)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            results["tests"][test_file] = {
                "returncode": proc.returncode,
                "success": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            print(f"{test_file}: {'PASS' if proc.returncode == 0 else 'FAIL'}", flush=True)
        except subprocess.TimeoutExpired as exc:
            results["tests"][test_file] = {
                "returncode": None,
                "success": False,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "error": "timeout",
            }
            print(f"{test_file}: TIMEOUT", flush=True)

    return results


@app.local_entrypoint()
def main(
    all: bool = False,
    tests: str = "",
    output: str = "logs/reference_test_results.json",
):
    available = _discover_local_tests()
    selected = available if all else [item.strip() for item in tests.split(",") if item.strip()]
    if not selected:
        raise SystemExit("Pass --all or --tests test_x.py,test_y.py")

    missing = [name for name in selected if name not in available]
    if missing:
        raise SystemExit(f"Unknown test files: {missing}")

    print(f"REFERENCE_TESTS_DIR={LOCAL_TESTS_DIR}")
    print(f"DEFINITIONS_DIR={LOCAL_DEFINITIONS_DIR}")
    print(f"Running {len(selected)} test file(s) on Modal GPU")
    result = run_tests_remote.remote(selected)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    passed = sum(1 for item in result["tests"].values() if item["success"])
    total = len(result["tests"])
    print(f"Saved: {out_path}")
    print(f"Summary: {passed}/{total} passed")
    if passed != total:
        raise SystemExit(1)
