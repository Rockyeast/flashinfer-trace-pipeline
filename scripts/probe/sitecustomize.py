"""Python startup hook injected into SGLang probe worker processes."""

from __future__ import annotations

import importlib
import os
import sys
import traceback


print(f"[sitecustomize] loaded from {__file__}", file=sys.stderr, flush=True)
print(
    "[sitecustomize] SGLANG_WORKER_TRACE_DIR = "
    f"{os.environ.get('SGLANG_WORKER_TRACE_DIR')}",
    file=sys.stderr,
    flush=True,
)

try:
    install_from_env = importlib.import_module("worker_trace_runtime").install_from_env
    install_from_env()
except Exception as exc:
    print(
        "[sitecustomize] failed to install worker trace runtime: "
        f"{exc}\n{traceback.format_exc()}",
        flush=True,
    )

try:
    if os.environ.get("PROBE_DISABLE_RUNTIME_FI_TRACE_ATTACH", "") not in (
        "1",
        "true",
        "True",
    ):
        attach_fi_trace = importlib.import_module(
            "fi_trace_runtime_patch"
        ).attach_fi_trace
        attach_fi_trace()
except Exception as exc:
    print(
        "[sitecustomize] failed to attach fi_trace templates: "
        f"{exc}\n{traceback.format_exc()}",
        flush=True,
    )
