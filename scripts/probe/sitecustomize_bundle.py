"""Build the sitecustomize injection bundle used by probe workers.

The bundle is a temporary PYTHONPATH directory containing:

- sitecustomize.py: copy of scripts/probe/sitecustomize.py.
- worker_trace_runtime.py: copy of scripts/probe/runtime.py.
- fi_trace_runtime_patch.py: copy of scripts/probe/fi_trace_runtime_patch.py.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def prepare_sitecustomize_bundle(
    *,
    custom_site_dir: Path,
    trace_dir: Path,
    remote_sitecustomize_source_path: Path,
    remote_runtime_source_path: Path,
    remote_fi_trace_runtime_patch_path: Path | None,
    fi_trace_out_dir: Path,
) -> None:
    """Create the temporary Python startup-hook bundle for probe workers."""
    shutil.rmtree(custom_site_dir, ignore_errors=True)
    shutil.rmtree(trace_dir, ignore_errors=True)
    custom_site_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    (custom_site_dir / "sitecustomize.py").write_text(
        remote_sitecustomize_source_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (custom_site_dir / "worker_trace_runtime.py").write_text(
        remote_runtime_source_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    if remote_fi_trace_runtime_patch_path is not None:
        (custom_site_dir / "fi_trace_runtime_patch.py").write_text(
            remote_fi_trace_runtime_patch_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    pythonpath_parts = [str(custom_site_dir)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)
    if str(custom_site_dir) not in sys.path:
        sys.path.insert(0, str(custom_site_dir))

    os.environ["SGLANG_WORKER_TRACE_DIR"] = str(trace_dir)
    os.environ["SGLANG_WORKER_TRACE_FAMILIES"] = ""
    os.environ["FIB_ENABLE_TRACING"] = "0"
    os.environ.pop("FIB_DATASET_PATH", None)
    os.environ["FLASHINFER_TRACE_DUMP"] = "1"
    os.environ["FLASHINFER_TRACE_DUMP_DIR"] = str(fi_trace_out_dir)
