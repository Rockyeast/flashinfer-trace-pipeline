"""Shared runtime defaults for Modal probe and collect jobs."""

from __future__ import annotations

import os
from collections.abc import Mapping


DEFAULT_MODAL_GPU = "A100-80GB"
DEFAULT_MODAL_GPU_COUNT = 2
DEFAULT_SGLANG_IMAGE = "lmsysorg/sglang:v0.5.12.post1"
DEFAULT_SGLANG_PACKAGE = "sglang[all]==0.5.12.post1"
DEFAULT_EXTRA_PIP_PACKAGES = "kernels==0.14.1"
DEFAULT_FLASHINFER_BENCH_PROBE_PACKAGE = "flashinfer-bench==0.1.2"
DEFAULT_FLASHINFER_BENCH_COLLECT_PACKAGE = "flashinfer-bench>=0.1.2"

DEFAULT_FORWARD_ENV_NAMES = (
    "PROBE_MEM_FRACTION_STATIC",
    "PROBE_REQUIRE_OFFICIAL_FI_TRACE",
    "PROBE_SKIP_FLASHINFER_PATCH",
    "PROBE_DISABLE_RUNTIME_FI_TRACE_ATTACH",
    "PROBE_FLASHINFER_PATCH_REPO",
    "PROBE_FLASHINFER_PATCH_REF",
)
DEFAULT_FORWARD_ENV_PREFIXES = ("SGLANG_",)
DEFAULT_FORWARD_ENV_EXCLUDES = (
    "SGLANG_FLASHINFER_USE_PAGED",
    "SGLANG_WORKER_TRACE_DIR",
    "SGLANG_WORKER_TRACE_FAMILIES",
)


def split_env_list(raw: str | None) -> tuple[str, ...]:
    """Parse comma/space separated environment variable names or prefixes."""
    if raw is None:
        return ()
    return tuple(item for item in raw.replace(",", " ").split() if item)


def forwarded_runtime_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return runtime env vars that should be copied into Modal images.

    Built-in names preserve current probe controls. Prefix forwarding is
    configurable through PROBE_FORWARD_ENV_PREFIXES and defaults to SGLANG_* so
    new SGLang runtime flags do not require code changes.
    """
    env = environ or os.environ
    names = set(DEFAULT_FORWARD_ENV_NAMES)
    names.update(split_env_list(env.get("PROBE_FORWARD_ENV_NAMES")))

    prefix_raw = env.get("PROBE_FORWARD_ENV_PREFIXES")
    prefixes = (
        split_env_list(prefix_raw)
        if prefix_raw is not None
        else DEFAULT_FORWARD_ENV_PREFIXES
    )
    excludes = set(DEFAULT_FORWARD_ENV_EXCLUDES)
    excludes.update(split_env_list(env.get("PROBE_FORWARD_ENV_EXCLUDES")))

    out: dict[str, str] = {}
    for name, value in env.items():
        if not value or name in excludes:
            continue
        if name in names or any(name.startswith(prefix) for prefix in prefixes):
            out[name] = value
    return out
