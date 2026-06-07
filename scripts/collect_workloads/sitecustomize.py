"""Build the injected sitecustomize.py used by SGLang worker processes."""

from __future__ import annotations

import json

from collect_workloads.hook_specs import PLAN_KWARG_MAP


def build_sitecustomize(specs: dict, capture_dir: str, debug: bool = False) -> str:
    """Return sitecustomize.py content that configures and imports _fi_hook."""
    lines = [
        f'import os as _os0; _os0.environ.setdefault("FI_CAPTURE_DIR", {capture_dir!r})',
        f'import os as _os1; _os1.environ.setdefault("FI_HOOK_SPECS",  {json.dumps(specs)!r})',
        f'import os as _os2; _os2.environ.setdefault("FI_PLAN_MAP",    {json.dumps(PLAN_KWARG_MAP)!r})',
        f'import os as _os3; _os3.environ.setdefault("FI_HOOK_VERBOSE", {"1" if debug else "0"!r})',
        "from collect_workloads import _fi_hook",
    ]
    return "\n".join(lines)
