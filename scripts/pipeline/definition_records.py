"""Definition loading and collect-scope helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path


LogFn = Callable[[str], None]


def _noop_log(_message: str) -> None:
    return None


def is_collectable_kernel(kernel: dict) -> bool:
    """Return whether a kernel can be collected by the fast hook collector."""
    hook_api = kernel.get("fi_api")
    def_name = kernel.get("definition_name", "")
    return (
        isinstance(hook_api, str)
        and hook_api.startswith("flashinfer.")
        and "NEEDS_CONFIG" not in def_name
    )


def is_paged_prefill_kernel(kernel: dict) -> bool:
    """Identify paged-prefill definitions, which need the piecewise graph pass."""
    fi_api = kernel.get("fi_api", "")
    return isinstance(fi_api, str) and fi_api.startswith(
        "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper"
    )


def fi_api_from_definition(defn: dict) -> str | None:
    """Extract the official FlashInfer API tag from one definition JSON."""
    fi_api = next(
        (
            tag.split(":", 1)[1]
            for tag in defn.get("tags", [])
            if isinstance(tag, str) and tag.startswith("fi_api:")
        ),
        None,
    )
    return fi_api if isinstance(fi_api, str) else None


def definition_record_from_file(path: Path, *, log: LogFn = _noop_log) -> dict | None:
    """Load one definition JSON and return normalized scope metadata."""
    try:
        defn = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log(f"WARNING: skipping unreadable definition {path}: {exc}")
        return None

    name = defn.get("name") or path.stem
    op_type = defn.get("op_type")
    if not isinstance(name, str) or not name:
        log(f"WARNING: skipping definition with missing name: {path}")
        return None

    fi_api = fi_api_from_definition(defn)
    return {
        "definition_name": name,
        "op_type": op_type if isinstance(op_type, str) else "",
        "fi_api": fi_api if isinstance(fi_api, str) else "",
        "path": path,
    }


def definition_records(
    definitions_dir: Path,
    *,
    scope: str,
    log: LogFn = _noop_log,
) -> list[dict]:
    """Read definitions_dir once and return de-duplicated definition metadata."""
    records: dict[str, dict] = {}
    for path in sorted(definitions_dir.rglob("*.json")):
        record = definition_record_from_file(path, log=log)
        if record is None:
            continue
        name = record["definition_name"]
        if name in records:
            log(f"WARNING: duplicate definition name in {scope} scope, keeping first: {name}")
            continue
        records[name] = record
    return [records[name] for name in sorted(records)]


def is_repo_global_definitions_dir(definitions_dir: Path, *, project_root: Path) -> bool:
    """Return True when a definitions dir points at the repo-global definitions/."""
    try:
        return definitions_dir.resolve() == (project_root / "definitions").resolve()
    except OSError:
        return False


def collect_definition_groups(
    definitions_dir: Path,
    *,
    log: LogFn = _noop_log,
) -> tuple[list[str], list[str]]:
    """Split collect scope from actual definition JSON files."""
    collectable: dict[str, dict] = {}
    for record in definition_records(definitions_dir, scope="collect", log=log):
        kernel = {
            "definition_name": record["definition_name"],
            "op_type": record["op_type"],
            "fi_api": record["fi_api"],
        }
        if not is_collectable_kernel(kernel):
            continue
        def_name = kernel["definition_name"]
        collectable[def_name] = kernel

    normal_defs: list[str] = []
    paged_prefill_defs: list[str] = []
    for def_name, kernel in sorted(collectable.items()):
        def_name = kernel.get("definition_name", "")
        if not def_name:
            continue
        if is_paged_prefill_kernel(kernel):
            paged_prefill_defs.append(def_name)
        else:
            normal_defs.append(def_name)

    return normal_defs, paged_prefill_defs
