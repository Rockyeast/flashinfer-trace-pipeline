"""Local definition loading helpers for workload collect."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from collect_workloads.hook_specs import extract_official_fi_api
from artifact_schemas import validate_definition


def find_definition_file(definitions_dir: Path, name: str) -> Path | None:
    """Find a definition JSON by name under a local definitions root."""
    direct = definitions_dir / f"{name}.json"
    if direct.exists():
        return direct
    for path in definitions_dir.rglob(f"{name}.json"):
        return path
    return None


def load_definition_payloads(
    definitions_dir: Path,
    definition_names: list[str],
) -> tuple[dict[str, str], list[str]]:
    """Load definition JSON text locally so Modal can use custom definition roots."""
    payloads: dict[str, str] = {}
    missing: list[str] = []
    for name in definition_names:
        path = find_definition_file(definitions_dir, name)
        if path is None:
            missing.append(name)
            continue
        payloads[name] = path.read_text()
    return payloads, missing


def load_definition_json(path: Path) -> dict | None:
    """Load one definition JSON, logging and returning None on invalid input."""
    try:
        data = json.loads(path.read_text())
        validate_definition(data)
        return data
    except Exception as exc:
        print(f"WARNING: cannot read definition {path}: {exc}", file=sys.stderr)
        return None


def iter_definition_files(definitions_dir: Path, op_type: str = "") -> list[Path]:
    """Return definition JSON files under a root or one op_type directory."""
    root = definitions_dir / op_type if op_type else definitions_dir
    if not root.exists():
        return []
    return sorted(root.glob("*.json")) if op_type else sorted(root.rglob("*.json"))


def collectable_definition_names(definitions_dir: Path, op_type: str = "") -> list[str]:
    """Return definition names that exist locally and target real FlashInfer APIs."""
    names: list[str] = []
    seen: set[str] = set()
    for path in iter_definition_files(definitions_dir, op_type):
        defn = load_definition_json(path)
        if not defn or extract_official_fi_api(defn) is None:
            continue
        name = str(defn.get("name") or path.stem)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names
