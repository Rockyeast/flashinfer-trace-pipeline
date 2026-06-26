"""Prepare and check review-only proposal inputs.

This is an offline onboarding tool, not part of the deterministic
``flashinfer_trace`` runtime core. Discovery belongs in the external
agent/skill workflow; this module only prepares source material, checks an
agent-authored proposal, and leaves approved runtime inputs for human review.
"""

from __future__ import annotations

import ast
import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import string
import time
from datetime import datetime
from dataclasses import fields
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from flashinfer_trace.core.schemas import CaptureSpec


DEFAULT_COOKBOOK_REPO = "https://github.com/sgl-project/sgl-cookbook.git"
DEFAULT_FLASHINFER_ROOTS = (
    Path("agent_inputs/flashinfer/flashinfer"),
)
DEFAULT_SGLANG_ROOT = Path("agent_inputs/sglang/python/sglang")
DEFAULT_FLASHINFER_ROOT = Path("agent_inputs/flashinfer/flashinfer")
DEFAULT_COOKBOOK_ROOT = Path("agent_inputs/sgl-cookbook")
CAPTURE_SPEC_FIELDS = {field_info.name for field_info in fields(CaptureSpec)}
DEFINITION_REQUIRED_FIELDS = {"name", "op_type", "axes", "inputs", "outputs"}
HINT_REQUIRED_FIELDS = {"schema_version", "definition_name", "op_type", "inputs"}
KNOWN_NON_FITRACE_COLLECTABLE_OPS = {"rmsnorm", "silu_and_mul"}
COMPANION_REQUIRED_WRAPPER_SUFFIXES = (
    "BatchDecodeWithPagedKVCacheWrapper.run",
    "BatchPrefillWithPagedKVCacheWrapper.run",
    "BatchPrefillWithRaggedKVCacheWrapper.run",
)
FLASHINFER_ATTENTION_TARGET_PREFIXES = ("flashinfer.decode.", "flashinfer.prefill.")


def slug_model_name(model_name: str) -> str:
    """Return a stable local filename slug for a HF model name."""
    last = model_name.strip().split("/")[-1]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", last).strip("_").lower()
    return slug or "model"


def _default_run_prefix(model_name: str) -> Path:
    return Path(slug_model_name(model_name)) / f"{datetime.now().strftime('%Y%m%d')}_firstpass"


def _default_hf_config_path(model_name: str) -> Path:
    return Path("agent_inputs/config") / f"{slug_model_name(model_name)}.json"


def _default_merge_output_dir(run_prefix: Path) -> Path:
    base = _resolve_run_dir(run_prefix)
    return base.with_name(f"{base.name}_merged") / "proposal"


def _find_codex_binary() -> Path:
    resolved = shutil.which("codex")
    if resolved:
        return Path(resolved)
    candidates = sorted(Path.home().glob(".vscode-server/extensions/*/bin/linux-x86_64/codex"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("codex binary not found on PATH or under ~/.vscode-server/extensions")


def _agent_command_from_shortcut(agent: str | None) -> tuple[list[str] | None, dict[str, str] | None]:
    if agent is None:
        return None, None
    if agent != "codex":
        raise ValueError(f"unsupported agent shortcut: {agent}")
    codex_bin = _find_codex_binary()
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(codex_bin.parent), env.get("PATH", "")])
    return [
        str(codex_bin),
        "exec",
        "-C",
        str(Path.cwd()),
        "-s",
        "workspace-write",
        "--ephemeral",
        "-",
    ], env


def _run_git(args: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": ["git", *args],
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def ensure_sgl_cookbook(*, root: Path, refresh: bool, repo_url: str = DEFAULT_COOKBOOK_REPO) -> dict[str, Any]:
    """Ensure sgl-cookbook exists under root."""
    target = root / "sgl-cookbook"
    if target.exists():
        if not (target / ".git").exists():
            return {"path": str(target), "status": "exists_not_git", "ok": False}
        if not refresh:
            return {"path": str(target), "status": "exists", "ok": True}
        result = _run_git(["pull", "--ff-only"], cwd=target)
        return {
            "path": str(target),
            "status": "updated" if result["returncode"] == 0 else "update_failed",
            "ok": result["returncode"] == 0,
            "git": result,
        }
    root.mkdir(parents=True, exist_ok=True)
    result = _run_git(["clone", "--depth", "1", repo_url, str(target)])
    return {
        "path": str(target),
        "status": "cloned" if result["returncode"] == 0 else "clone_failed",
        "ok": result["returncode"] == 0,
        "git": result,
    }


def fetch_hf_config(*, model_name: str, config_dir: Path, refresh: bool) -> dict[str, Any]:
    """Download one HuggingFace config.json unless an existing copy is accepted."""
    config_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_model_name(model_name)
    path = config_dir / f"{slug}.json"
    if path.exists() and not refresh:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        return {
            "model": model_name,
            "slug": slug,
            "path": str(path),
            "status": "exists",
            "ok": True,
            "model_type": data.get("model_type"),
        }

    url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    req = Request(url, headers={"User-Agent": "flashinfer-trace-redesign-config-fetch"})
    try:
        with urlopen(req, timeout=60) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "model": model_name,
            "slug": slug,
            "path": str(path),
            "status": "downloaded",
            "ok": True,
            "model_type": data.get("model_type"),
        }
    except Exception as exc:  # noqa: BLE001 - report diagnostics for CLI users
        return {
            "model": model_name,
            "slug": slug,
            "path": str(path),
            "status": "download_failed",
            "ok": False,
            "error": str(exc),
        }


def _version_key(path: Path) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", path.name)
    return tuple(int(number) for number in numbers)


def _latest_generated_dirs(cookbook_root: Path) -> list[Path]:
    generated = cookbook_root / "data/models/generated"
    if not generated.exists():
        return []
    dirs = [path for path in generated.iterdir() if path.is_dir()]
    return sorted(dirs, key=_version_key, reverse=True)


def _tokens_for_model(model_name: str) -> set[str]:
    text = model_name.split("/")[-1].lower()
    pieces = re.split(r"[^a-z0-9]+", text)
    tokens = {piece for piece in pieces if len(piece) >= 3}
    if "qwen3" in text:
        tokens.add("qwen3")
    if "tinyllama" in text:
        tokens.add("tinyllama")
        tokens.add("llama")
    return tokens


def find_cookbook_candidates(*, cookbook_root: Path, model_name: str, limit: int = 8) -> dict[str, Any]:
    """Find likely sgl-cookbook YAMLs. Missing is diagnostic, not fatal."""
    latest_dirs = _latest_generated_dirs(cookbook_root)
    tokens = _tokens_for_model(model_name)
    matches: list[str] = []
    searched: list[str] = []
    for directory in latest_dirs:
        searched.append(str(directory))
        for yaml_path in sorted(directory.glob("*.yaml")):
            name = yaml_path.stem.lower()
            if any(token in name for token in tokens):
                matches.append(str(yaml_path))
                if len(matches) >= limit:
                    return {
                        "model": model_name,
                        "status": "found",
                        "ok": True,
                        "matches": matches,
                        "searched_latest_first": searched,
                    }
    status = "found" if matches else "missing"
    return {"model": model_name, "status": status, "ok": True, "matches": matches, "searched_latest_first": searched}


def prepare_agent_inputs(
    *,
    models: list[str],
    output_root: Path,
    refresh: bool = False,
    cookbook_repo: str = DEFAULT_COOKBOOK_REPO,
    check_sglang_root: Path | None = None,
    check_flashinfer_root: Path | None = None,
) -> dict[str, Any]:
    """Prepare external inputs and return a reproducible report."""
    output_root.mkdir(parents=True, exist_ok=True)
    cookbook = ensure_sgl_cookbook(root=output_root, refresh=refresh, repo_url=cookbook_repo)
    configs = [
        fetch_hf_config(model_name=model, config_dir=output_root / "config", refresh=refresh)
        for model in models
    ]

    cookbook_root = output_root / "sgl-cookbook"
    cookbook_matches = [
        find_cookbook_candidates(cookbook_root=cookbook_root, model_name=model)
        for model in models
    ]

    source_checks = []
    for label, path in (("sglang", check_sglang_root), ("flashinfer", check_flashinfer_root)):
        if path is None:
            source_checks.append({"name": label, "path": None, "status": "not_checked", "ok": True})
        else:
            source_checks.append({
                "name": label,
                "path": str(path),
                "status": "exists" if path.exists() else "missing",
                "ok": path.exists(),
            })

    return {
        "summary": {
            "models": len(models),
            "configs_ok": sum(1 for item in configs if item.get("ok")),
            "cookbook_ok": cookbook.get("ok", False),
            "cookbook_matches": sum(1 for item in cookbook_matches if item.get("matches")),
            "source_checks_ok": sum(1 for item in source_checks if item.get("ok")),
        },
        "output_root": str(output_root),
        "refresh": refresh,
        "sgl_cookbook": cookbook,
        "configs": configs,
        "cookbook_candidates": cookbook_matches,
        "source_checks": source_checks,
    }


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _check_candidate_fields(path: Path) -> dict[str, Any]:
    """Return field findings for candidate/approved target JSON.

    This is a static field check for proposal hygiene. It does not import targets or
    decide approval automatically.
    """
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"candidate targets must be a list: {path}")

    findings: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            findings.append({
                "severity": "error",
                "name": f"#{index}",
                "reason": "entry is not a JSON object",
            })
            continue
        name = str(item.get("name") or f"#{index}")
        status = item.get("status")
        role = item.get("role", "target")
        backend = item.get("backend", "unknown")
        target = item.get("target")
        module = item.get("module")
        attr = item.get("attr")
        collect = item.get("collect", False)
        definition_source = item.get("definition_source", "unknown")
        definition_name = item.get("definition_name")
        op_type = item.get("op_type")
        raw_companions = item.get("companion_attrs", [])
        companion_attrs = raw_companions if isinstance(raw_companions, list) else []
        capture = item.get("capture")
        dispatch = item.get("dispatch")
        dispatch_value = item.get("dispatch_value")

        if not isinstance(collect, bool):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "collect must be a boolean",
            })

        if role not in {"target", "warmup"}:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "role must be target or warmup",
            })
            continue
        if role == "warmup":
            if not isinstance(module, str) or not module or not isinstance(attr, str) or not attr:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "warmup entry requires explicit module/attr",
                })
            if collect is True:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "warmup entry cannot use collect=true",
                })
            if isinstance(target, str) and target:
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "warmup entry ignores target; use module/attr as the reviewed hook spec",
                })
            if capture is not None:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "warmup entry must not declare capture",
                })
            if dispatch is not None:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "warmup entry must not declare dispatch",
                })
            continue

        if not isinstance(capture, dict):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "target entry must declare capture",
            })
        else:
            unexpected_capture = sorted(set(capture) - CAPTURE_SPEC_FIELDS)
            if unexpected_capture:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": f"capture has unexpected fields: {unexpected_capture}",
                })
        if item.get("page_size") is not None and not isinstance(dispatch, dict):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "page_size target must declare reviewed dispatch",
            })
        if isinstance(dispatch, dict) and item.get("page_size") is None and dispatch_value is None:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "non-page_size dispatch target must declare dispatch_value",
            })
        if dispatch_value is not None and type(dispatch_value) is not int:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "dispatch_value must be an integer",
            })

        if status == "approved" and not isinstance(target, str):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "approved entry has no hook target",
            })
        if (
            backend != "flashinfer"
            and isinstance(op_type, str)
            and op_type in KNOWN_NON_FITRACE_COLLECTABLE_OPS
        ):
            if collect is not True:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": f"known non-FlashInfer op {op_type} must be proposed with collect=true",
                })
            if definition_source != "agent":
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": f"known non-FlashInfer op {op_type} must use definition_source=agent",
                })
        if backend != "flashinfer" and collect is True:
            if not isinstance(definition_name, str) or not definition_name:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "non-FlashInfer collect target must declare reviewed definition_name",
                })
            if definition_source not in {"agent", "manual"}:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "non-FlashInfer collect target must use definition_source=agent or manual",
                })
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "non-FlashInfer collect target should include source evidence for reviewed definition/hints",
                })
        if backend == "flashinfer" and collect is True:
            if not isinstance(module, str) or not module or not isinstance(attr, str) or not attr:
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "collectable FlashInfer target should declare explicit module/attr hook spec",
                })
            if _requires_companion_attrs(target, attr) and not companion_attrs:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": (
                        "FlashInfer attention wrapper collect target must declare reviewed "
                        "companion_attrs so scalars like sm_scale are captured from sibling calls"
                    ),
                })
            if definition_source == "agent":
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": "collectable FlashInfer target must not use agent-guessed final definition",
                })
            elif (
                status != "approved"
                and definition_source == "fitrace"
                and isinstance(definition_name, str)
                and definition_name
            ):
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "fitrace target has preview definition_name; final name must be verified from fitrace dump",
                })
            elif definition_source == "unknown":
                findings.append({
                    "severity": "warning",
                    "name": name,
                    "reason": "collectable FlashInfer target should declare definition_source=fitrace or manual",
                })
        if definition_source == "fitrace" and not isinstance(target, str):
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "fitrace definition source needs a FlashInfer API target",
            })

    summary = {
        "entries": len(raw),
        "errors": sum(1 for item in findings if item["severity"] == "error"),
        "warnings": sum(1 for item in findings if item["severity"] == "warning"),
    }
    summary["ok"] = summary["errors"] == 0
    return {"summary": summary, "path": str(path), "findings": findings}


def _requires_companion_attrs(target: Any, attr: Any) -> bool:
    """Return whether this proposal target needs reviewed companion captures."""
    candidates = [value for value in (target, attr) if isinstance(value, str)]
    return any(
        candidate.startswith(FLASHINFER_ATTENTION_TARGET_PREFIXES)
        and any(candidate.endswith(wrapper) for wrapper in COMPANION_REQUIRED_WRAPPER_SUFFIXES)
        for candidate in candidates
    )


CANDIDATE_MERGE_META_FIELDS = {"name", "status", "evidence", "review_note"}


def _resolve_proposal_dir(path: Path) -> Path:
    if (path / "candidate_targets.json").exists():
        return path
    nested = path / "proposal"
    if (nested / "candidate_targets.json").exists():
        return nested
    raise FileNotFoundError(f"proposal candidate_targets.json not found under: {path}")


def _candidate_merge_key(item: dict[str, Any]) -> tuple[Any, ...]:
    def key_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return _json_key(value)

    role = item.get("role", "target")
    if role == "warmup":
        return ("warmup", key_value(item.get("module")), key_value(item.get("attr")))

    dispatch = item.get("dispatch")
    dispatch_field = dispatch.get("field") if isinstance(dispatch, dict) else None
    dispatch_value = item.get("page_size") if item.get("page_size") is not None else item.get("dispatch_value")
    module = item.get("module")
    attr = item.get("attr")
    target = item.get("target")
    hook_key = ("module_attr", module, attr) if module and attr else ("target", target)
    return (
        "target",
        tuple(key_value(value) for value in hook_key),
        key_value(item.get("backend")),
        key_value(item.get("definition_source")),
        key_value(item.get("op_type")),
        key_value(item.get("variant")),
        key_value(item.get("definition_name")),
        key_value(dispatch_field),
        key_value(dispatch_value),
    )


def _candidate_core(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in CANDIDATE_MERGE_META_FIELDS
    }


def _unique_json_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    output: list[Any] = []
    for value in values:
        key = _json_key(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _candidate_conflict_fields(items: list[dict[str, Any]]) -> list[str]:
    keys = sorted({key for item in items for key in item if key not in CANDIDATE_MERGE_META_FIELDS})
    fields: list[str] = []
    for key in keys:
        values = [_json_key(item.get(key)) for item in items]
        if len(set(values)) > 1:
            fields.append(key)
    return fields


def _merge_candidate_group(entries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidates = [entry["candidate"] for entry in entries]
    first_core = _candidate_core(candidates[0])
    if any(_candidate_core(candidate) != first_core for candidate in candidates[1:]):
        return None, {
            "kind": "candidate",
            "key": list(_candidate_merge_key(candidates[0])),
            "reason": "conflicting candidate fields",
            "fields": _candidate_conflict_fields(candidates),
            "variants": [
                {
                    "proposal": entry["proposal"],
                    "name": entry["candidate"].get("name"),
                    "candidate": entry["candidate"],
                }
                for entry in entries
            ],
        }

    merged = dict(candidates[0])
    evidence = []
    notes = []
    for entry in entries:
        candidate = entry["candidate"]
        raw_evidence = candidate.get("evidence")
        if isinstance(raw_evidence, list):
            evidence.extend(raw_evidence)
        note = candidate.get("review_note")
        if isinstance(note, str) and note.strip():
            notes.append(f"[{entry['label']}:{candidate.get('name', 'unknown')}] {note.strip()}")
    if evidence:
        merged["evidence"] = _unique_json_values(evidence)
    if notes:
        merged["review_note"] = "\n".join(dict.fromkeys(notes))
    return merged, None


def _draft_payload_without_description(payload: Any) -> Any:
    if not isinstance(payload, dict) or "description" not in payload:
        return payload
    stripped = dict(payload)
    stripped.pop("description", None)
    return stripped


def _copy_proposal_draft_files(
    *,
    kind: str,
    proposal_dirs: list[Path],
    output_dir: Path,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    copied = 0
    conflicts: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    by_relative: dict[Path, list[dict[str, Any]]] = {}
    for proposal_dir in proposal_dirs:
        root = proposal_dir / kind
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            rel = path.relative_to(root)
            by_relative.setdefault(rel, []).append({
                "proposal": str(proposal_dir),
                "path": path,
                "payload": _load_json(path),
            })

    for rel, entries in sorted(by_relative.items(), key=lambda item: str(item[0])):
        payload_keys = {_json_key(entry["payload"]) for entry in entries}
        if len(payload_keys) > 1:
            stripped_keys = {_json_key(_draft_payload_without_description(entry["payload"])) for entry in entries}
            if len(stripped_keys) == 1:
                warnings.append({
                    "kind": kind,
                    "path": str(rel),
                    "reason": "draft file descriptions differ",
                    "variants": [
                        {
                            "proposal": entry["proposal"],
                            "path": str(entry["path"]),
                        }
                        for entry in entries
                    ],
                })
                dst = output_dir / kind / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entries[0]["path"], dst)
                copied += 1
                continue
            conflicts.append({
                "kind": kind,
                "path": str(rel),
                "reason": "conflicting draft file payloads",
                "variants": [
                    {
                        "proposal": entry["proposal"],
                        "path": str(entry["path"]),
                    }
                    for entry in entries
                ],
            })
            continue
        dst = output_dir / kind / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entries[0]["path"], dst)
        copied += 1
    return copied, conflicts, warnings


def _merge_review_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Merged Proposal Review",
        "",
        f"- input proposals: {summary['proposals']}",
        f"- input candidates: {summary['input_candidates']}",
        f"- merged candidates: {summary['merged_candidates']}",
        f"- conflicts: {summary['conflicts']}",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- {path}" for path in report["inputs"])
    lines.extend(["", "## Conflicts", ""])
    if not report["conflicts"]:
        lines.append("- none")
    else:
        for item in report["conflicts"]:
            name = item.get("path") or item.get("key") or "unknown"
            fields = item.get("fields")
            detail = f"; fields={fields}" if fields else ""
            lines.append(f"- {item.get('kind', 'unknown')}: {name} ({item.get('reason', 'conflict')}{detail})")
    warnings = report.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings", ""])
        for item in warnings:
            name = item.get("path") or item.get("key") or "unknown"
            lines.append(f"- {item.get('kind', 'unknown')}: {name} ({item.get('reason', 'warning')})")
    lines.extend([
        "",
        "## Human Action",
        "",
        "Review conflicts before promoting anything into config/. This merged proposal is review-only and does not approve targets.",
        "",
    ])
    return "\n".join(lines)


def merge_proposals(*, proposal_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Merge multiple review-only proposal bundles into a union proposal.

    The merge is intentionally conservative: identical candidates are deduped,
    evidence is unioned, and conflicting candidate/definition/hints payloads are
    reported for human review instead of being auto-resolved.
    """
    if len(proposal_dirs) < 2:
        raise ValueError("merge-proposals requires at least two proposal dirs")
    resolved_dirs = [_resolve_proposal_dir(path) for path in proposal_dirs]
    output_resolved = output_dir.resolve()
    for proposal_dir in resolved_dirs:
        if output_resolved == proposal_dir.resolve():
            raise ValueError("merge-proposals output-dir must be distinct from input proposal dirs")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir / "definitions", ignore_errors=True)
    shutil.rmtree(output_dir / "definition_hints", ignore_errors=True)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    input_candidates = 0
    for index, proposal_dir in enumerate(resolved_dirs, start=1):
        raw = _load_json(proposal_dir / "candidate_targets.json")
        if not isinstance(raw, list):
            raise ValueError(f"candidate_targets.json must be a list: {proposal_dir}")
        label = proposal_dir.parent.name if proposal_dir.name == "proposal" else proposal_dir.name
        for item_index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"candidate #{item_index} must be an object: {proposal_dir}")
            input_candidates += 1
            entry = {
                "proposal": str(proposal_dir),
                "label": f"{label or f'proposal{index}'}",
                "candidate": item,
            }
            groups.setdefault(_candidate_merge_key(item), []).append(entry)

    merged_candidates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for entries in groups.values():
        merged, conflict = _merge_candidate_group(entries)
        if conflict is not None:
            conflicts.append(conflict)
            continue
        if merged is not None:
            merged_candidates.append(merged)
    merged_candidates.sort(key=lambda item: str(item.get("name") or _candidate_merge_key(item)))

    definitions_copied, definition_conflicts, definition_warnings = _copy_proposal_draft_files(
        kind="definitions",
        proposal_dirs=resolved_dirs,
        output_dir=output_dir,
    )
    hints_copied, hint_conflicts, hint_warnings = _copy_proposal_draft_files(
        kind="definition_hints",
        proposal_dirs=resolved_dirs,
        output_dir=output_dir,
    )
    conflicts.extend(definition_conflicts)
    conflicts.extend(hint_conflicts)
    warnings = definition_warnings + hint_warnings

    _write_json(output_dir / "candidate_targets.json", merged_candidates)
    (output_dir / "architecture.md").write_text(
        "# Merged Proposal\n\n"
        "This proposal was generated by `tools.proposal_tools merge-proposals`.\n"
        "Use `merge_review.md` before promoting anything into config/.\n",
        encoding="utf-8",
    )
    report = {
        "summary": {
            "ok": not conflicts,
            "proposals": len(resolved_dirs),
            "input_candidates": input_candidates,
            "merged_candidates": len(merged_candidates),
            "candidate_conflicts": sum(1 for item in conflicts if item.get("kind") == "candidate"),
            "draft_file_conflicts": sum(1 for item in conflicts if item.get("kind") in {"definitions", "definition_hints"}),
            "conflicts": len(conflicts),
            "definitions_copied": definitions_copied,
            "definition_hints_copied": hints_copied,
            "warnings": len(warnings),
        },
        "inputs": [str(path) for path in resolved_dirs],
        "output_dir": str(output_dir),
        "conflicts": conflicts,
        "warnings": warnings,
    }
    _write_json(output_dir / "merge_report.json", report)
    (output_dir / "merge_review.md").write_text(_merge_review_markdown(report), encoding="utf-8")
    (output_dir / "review_checklist.md").write_text(
        "# Review Checklist\n\n"
        "- Read merge_review.md.\n"
        "- Resolve every conflict before promoting candidates into config/.\n"
        "- Run check-proposal on the merged proposal after manual conflict resolution.\n",
        encoding="utf-8",
    )
    return report


def _resolve_dotted_object(target: str) -> tuple[Any | None, str | None]:
    """Resolve a dotted Python object without importing guessed parent modules."""
    parts = target.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        return None, "target is not a valid dotted path"

    last_error: Exception | None = None
    for split_at in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_at])
        attr_parts = parts[split_at:]
        try:
            obj = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exact import errors vary by install
            last_error = exc
            continue
        try:
            for attr in attr_parts:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            return None, f"attribute not found: {'.'.join(attr_parts)} ({exc})"
        return obj, None

    detail = f": {type(last_error).__name__}: {last_error}" if last_error else ""
    return None, f"module import failed{detail}"


def _decorator_uses_flashinfer_api(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        candidate = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(candidate, ast.Name) and candidate.id == "flashinfer_api":
            return True
        if isinstance(candidate, ast.Attribute) and candidate.attr == "flashinfer_api":
            return True
    return False


def _trace_template_metadata(flashinfer_root: Path | None, trace_name: str | None) -> dict[str, str | None]:
    if flashinfer_root is None or not trace_name:
        return {"trace_name": trace_name, "trace_op_type": None, "trace_name_prefix": None}
    for path in sorted(flashinfer_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == trace_name for target in node.targets):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (
                isinstance(func, ast.Name) and func.id == "TraceTemplate"
                or isinstance(func, ast.Attribute) and func.attr == "TraceTemplate"
            ):
                continue
            metadata: dict[str, str | None] = {
                "trace_name": trace_name,
                "trace_op_type": None,
                "trace_name_prefix": None,
            }
            for keyword in call.keywords:
                if keyword.arg in {"op_type", "name_prefix"} and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    if isinstance(value, str):
                        metadata[f"trace_{keyword.arg}"] = value
            return metadata
    return {"trace_name": trace_name, "trace_op_type": None, "trace_name_prefix": None}


def _trace_name_from_decorator(node: ast.FunctionDef) -> str | None:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        candidate = decorator.func
        if not (
            isinstance(candidate, ast.Name) and candidate.id == "flashinfer_api"
            or isinstance(candidate, ast.Attribute) and candidate.attr == "flashinfer_api"
        ):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "trace" and isinstance(keyword.value, ast.Name):
                return keyword.value.id
    return None


def _function_fitrace_metadata(
    module_path: Path,
    attr_parts: list[str],
    flashinfer_root: Path | None,
) -> dict[str, Any]:
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"source parse failed: {type(exc).__name__}: {exc}"}

    scope: list[ast.stmt] = list(tree.body)
    node: ast.AST | None = None
    for index, part in enumerate(attr_parts):
        node = None
        matches = [
            item
            for item in scope
            if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == part
        ]
        if matches and index == len(attr_parts) - 1:
            function_matches = [item for item in matches if isinstance(item, ast.FunctionDef)]
            if function_matches:
                for item in function_matches:
                    if _decorator_uses_flashinfer_api(item):
                        return {
                            "ok": True,
                            "error": None,
                            **_trace_template_metadata(flashinfer_root, _trace_name_from_decorator(item)),
                        }
                return {"ok": False, "error": "source function exists but has no @flashinfer_api decorator"}
        if matches:
            node = matches[0]
        if node is None:
            return {"ok": False, "error": f"source attribute not found: {'.'.join(attr_parts)}"}
        if isinstance(node, ast.ClassDef):
            scope = list(node.body)
        elif isinstance(node, ast.FunctionDef):
            scope = []
        else:
            return {"ok": False, "error": f"unsupported source node for {part}"}

    if isinstance(node, ast.FunctionDef) and _decorator_uses_flashinfer_api(node):
        return {
            "ok": True,
            "error": None,
            **_trace_template_metadata(flashinfer_root, _trace_name_from_decorator(node)),
        }
    if isinstance(node, ast.FunctionDef):
        return {"ok": False, "error": "source function exists but has no @flashinfer_api decorator"}
    return {"ok": False, "error": "source target is not a function"}


def _suggest_fitrace_attr_parts(module_path: Path, attr_parts: list[str]) -> list[str] | None:
    """Suggest a sibling decorated function when a target points at a wrapper alias."""
    if len(attr_parts) < 2:
        return None
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    scope: list[ast.stmt] = list(tree.body)
    for part in attr_parts[:-1]:
        matches = [
            item
            for item in scope
            if isinstance(item, ast.ClassDef) and item.name == part
        ]
        if not matches:
            return None
        scope = list(matches[0].body)

    decorated_methods = [
        item.name
        for item in scope
        if isinstance(item, ast.FunctionDef) and _decorator_uses_flashinfer_api(item)
    ]
    if "run" in decorated_methods and attr_parts[-1] != "run":
        return [*attr_parts[:-1], "run"]
    if decorated_methods:
        return [*attr_parts[:-1], decorated_methods[0]]
    return None


def _find_flashinfer_source_target(
    target: str,
    flashinfer_root: Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_checked": False,
        "source_path": None,
        "source_fitrace_ok": False,
        "source_error": None,
        "suggested_target": None,
    }
    if not target.startswith("flashinfer."):
        result["source_error"] = "not a flashinfer target"
        return result

    roots = [flashinfer_root] if flashinfer_root is not None else list(DEFAULT_FLASHINFER_ROOTS)
    parts = target.split(".")
    for root in roots:
        if root is None:
            continue
        root = root.resolve() if root.exists() else root
        for split_at in range(len(parts) - 1, 1, -1):
            module_parts = parts[1:split_at]
            attr_parts = parts[split_at:]
            candidates = [
                root.joinpath(*module_parts).with_suffix(".py"),
                root.joinpath(*module_parts, "__init__.py"),
            ]
            for module_path in candidates:
                if not module_path.exists():
                    continue
                metadata = _function_fitrace_metadata(module_path, attr_parts, root)
                ok = bool(metadata["ok"])
                error = metadata["error"]
                suggested_attr_parts = None if ok else _suggest_fitrace_attr_parts(module_path, attr_parts)
                suggested_target = None
                if suggested_attr_parts is not None:
                    suggested_target = ".".join([*parts[:split_at], *suggested_attr_parts])
                result.update({
                    "source_checked": True,
                    "source_path": str(module_path),
                    "source_fitrace_ok": ok,
                    "source_error": error,
                    "suggested_target": suggested_target,
                    "trace_name": metadata.get("trace_name"),
                    "trace_op_type": metadata.get("trace_op_type"),
                    "trace_name_prefix": metadata.get("trace_name_prefix"),
                })
                return result

    result["source_error"] = "source file not found"
    return result


def _expected_fitrace_preview(hf_config: dict[str, Any]) -> dict[str, Any]:
    """Return non-authoritative model facts useful for reviewing fitrace candidates."""
    heads = hf_config.get("num_attention_heads")
    kv_heads = hf_config.get("num_key_value_heads", heads)
    hidden_size = hf_config.get("hidden_size")
    head_dim = hf_config.get("head_dim")
    if not isinstance(head_dim, int) and isinstance(hidden_size, int) and isinstance(heads, int) and heads > 0:
        head_dim = hidden_size // heads
    return {
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": hf_config.get("vocab_size"),
        "note": "diagnostic only; final definition name/schema must come from fitrace dump",
    }


def _sglang_config_compat_engine_kwargs(hf_config: dict[str, Any]) -> dict[str, str]:
    """Return reviewed Engine kwargs needed before SGLang can load this config."""
    rope_scaling = hf_config.get("rope_scaling")
    if (
        hf_config.get("model_type") == "phi3"
        and isinstance(rope_scaling, dict)
        and rope_scaling.get("type") == "longrope"
        and "rope_theta" in hf_config
    ):
        cleaned = dict(hf_config)
        cleaned.pop("rope_theta", None)
        cleaned_rope = {
            key: value
            for key, value in rope_scaling.items()
            if key in {"type", "short_factor", "long_factor"}
        }
        if set(cleaned_rope) == {"type", "short_factor", "long_factor"}:
            cleaned["rope_scaling"] = cleaned_rope
            return {
                "decrypted_config_json": json.dumps(cleaned, separators=(",", ":")),
            }
    return {}


def _check_run_config(
    *,
    proposal_dir: Path,
    hf_config: dict[str, Any],
) -> dict[str, Any]:
    """Check proposal-time run_config requirements that prevent known startup failures."""
    findings: list[dict[str, str]] = []
    path = proposal_dir.parent / "config" / "run_config.json"
    required_engine_kwargs = _sglang_config_compat_engine_kwargs(hf_config)
    if not path.exists():
        if required_engine_kwargs:
            findings.append({
                "severity": "error",
                "name": "run_config",
                "reason": (
                    "missing proposed runtime config required for model startup compatibility; "
                    f"run_config.engine_kwargs must include {sorted(required_engine_kwargs)}: {path}"
                ),
            })
    else:
        data = _load_json(path)
        if not isinstance(data, dict):
            findings.append({
                "severity": "error",
                "name": "run_config",
                "reason": f"run_config must be a JSON object: {path}",
            })
        else:
            engine_kwargs = data.get("engine_kwargs")
            if engine_kwargs is None:
                engine_kwargs = {}
            if not isinstance(engine_kwargs, dict):
                findings.append({
                    "severity": "error",
                    "name": "run_config",
                    "reason": "run_config.engine_kwargs must be an object",
                })
            else:
                for key, expected in required_engine_kwargs.items():
                    actual = engine_kwargs.get(key)
                    if actual != expected:
                        findings.append({
                            "severity": "error",
                            "name": "run_config",
                            "reason": (
                                f"run_config.engine_kwargs.{key} must contain the sanitized "
                                "HF config override required before SGLang loads this model"
                            ),
                        })
    errors = sum(1 for item in findings if item["severity"] == "error")
    warnings = sum(1 for item in findings if item["severity"] == "warning")
    return {
        "summary": {
            "ok": errors == 0,
            "errors": errors,
            "warnings": warnings,
            "required_engine_kwargs": sorted(required_engine_kwargs),
        },
        "run_config_path": str(path),
        "findings": findings,
    }


def _evaluate_fitrace_targets(
    *,
    candidates_path: Path,
    hf_config_path: Path,
    flashinfer_root: Path | None = None,
) -> dict[str, Any]:
    """Check collect candidates against fitrace capability, not old definitions."""
    candidates = _load_json(candidates_path)
    hf_config = _load_json(hf_config_path)
    if not isinstance(candidates, list):
        raise ValueError(f"candidate targets must be a list: {candidates_path}")
    if not isinstance(hf_config, dict):
        raise ValueError(f"HF config must be a JSON object: {hf_config_path}")

    collect_candidates = [
        item
        for item in candidates
        if isinstance(item, dict) and item.get("collect") is True and item.get("backend") == "flashinfer"
    ]

    target_results: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    for index, item in enumerate(collect_candidates):
        name = str(item.get("name") or item.get("definition_name") or f"#{index}")
        target = item.get("target")
        result = {
            "name": name,
            "definition_name": item.get("definition_name"),
            "target": target,
            "backend": item.get("backend"),
            "op_type": item.get("op_type"),
            "variant": item.get("variant"),
            "definition_source": item.get("definition_source"),
            "import_ok": False,
            "fitrace_ok": False,
            "source_checked": False,
            "source_fitrace_ok": False,
            "suggested_target": None,
            "trace_name": None,
            "trace_op_type": None,
            "trace_name_prefix": None,
            "resolved_type": None,
        }
        if not isinstance(target, str) or not target:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": "collect candidate has no target",
            })
            target_results.append(result)
            continue

        obj, error = _resolve_dotted_object(target)
        source_result = _find_flashinfer_source_target(target, flashinfer_root)
        result.update(source_result)
        has_fi_trace = False
        if error is None:
            result["import_ok"] = True
            result["resolved_type"] = type(obj).__name__
            has_fi_trace = callable(getattr(obj, "fi_trace", None))
        else:
            result["import_error"] = error

        result["fitrace_ok"] = has_fi_trace or bool(source_result["source_fitrace_ok"])
        if not result["fitrace_ok"]:
            reason = (
                "target is importable but has no callable .fi_trace"
                if error is None
                else f"target cannot be imported and source check failed: {error}; {source_result['source_error']}"
            )
            if source_result.get("suggested_target"):
                reason = f"{reason}; suggested fitrace target: {source_result['suggested_target']}"
            findings.append({
                "severity": "error",
                "name": name,
                "reason": f"{reason}; final definition cannot come from fitrace",
            })

        if item.get("backend") != "flashinfer":
            findings.append({
                "severity": "warning",
                "name": name,
                "reason": "collect candidate should declare backend=flashinfer for fitrace evaluation",
            })
        if item.get("definition_source") != "fitrace":
            findings.append({
                "severity": "warning",
                "name": name,
                "reason": "collect candidate should declare definition_source=fitrace when the target is fitrace-backed",
            })
        trace_op_type = source_result.get("trace_op_type")
        if isinstance(trace_op_type, str) and item.get("op_type") != trace_op_type:
            findings.append({
                "severity": "error",
                "name": name,
                "reason": (
                    f"candidate op_type {item.get('op_type')!r} does not match "
                    f"FlashInfer trace template op_type {trace_op_type!r}"
                ),
            })

        definition_name = item.get("definition_name")
        if isinstance(definition_name, str) and definition_name:
            findings.append({
                "severity": "warning",
                "name": name,
                "reason": "definition_name is only a preview for fitrace-backed targets; final name/schema must come from fitrace dump",
            })
        target_results.append(result)

    errors = sum(1 for item in findings if item["severity"] == "error")
    warnings = sum(1 for item in findings if item["severity"] == "warning")
    return {
        "summary": {
            "collect_candidates": len(collect_candidates),
            "importable_targets": sum(1 for item in target_results if item["import_ok"]),
            "fitrace_targets": sum(1 for item in target_results if item["fitrace_ok"]),
            "errors": errors,
            "warnings": warnings,
            "ok": errors == 0,
        },
        "candidates_path": str(candidates_path),
        "hf_config_path": str(hf_config_path),
        "flashinfer_root": str(flashinfer_root) if flashinfer_root is not None else None,
        "fitrace_preview": _expected_fitrace_preview(hf_config),
        "targets": target_results,
        "findings": findings,
    }


def _definition_draft_path(proposal_dir: Path, op_type: str, definition_name: str) -> Path:
    return proposal_dir / "definitions" / op_type / f"{definition_name}.json"


def _definition_hint_path(proposal_dir: Path, op_type: str, definition_name: str) -> Path:
    return proposal_dir / "definition_hints" / op_type / f"{definition_name}.json"


def _check_definition_object(
    *,
    path: Path,
    name: str,
    op_type: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    try:
        data = _load_json(path)
    except Exception as exc:  # noqa: BLE001 - report agent-facing diagnostics
        return [{"severity": "error", "name": name, "reason": f"definition draft is not readable JSON: {exc}"}]
    if not isinstance(data, dict):
        return [{"severity": "error", "name": name, "reason": "definition draft must be a JSON object"}]
    missing = sorted(field for field in DEFINITION_REQUIRED_FIELDS if field not in data)
    if missing:
        findings.append({"severity": "error", "name": name, "reason": f"definition draft missing fields: {missing}"})
    if data.get("name") != name:
        findings.append({"severity": "error", "name": name, "reason": f"definition draft name mismatch: {data.get('name')!r}"})
    if data.get("op_type") != op_type:
        findings.append({"severity": "error", "name": name, "reason": f"definition draft op_type mismatch: {data.get('op_type')!r}"})
    if not isinstance(data.get("axes"), dict):
        findings.append({"severity": "error", "name": name, "reason": "definition draft axes must be an object"})
    inputs = data.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        findings.append({"severity": "error", "name": name, "reason": "definition draft inputs must be a non-empty object"})
    else:
        for input_name, input_spec in inputs.items():
            if not isinstance(input_name, str) or not input_name:
                findings.append({"severity": "error", "name": name, "reason": "definition draft input name must be a non-empty string"})
                continue
            if not isinstance(input_spec, dict):
                findings.append({"severity": "error", "name": name, "reason": f"definition draft input {input_name} must be an object"})
                continue
            if "shape" not in input_spec or "dtype" not in input_spec:
                findings.append({"severity": "error", "name": name, "reason": f"definition draft input {input_name} must declare shape and dtype"})
            elif input_spec.get("shape") == []:
                findings.append({
                    "severity": "error",
                    "name": name,
                    "reason": f"definition draft input {input_name} uses shape []; use null for scalar inputs",
                })
    if not isinstance(data.get("outputs"), dict):
        findings.append({"severity": "error", "name": name, "reason": "definition draft outputs must be an object"})
    reference = data.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        findings.append({"severity": "error", "name": name, "reason": "definition draft reference must be a non-empty string"})
    elif not _reference_has_top_level_run(reference):
        findings.append({"severity": "error", "name": name, "reason": "definition draft reference must define top-level run(...)"})
    return findings


def _reference_has_top_level_run(reference: str) -> bool:
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == "run" for node in tree.body)


def _check_hint_object(
    *,
    path: Path,
    name: str,
    op_type: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    try:
        data = _load_json(path)
    except Exception as exc:  # noqa: BLE001 - report agent-facing diagnostics
        return [{"severity": "error", "name": name, "reason": f"definition hints draft is not readable JSON: {exc}"}]
    if not isinstance(data, dict):
        return [{"severity": "error", "name": name, "reason": "definition hints draft must be a JSON object"}]
    missing = sorted(field for field in HINT_REQUIRED_FIELDS if field not in data)
    if missing:
        findings.append({"severity": "error", "name": name, "reason": f"definition hints draft missing fields: {missing}"})
    if data.get("definition_name") != name:
        findings.append({"severity": "error", "name": name, "reason": f"definition hints name mismatch: {data.get('definition_name')!r}"})
    if data.get("op_type") != op_type:
        findings.append({"severity": "error", "name": name, "reason": f"definition hints op_type mismatch: {data.get('op_type')!r}"})
    if type(data.get("schema_version")) is not int:
        findings.append({"severity": "error", "name": name, "reason": "definition hints schema_version must be an integer"})
    inputs = data.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        findings.append({"severity": "error", "name": name, "reason": "definition hints inputs must be a non-empty object"})
    return findings


def _check_non_fitrace_definition_drafts(*, proposal_dir: Path, candidates_path: Path) -> dict[str, Any]:
    """Check review-only definition/hints drafts for non-fitrace agent targets."""
    candidates = _load_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError(f"candidate targets must be a list: {candidates_path}")

    checked: list[dict[str, str]] = []
    findings: list[dict[str, str]] = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        if item.get("role", "target") != "target":
            continue
        if item.get("backend") == "flashinfer" or item.get("definition_source") != "agent":
            continue
        candidate_name = str(item.get("name") or f"#{index}")
        definition_name = item.get("definition_name")
        op_type = item.get("op_type")
        if not isinstance(definition_name, str) or not definition_name:
            findings.append({
                "severity": "error",
                "name": candidate_name,
                "reason": "non-fitrace agent target must declare definition_name for draft checking",
            })
            continue
        if not isinstance(op_type, str) or not op_type:
            findings.append({
                "severity": "error",
                "name": candidate_name,
                "reason": "non-fitrace agent target must declare op_type for draft checking",
            })
            continue
        definition_path = _definition_draft_path(proposal_dir, op_type, definition_name)
        hints_path = _definition_hint_path(proposal_dir, op_type, definition_name)
        checked.append({
            "name": candidate_name,
            "definition_name": definition_name,
            "op_type": op_type,
            "definition_path": str(definition_path),
            "hints_path": str(hints_path),
        })
        if not definition_path.exists():
            findings.append({
                "severity": "error",
                "name": candidate_name,
                "reason": f"missing review-only definition draft: {definition_path}",
            })
        else:
            findings.extend(
                {"severity": item["severity"], "name": candidate_name, "reason": item["reason"]}
                for item in _check_definition_object(path=definition_path, name=definition_name, op_type=op_type)
            )
        if not hints_path.exists():
            findings.append({
                "severity": "error",
                "name": candidate_name,
                "reason": f"missing review-only definition hints draft: {hints_path}",
            })
        else:
            findings.extend(
                {"severity": item["severity"], "name": candidate_name, "reason": item["reason"]}
                for item in _check_hint_object(path=hints_path, name=definition_name, op_type=op_type)
            )

    errors = sum(1 for item in findings if item["severity"] == "error")
    warnings = sum(1 for item in findings if item["severity"] == "warning")
    return {
        "summary": {
            "draft_targets": len(checked),
            "errors": errors,
            "warnings": warnings,
            "ok": errors == 0,
        },
        "proposal_dir": str(proposal_dir),
        "checked": checked,
        "findings": findings,
    }


def _check_merge_report(proposal_dir: Path) -> dict[str, Any]:
    path = proposal_dir / "merge_report.json"
    if not path.exists():
        return {"summary": {"conflicts": 0, "errors": 0, "ok": True}, "findings": []}
    report = _load_json(path)
    if not isinstance(report, dict):
        return {
            "summary": {"conflicts": 1, "errors": 1, "ok": False},
            "findings": [{
                "severity": "error",
                "name": str(path),
                "reason": "merge_report.json must be a JSON object",
            }],
        }
    conflicts = report.get("conflicts")
    findings: list[dict[str, str]] = []
    if isinstance(conflicts, list):
        for index, item in enumerate(conflicts):
            if not isinstance(item, dict):
                findings.append({
                    "severity": "error",
                    "name": f"merge_conflict_{index}",
                    "reason": "unresolved merge conflict",
                })
                continue
            name = str(item.get("path") or item.get("key") or f"merge_conflict_{index}")
            reason = str(item.get("reason") or "unresolved merge conflict")
            fields = item.get("fields")
            if isinstance(fields, list) and fields:
                reason = f"{reason}; fields={fields}"
            findings.append({"severity": "error", "name": name, "reason": reason})
    errors = len(findings)
    return {
        "summary": {
            "conflicts": len(conflicts) if isinstance(conflicts, list) else 0,
            "errors": errors,
            "ok": errors == 0,
        },
        "path": str(path),
        "findings": findings,
    }


def check_proposal(
    *,
    candidates_path: Path,
    hf_config_path: Path,
    proposal_dir: Path | None = None,
    flashinfer_root: Path | None = None,
) -> dict[str, Any]:
    """Run all proposal checks in one review gate."""
    if proposal_dir is None:
        proposal_dir = candidates_path.parent
    candidate_fields = _check_candidate_fields(candidates_path)
    hf_config = _load_json(hf_config_path)
    if not isinstance(hf_config, dict):
        raise ValueError(f"HF config must be a JSON object: {hf_config_path}")
    run_config = _check_run_config(
        proposal_dir=proposal_dir,
        hf_config=hf_config,
    )
    fitrace_eval = _evaluate_fitrace_targets(
        candidates_path=candidates_path,
        hf_config_path=hf_config_path,
        flashinfer_root=flashinfer_root,
    )
    definition_drafts = _check_non_fitrace_definition_drafts(
        proposal_dir=proposal_dir,
        candidates_path=candidates_path,
    )
    merge_report = _check_merge_report(proposal_dir)
    findings = [
        {"check": "candidate_fields", **item}
        for item in candidate_fields["findings"]
    ]
    findings.extend({"check": "run_config", **item} for item in run_config["findings"])
    findings.extend({"check": "fitrace_target", **item} for item in fitrace_eval["findings"])
    findings.extend({"check": "definition_draft", **item} for item in definition_drafts["findings"])
    findings.extend({"check": "merge_report", **item} for item in merge_report["findings"])
    errors = sum(1 for item in findings if item["severity"] == "error")
    warnings = sum(1 for item in findings if item["severity"] == "warning")
    return {
        "summary": {
            "entries": candidate_fields["summary"]["entries"],
            "collect_candidates": fitrace_eval["summary"]["collect_candidates"],
            "importable_targets": fitrace_eval["summary"]["importable_targets"],
            "fitrace_targets": fitrace_eval["summary"]["fitrace_targets"],
            "definition_draft_targets": definition_drafts["summary"]["draft_targets"],
            "merge_conflicts": merge_report["summary"]["conflicts"],
            "errors": errors,
            "warnings": warnings,
            "ok": errors == 0,
        },
        "candidates_path": str(candidates_path),
        "hf_config_path": str(hf_config_path),
        "flashinfer_root": str(flashinfer_root) if flashinfer_root is not None else None,
        "candidate_fields": candidate_fields,
        "run_config": run_config,
        "fitrace_eval": fitrace_eval,
        "definition_drafts": definition_drafts,
        "merge_report": merge_report,
        "findings": findings,
    }


def _proposal_feedback_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    findings = report.get("findings", [])
    status = "PASS" if summary["ok"] else "FIX_REQUIRED"
    lines = [
        "# Agent Proposal Feedback",
        "",
        f"- status: {status}",
        f"- entries: {summary['entries']}",
        f"- collect candidates: {summary['collect_candidates']}",
        f"- fitrace targets: {summary['fitrace_targets']}",
        f"- non-fitrace definition drafts: {summary['definition_draft_targets']}",
        f"- merge conflicts: {summary.get('merge_conflicts', 0)}",
        f"- errors: {summary['errors']}",
        f"- warnings: {summary['warnings']}",
        "",
    ]
    if summary["ok"]:
        lines.extend([
            "## Next Action",
            "",
            "Proposal checks passed. Stop revising and hand the proposal to human review.",
            "",
        ])
        return "\n".join(lines)

    lines.extend([
        "## Next Agent Action",
        "",
        "Revise only the proposal bundle. Do not edit config/approved_targets.json, do not run Modal, and do not approve candidates automatically.",
        "",
        "## Findings",
        "",
    ])
    for item in findings:
        severity = item.get("severity", "unknown")
        check = item.get("check", "unknown")
        name = item.get("name", "unknown")
        reason = item.get("reason", "unknown")
        lines.append(f"- {severity} [{check}] {name}: {reason}")

    suggestions = [
        item
        for item in report.get("fitrace_eval", {}).get("targets", [])
        if isinstance(item, dict) and item.get("suggested_target")
    ]
    if suggestions:
        lines.extend(["", "## Suggested Fitrace Targets", ""])
        for item in suggestions:
            lines.append(f"- {item.get('name')}: {item['suggested_target']}")
    lines.append("")
    return "\n".join(lines)


def run_agent_loop(
    *,
    proposal_dir: Path,
    hf_config_path: Path,
    candidates_path: Path | None = None,
    flashinfer_root: Path | None = None,
) -> dict[str, Any]:
    """Run one deterministic proposal-check loop and write agent feedback.

    The loop intentionally does not call an LLM. The external agent reads the
    generated feedback, edits the proposal bundle, and invokes this command
    again until the check passes.
    """
    proposal_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidates_path or proposal_dir / "candidate_targets.json"
    check_report = check_proposal(
        proposal_dir=proposal_dir,
        candidates_path=candidates,
        hf_config_path=hf_config_path,
        flashinfer_root=flashinfer_root,
    )
    check_path = proposal_dir / "proposal_check.json"
    feedback_path = proposal_dir / "agent_feedback.md"
    loop_path = proposal_dir / "agent_loop.json"
    _write_json(check_path, check_report)
    feedback_path.write_text(_proposal_feedback_markdown(check_report), encoding="utf-8")
    result = {
        "summary": {
            "ok": check_report["summary"]["ok"],
            "ready_for_human_review": check_report["summary"]["ok"],
            "errors": check_report["summary"]["errors"],
            "warnings": check_report["summary"]["warnings"],
        },
        "proposal_dir": str(proposal_dir),
        "candidates_path": str(candidates),
        "hf_config_path": str(hf_config_path),
        "flashinfer_root": str(flashinfer_root) if flashinfer_root is not None else None,
        "outputs": {
            "check_report": str(check_path),
            "feedback": str(feedback_path),
            "loop_report": str(loop_path),
        },
    }
    _write_json(loop_path, result)
    return result


def _resolve_run_dir(run: Path) -> Path:
    return run if run.exists() else Path("runs") / run


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _agent_suffix(index: int) -> str:
    if index < len(string.ascii_lowercase):
        return string.ascii_lowercase[index]
    return str(index + 1)


def _agent_run_dir(run_prefix: Path, *, count: int, index: int) -> Path:
    if count == 1:
        return run_prefix
    return run_prefix.with_name(f"{run_prefix.name}_agent_{_agent_suffix(index)}")


def first_pass_prompt_markdown(
    *,
    model_name: str,
    run_dir: Path,
    hf_config_path: Path,
    sglang_root: Path,
    flashinfer_root: Path,
    cookbook_root: Path,
    sglang_model_hints: list[str],
) -> str:
    """Return the fixed first-pass prompt for one review-only agent run."""
    proposal_dir = run_dir / "proposal"
    hints = "\n".join(f"  - {item}" for item in sglang_model_hints) if sglang_model_hints else "  - none provided"
    runtime_guidance: list[str] = []
    hf_config = _load_json(hf_config_path) if hf_config_path.exists() else {}
    if isinstance(hf_config, dict):
        compat_kwargs = _sglang_config_compat_engine_kwargs(hf_config)
        if compat_kwargs:
            runtime_guidance.extend([
                "Runtime config compatibility requirement:",
                "",
                "```json",
                json.dumps({"engine_kwargs": compat_kwargs}, indent=2, ensure_ascii=False),
                "```",
                "",
                "Include these `engine_kwargs` in `config/run_config.json`. They are required before SGLang can load this HF config.",
                "",
            ])
    return "\n".join([
        f"# {model_name} First-Pass Proposal",
        "",
        "Use the `onboard-model-proposal` skill to generate a review-only proposal for:",
        "",
        "```text",
        model_name,
        "```",
        "",
        "Working directory:",
        "",
        "```text",
        ".",
        "```",
        "",
        "Inputs:",
        "",
        "```text",
        f"HF config: {_display_path(hf_config_path)}",
        f"SGLang source root: {_display_path(sglang_root)}",
        "SGLang model implementation hints:",
        hints,
        f"FlashInfer source root: {_display_path(flashinfer_root)}",
        f"sgl-cookbook root: {_display_path(cookbook_root)}",
        "diagnostics: omit; first-pass",
        f"run dir: {_display_path(run_dir)}",
        "```",
        "",
        "Outputs:",
        "",
        "```text",
        f"{_display_path(proposal_dir / 'architecture.md')}",
        f"{_display_path(proposal_dir / 'candidate_targets.json')}",
        f"{_display_path(proposal_dir / 'review_checklist.md')}",
        f"{_display_path(proposal_dir / 'definitions/...')} only for non-FI definition_source=agent drafts",
        f"{_display_path(proposal_dir / 'definition_hints/...')} only for non-FI definition_source=agent drafts",
        f"{_display_path(run_dir / 'config' / 'run_config.json')}",
        "```",
        "",
        *runtime_guidance,
        "Strictly follow `.agents/skills/onboard-model-proposal/SKILL.md`.",
        "",
        "Do not apply or approve anything. Do not write `config/approved_targets.json`.",
        "Do not write official `output/definitions`, `output/workloads`, or `output/blob`.",
        "Do not run Modal, collect, validate, or commit.",
        "",
        "After writing the proposal, run:",
        "",
        "```bash",
        "python3 -B -m tools.proposal_tools agent-loop \\",
        f"  --proposal-dir {_display_path(proposal_dir)} \\",
        f"  --hf-config {_display_path(hf_config_path)} \\",
        f"  --flashinfer-root {_display_path(flashinfer_root)}",
        "```",
        "",
        "If `agent_feedback.md` reports `FIX_REQUIRED`, revise only the review-only",
        "proposal bundle and repeat the same deterministic check until it prints:",
        "",
        "```text",
        "ready for human review: True",
        "```",
        "",
    ])


def spawn_agents(
    *,
    model_name: str,
    run_prefix: Path,
    hf_config_path: Path,
    sglang_root: Path,
    flashinfer_root: Path,
    cookbook_root: Path,
    sglang_model_hints: list[str],
    count: int = 1,
    agent_command: list[str] | None = None,
    agent_env: dict[str, str] | None = None,
    merge_output_dir: Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Generate first-pass prompts and optionally run N external agents."""
    if count < 1:
        raise ValueError("count must be >= 1")
    run_base = _resolve_run_dir(run_prefix)
    prompts: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []

    for index in range(count):
        run_dir = _agent_run_dir(run_base, count=count, index=index)
        proposal_dir = run_dir / "proposal"
        config_dir = run_dir / "config"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        prompt_text = first_pass_prompt_markdown(
            model_name=model_name,
            run_dir=run_dir,
            hf_config_path=hf_config_path,
            sglang_root=sglang_root,
            flashinfer_root=flashinfer_root,
            cookbook_root=cookbook_root,
            sglang_model_hints=sglang_model_hints,
        )
        prompt_path = proposal_dir / "first_pass_prompt.md"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        item = {
            "index": index + 1,
            "run_dir": str(run_dir),
            "proposal_dir": str(proposal_dir),
            "prompt": str(prompt_path),
        }
        prompts.append(item)
        if progress:
            print(f"[spawn-agents] prompt {index + 1}/{count}: {prompt_path}", flush=True)

        if agent_command:
            stdout_path = proposal_dir / "agent_stdout.log"
            stderr_path = proposal_dir / "agent_stderr.log"
            stdout_file = stdout_path.open("w", encoding="utf-8")
            stderr_file = stderr_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                agent_command,
                text=True,
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                env=agent_env,
            )
            if progress:
                print(
                    f"[spawn-agents] started agent {index + 1}/{count}: "
                    f"pid={proc.pid} stdout={stdout_path} stderr={stderr_path}",
                    flush=True,
                )
            processes.append({
                "process": proc,
                "stdin": prompt_text,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "prompt": item,
            })

    for proc_info in processes:
        proc_info["process"].stdin.write(proc_info["stdin"])
        proc_info["process"].stdin.close()

    agent_results: list[dict[str, Any]] = []
    pending = list(processes)
    last_progress = 0.0
    while pending:
        remaining: list[dict[str, Any]] = []
        for proc_info in pending:
            proc = proc_info["process"]
            returncode = proc.poll()
            if returncode is None:
                remaining.append(proc_info)
                continue
            proc_info["stdout_file"].close()
            proc_info["stderr_file"].close()
            prompt = proc_info["prompt"]
            result = {
                "run_dir": prompt["run_dir"],
                "proposal_dir": prompt["proposal_dir"],
                "returncode": returncode,
                "stdout": str(proc_info["stdout_path"]),
                "stderr": str(proc_info["stderr_path"]),
            }
            agent_results.append(result)
            if progress:
                status = "ok" if returncode == 0 else "failed"
                print(
                    f"[spawn-agents] finished agent {prompt['index']}/{count}: "
                    f"{status} returncode={returncode} proposal={prompt['proposal_dir']}",
                    flush=True,
                )
        pending = remaining
        if pending:
            now = time.monotonic()
            if progress and now - last_progress >= 30:
                running = ", ".join(
                    f"{item['prompt']['index']}(pid={item['process'].pid})"
                    for item in pending
                )
                print(f"[spawn-agents] still running: {running}", flush=True)
                last_progress = now
            time.sleep(1)

    merge_report: dict[str, Any] | None = None
    if merge_output_dir is not None:
        if progress:
            print(f"[spawn-agents] merging proposals -> {merge_output_dir}", flush=True)
        merge_report = merge_proposals(
            proposal_dirs=[Path(item["proposal_dir"]) for item in prompts],
            output_dir=merge_output_dir,
        )

    result = {
        "summary": {
            "model": model_name,
            "count": count,
            "agents_started": len(processes),
            "agent_failures": sum(1 for item in agent_results if item["returncode"] != 0),
            "merged": merge_report is not None,
            "merge_ok": merge_report["summary"]["ok"] if merge_report is not None else None,
        },
        "run_prefix": str(run_base),
        "prompts": prompts,
        "agent_results": agent_results,
        "merge_report": str(merge_output_dir / "merge_report.json") if merge_output_dir is not None else None,
    }
    report_path = run_base.parent / f"{run_base.name}_spawn_agents.json"
    _write_json(report_path, result)
    result["report"] = str(report_path)
    if progress:
        print(f"[spawn-agents] report: {report_path}", flush=True)
    return result


def _brief(value: Any, *, limit: int = 500) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _append_findings_from_items(
    findings: list[dict[str, Any]],
    *,
    source: str,
    severity: str,
    items: Any,
) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        reason = item.get("reason") or item.get("error") or item
        if item.get("reason") == "sanitize_failed" and isinstance(item.get("sanitize_reject_reasons"), dict):
            reason = f"sanitize_failed: {item['sanitize_reject_reasons']}"
            if isinstance(item.get("sanitize_reject_examples"), dict):
                reason = f"{reason}; examples: {item['sanitize_reject_examples']}"
        findings.append({
            "source": source,
            "severity": str(item.get("severity") or severity),
            "name": str(item.get("name") or item.get("definition_name") or item.get("source_name") or "unknown"),
            "reason": _brief(reason),
        })


def _ignored_definition_reasons(proposal_dir: Path) -> dict[str, str]:
    path = proposal_dir / "ignored_definitions.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"ignored definitions must be a JSON list: {path}")
    ignored: dict[str, str] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"ignored definition #{index} must be an object")
        name = item.get("name")
        reason = item.get("reason")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"ignored definition #{index} has no name")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"ignored definition {name} has no reason")
        ignored[name.strip()] = reason.strip()
    return ignored


def _proposed_definition_candidates(proposal_dir: Path) -> dict[str, str]:
    path = proposal_dir / "candidate_targets.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, list):
        return {}
    proposed: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict) or item.get("collect") is not True:
            continue
        if item.get("status") == "rejected":
            continue
        definition_name = item.get("definition_name")
        if not isinstance(definition_name, str) or not definition_name.strip():
            continue
        candidate_name = item.get("name")
        proposed[definition_name.strip()] = (
            candidate_name.strip()
            if isinstance(candidate_name, str) and candidate_name.strip()
            else definition_name.strip()
        )
    return proposed


def _uncollected_definition_findings(
    *,
    items: Any,
    ignored: dict[str, str],
    proposed: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    findings: list[dict[str, Any]] = []
    ignored_items: list[dict[str, str]] = []
    proposed_items: list[dict[str, str]] = []
    if not isinstance(items, list):
        return findings, ignored_items, proposed_items
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("definition_name") or item.get("source_name") or "unknown")
        reason = _brief(item.get("reason") or item.get("details") or item.get("error") or "uncollected definition")
        candidate_name = proposed.get(name)
        if candidate_name:
            proposed_items.append({
                "name": name,
                "candidate": candidate_name,
            })
            continue
        ignore_reason = ignored.get(name)
        if ignore_reason:
            ignored_items.append({
                "name": name,
                "reason": _brief(ignore_reason),
            })
            continue
        findings.append({
            "source": "definition_audit.uncollected",
            "severity": "action_required",
            "name": name,
            "reason": reason,
        })
    return findings, ignored_items, proposed_items


def diagnose_run(*, run: Path) -> dict[str, Any]:
    """Convert a completed run report into agent-readable proposal feedback."""
    run_dir = _resolve_run_dir(run)
    proposal_dir = run_dir / "proposal"
    report_path = run_dir / "reports" / "run_report.json"
    diagnostics_path = proposal_dir / "run_diagnostics.json"
    feedback_path = proposal_dir / "agent_feedback.md"
    proposal_dir.mkdir(parents=True, exist_ok=True)

    findings: list[dict[str, Any]] = []
    if not report_path.exists():
        findings.append({
            "source": "run_report",
            "severity": "error",
            "name": str(run_dir),
            "reason": f"missing run report: {report_path}",
        })
        report: dict[str, Any] = {}
    else:
        report = _load_json(report_path)
        if not isinstance(report, dict):
            findings.append({
                "source": "run_report",
                "severity": "error",
                "name": str(report_path),
                "reason": "run report must be a JSON object",
            })
            report = {}

    remote = report.get("remote") if isinstance(report.get("remote"), dict) else {}
    remote_summary = remote.get("summary") if isinstance(remote.get("summary"), dict) else {}
    early_stopped = bool(remote_summary.get("early_stopped"))
    early_stop_reason = remote_summary.get("early_stop_reason") or ""

    parse_report = report.get("parse_report") if isinstance(report.get("parse_report"), dict) else {}
    for name in parse_report.get("missing_targets", []) if isinstance(parse_report.get("missing_targets"), list) else []:
        if early_stopped:
            reason = (
                f"approved target had zero non-warmup events because the run was early-stopped "
                f"(reason: {early_stop_reason}). This target was NOT reached before the stop — "
                f"do NOT set collect:false. Fix the early-stop cause first, then re-run."
            )
            severity = "warning"
        else:
            reason = "approved target had zero non-warmup events; check hook target, runtime route, probe_mode, or run_config coverage"
            severity = "error"
        findings.append({
            "source": "parse_report",
            "severity": severity,
            "name": str(name),
            "reason": reason,
        })

    definition_audit = report.get("definition_audit") if isinstance(report.get("definition_audit"), dict) else {}
    _append_findings_from_items(
        findings,
        source="definition_audit.rejected",
        severity="error",
        items=definition_audit.get("rejected"),
    )
    diagnostics_section = report.get("diagnostics") if isinstance(report.get("diagnostics"), dict) else {}
    ignored_definitions = _ignored_definition_reasons(proposal_dir)
    proposed_definitions = _proposed_definition_candidates(proposal_dir)
    uncollected_findings, ignored_uncollected, proposed_uncollected = _uncollected_definition_findings(
        items=diagnostics_section.get("uncollected_definitions"),
        ignored=ignored_definitions,
        proposed=proposed_definitions,
    )
    findings.extend(uncollected_findings)

    collect = report.get("collect") if isinstance(report.get("collect"), dict) else {}
    manifest = collect.get("manifest") if isinstance(collect.get("manifest"), dict) else {}
    _append_findings_from_items(
        findings,
        source="collect.skipped",
        severity="error",
        items=[
            item
            for item in manifest.get("skipped", [])
            if isinstance(item, dict) and item.get("reason") != "collect is false"
        ] if isinstance(manifest.get("skipped"), list) else [],
    )
    workloads = manifest.get("workloads")
    if isinstance(workloads, list):
        for workload in workloads:
            if not isinstance(workload, dict):
                continue
            reject_reasons = workload.get("reject_reasons")
            sanitized_count = workload.get("sanitized_count")
            if isinstance(reject_reasons, dict) and reject_reasons:
                findings.append({
                    "source": "workload_sanitize",
                    "severity": "error",
                    "name": str(workload.get("name") or workload.get("definition_name") or "unknown"),
                    "reason": f"sanitize rejects: {reject_reasons}",
                })
            if sanitized_count == 0:
                findings.append({
                    "source": "workload_sanitize",
                    "severity": "error",
                    "name": str(workload.get("name") or workload.get("definition_name") or "unknown"),
                    "reason": "no captures were converted into workload entries",
                })

    internal = report.get("internal_validation") if isinstance(report.get("internal_validation"), dict) else {}
    _early_stop_suppressed_reasons = {
        "expected collect target has no workload",
        "approved target not observed in non-warmup events",
        "parse report marked target missing",
    }
    internal_findings = internal.get("findings")
    if early_stopped and isinstance(internal_findings, list):
        internal_findings = [
            {**item, "severity": "warning", "reason": f"[suppressed: early-stop] {item.get('reason', '')}"}
            if item.get("severity") == "error" and item.get("reason") in _early_stop_suppressed_reasons
            else item
            for item in internal_findings
            if isinstance(item, dict)
        ]
    _append_findings_from_items(
        findings,
        source="internal_validation",
        severity="error",
        items=internal_findings,
    )

    official = report.get("official_validation") if isinstance(report.get("official_validation"), dict) else {}
    if official and not official.get("ok", False):
        stdout = official.get("stdout")
        stderr = official.get("stderr")
        reason = _brief(stdout or stderr or f"returncode={official.get('returncode')}", limit=1200)
        findings.append({
            "source": "official_validation",
            "severity": "error",
            "name": "upstream_validator",
            "reason": reason,
        })

    errors = sum(1 for item in findings if item.get("severity") == "error")
    warnings = sum(1 for item in findings if item.get("severity") == "warning")
    action_required = sum(1 for item in findings if item.get("severity") == "action_required")
    diagnostics = {
        "summary": {
            "ok": errors == 0 and action_required == 0,
            "errors": errors,
            "warnings": warnings,
            "action_required": action_required,
            "findings": len(findings),
        },
        "run_dir": str(run_dir),
        "run_report": str(report_path),
        "findings": findings,
    }
    if ignored_uncollected:
        diagnostics["ignored_uncollected_definitions"] = ignored_uncollected
    if proposed_uncollected:
        diagnostics["proposed_uncollected_definitions"] = proposed_uncollected
    _write_json(diagnostics_path, diagnostics)
    feedback_path.write_text(_run_diagnostics_feedback_markdown(diagnostics), encoding="utf-8")
    return {
        "summary": diagnostics["summary"],
        "run_dir": str(run_dir),
        "outputs": {
            "diagnostics": str(diagnostics_path),
            "feedback": str(feedback_path),
        },
    }


def _run_diagnostics_feedback_markdown(diagnostics: dict[str, Any]) -> str:
    summary = diagnostics["summary"]
    status = "PASS" if summary["ok"] else "FIX_REQUIRED"
    lines = [
        "# Agent Proposal Feedback",
        "",
        f"- status: {status}",
        f"- source: run diagnostics",
        f"- run: {diagnostics['run_dir']}",
        f"- errors: {summary['errors']}",
        f"- warnings: {summary['warnings']}",
        f"- action_required: {summary.get('action_required', 0)}",
        "",
    ]
    if summary["ok"]:
        lines.extend([
            "## Next Action",
            "",
            "Run diagnostics found no proposal-level failures. Stop revising and hand the result to human review.",
            "",
        ])
        proposed = diagnostics.get("proposed_uncollected_definitions")
        if isinstance(proposed, list) and proposed:
            lines.extend([
                "## Proposed Uncollected Definitions",
                "",
                "These definitions were uncollected in the previous run, but the current proposal now contains collect candidates. After human approval, rerun the pipeline.",
                "",
            ])
            for item in proposed:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('name', 'unknown')}: proposed by {item.get('candidate', 'unknown')}")
            lines.append("")
        ignored = diagnostics.get("ignored_uncollected_definitions")
        if isinstance(ignored, list) and ignored:
            lines.extend(["## Ignored Uncollected Definitions", ""])
            for item in ignored:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('name', 'unknown')}: {item.get('reason', '')}")
            lines.append("")
        return "\n".join(lines)

    lines.extend([
        "## Next Agent Action",
        "",
        "Revise only the review-only proposal bundle. Prefer fixing proposal/definitions and proposal/definition_hints for non-FI issues, or candidate_targets/run_config when the failure is a missing target or wrong route. Do not approve targets and do not run Modal.",
        "",
        "## Findings",
        "",
    ])
    for item in diagnostics.get("findings", []):
        lines.append(
            f"- {item.get('severity', 'unknown')} [{item.get('source', 'unknown')}] "
            f"{item.get('name', 'unknown')}: {item.get('reason', 'unknown')}"
        )
    lines.append("")
    return "\n".join(lines)


def _repair_prompt_markdown(
    *,
    run_dir: Path,
    proposal_dir: Path,
    hf_config_path: Path,
    flashinfer_root: Path | None,
    diagnostics: dict[str, Any],
    check_result: dict[str, Any] | None = None,
) -> str:
    summary = diagnostics["summary"]
    flashinfer_arg = f" --flashinfer-root {flashinfer_root}" if flashinfer_root is not None else ""
    agent_loop_cmd = (
        "python3 -B -m tools.proposal_tools agent-loop "
        f"--proposal-dir {proposal_dir} "
        f"--hf-config {hf_config_path}"
        f"{flashinfer_arg}"
    )
    lines = [
        "# Repair Prompt",
        "",
        "Use the onboard-model-proposal skill in repair-pass mode to repair this review-only proposal.",
        "Do not restart first-pass onboarding for this run.",
        "",
        "## Scope",
        "",
        f"- run_dir: {run_dir}",
        f"- proposal_dir: {proposal_dir}",
        f"- diagnostics: {proposal_dir / 'agent_feedback.md'}",
        f"- hf_config: {hf_config_path}",
        f"- flashinfer_root: {flashinfer_root if flashinfer_root is not None else 'not provided'}",
        "",
        "## Hard Rules",
        "",
        "- This is repair-pass, not first-pass.",
        "- Edit only proposal artifacts: proposal/candidate_targets.json, proposal/architecture.md, proposal/review_checklist.md, proposal/definitions, and proposal/definition_hints.",
        "- Do not edit config/approved_targets.json, config/run_config.json, output/, reports/, or committed source code.",
        "- Do not approve candidates automatically.",
        "- Do not run Modal or any GPU job.",
        "- Fix the proposal so the deterministic checker passes, then stop for human review.",
        "",
        "## Current Diagnostics Summary",
        "",
        f"- status: {'PASS' if summary['ok'] else 'FIX_REQUIRED'}",
        f"- errors: {summary['errors']}",
        f"- warnings: {summary['warnings']}",
        f"- action_required: {summary.get('action_required', 0)}",
        "",
    ]
    findings = diagnostics.get("findings", [])
    if findings:
        lines.extend(["## Findings To Fix", ""])
        for item in findings:
            lines.append(
                f"- {item.get('severity', 'unknown')} [{item.get('source', 'unknown')}] "
                f"{item.get('name', 'unknown')}: {item.get('reason', 'unknown')}"
            )
        lines.append("")
    if check_result is not None:
        check_summary = check_result["summary"]
        lines.extend([
            "## Current Proposal Check Summary",
            "",
            f"- ready_for_human_review: {check_summary['ready_for_human_review']}",
            f"- errors: {check_summary['errors']}",
            f"- warnings: {check_summary['warnings']}",
            f"- feedback: {check_result['outputs']['feedback']}",
            f"- report: {check_result['outputs']['loop_report']}",
            "",
        ])
    lines.extend([
        "## Required Check",
        "",
        "After editing the proposal, run:",
        "",
        "```bash",
        agent_loop_cmd,
        "```",
        "",
        "Repeat proposal edits only until `ready for human review: True`, then stop.",
        "",
    ])
    return "\n".join(lines)


def repair_loop(
    *,
    run: Path,
    hf_config_path: Path,
    flashinfer_root: Path | None = None,
    agent_command: list[str] | None = None,
    max_rounds: int = 1,
) -> dict[str, Any]:
    """Generate repair prompts and optionally drive an external agent.

    This remains outside the runtime core. The optional agent command is a
    caller-provided executable that receives the repair prompt on stdin; this
    tool never approves config or runs Modal.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    run_dir = _resolve_run_dir(run)
    proposal_dir = run_dir / "proposal"
    prompt_path = proposal_dir / "repair_prompt.md"
    diagnostics_result: dict[str, Any] | None = None
    check_result: dict[str, Any] | None = None
    diagnostics_path: Path | None = None
    diagnostics: dict[str, Any] | None = None
    agent_rounds: list[dict[str, Any]] = []

    for round_index in range(1, max_rounds + 1):
        print(f"[repair-loop] round {round_index}/{max_rounds}: running diagnostics ...", flush=True)
        diagnostics_result = diagnose_run(run=run_dir)
        diagnostics_path = Path(diagnostics_result["outputs"]["diagnostics"])
        loaded = _load_json(diagnostics_path)
        if not isinstance(loaded, dict):
            raise ValueError(f"diagnostics must be a JSON object: {diagnostics_path}")
        diagnostics = loaded

        check_result = None
        if diagnostics_result["summary"]["ok"]:
            print(f"[repair-loop] round {round_index}/{max_rounds}: diagnostics ok, running proposal check ...", flush=True)
            check_result = run_agent_loop(
                proposal_dir=proposal_dir,
                hf_config_path=hf_config_path,
                flashinfer_root=flashinfer_root,
            )

        prompt_text = _repair_prompt_markdown(
            run_dir=run_dir,
            proposal_dir=proposal_dir,
            hf_config_path=hf_config_path,
            flashinfer_root=flashinfer_root,
            diagnostics=diagnostics,
            check_result=check_result,
        )
        prompt_path.write_text(prompt_text, encoding="utf-8")

        ready = bool(
            diagnostics_result["summary"]["ok"]
            and check_result
            and check_result["summary"]["ready_for_human_review"]
        )
        if ready or not agent_command or round_index >= max_rounds:
            break
        print(f"[repair-loop] round {round_index}/{max_rounds}: invoking agent ...", flush=True)
        completed = subprocess.run(
            agent_command,
            check=False,
            text=True,
            input=prompt_text,
        )
        print(f"[repair-loop] round {round_index}/{max_rounds}: agent exited (returncode={completed.returncode})", flush=True)
        agent_round = {
            "round": round_index,
            "command": agent_command,
            "returncode": completed.returncode,
        }
        agent_rounds.append(agent_round)
        if completed.returncode != 0:
            break

    if diagnostics_result is None or diagnostics_path is None:
        raise RuntimeError("repair loop did not run")
    check_summary = check_result["summary"] if check_result else diagnostics_result["summary"]
    ready_for_human_review = bool(
        diagnostics_result["summary"]["ok"]
        and check_result
        and check_result["summary"]["ready_for_human_review"]
    )
    result = {
        "summary": {
            "diagnostics_ok": diagnostics_result["summary"]["ok"],
            "diagnostics_action_required": diagnostics_result["summary"].get("action_required", 0),
            "ready_for_human_review": ready_for_human_review,
            "errors": check_summary["errors"],
            "warnings": check_summary["warnings"],
            "agent_ran": bool(agent_rounds),
            "rounds": round_index,
            "max_rounds": max_rounds,
        },
        "run_dir": str(run_dir),
        "outputs": {
            "diagnostics": str(diagnostics_path),
            "feedback": diagnostics_result["outputs"]["feedback"],
            "repair_prompt": str(prompt_path),
            "agent_loop": check_result["outputs"]["loop_report"] if check_result else None,
        },
        "agent": agent_rounds[-1] if agent_rounds else None,
        "agent_rounds": agent_rounds,
    }
    _write_json(proposal_dir / "repair_loop.json", result)
    print(f"[repair-loop] done: ready={ready_for_human_review}, errors={check_summary['errors']}, warnings={check_summary['warnings']}", flush=True)
    print(f"[repair-loop] repair_loop.json -> {proposal_dir / 'repair_loop.json'}", flush=True)
    return result


def _add_prepare_agent_inputs_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-agent-inputs",
        help="Refresh local HF config and sgl-cookbook material for review-only onboarding.",
    )
    parser.add_argument("--model", action="append", required=True, help="HF model name. Can be passed multiple times.")
    parser.add_argument("--output-root", type=Path, default=Path("agent_inputs"))
    parser.add_argument("--refresh", action="store_true", help="Update cookbook and re-download configs.")
    parser.add_argument("--cookbook-repo", default=DEFAULT_COOKBOOK_REPO)
    parser.add_argument("--check-sglang-root", type=Path, default=Path("agent_inputs/sglang/python/sglang"))
    parser.add_argument("--check-flashinfer-root", type=Path, default=Path("agent_inputs/flashinfer/flashinfer"))


def _add_check_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "check",
        help="Check candidate fields and FlashInfer fitrace collect targets in one gate.",
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        help="Path to flashinfer/ source root for static fitrace checks. Defaults to agent_inputs/flashinfer/flashinfer when present.",
    )
    parser.add_argument("--output", type=Path)


def _add_agent_loop_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "agent-loop",
        help="Run one proposal check loop and write feedback for the external agent.",
    )
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        help="Path to flashinfer/ source root for static fitrace checks.",
    )


def _add_check_proposal_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "check-proposal",
        help="Run deterministic checks for a full proposal bundle. Does not invoke an agent.",
    )
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        help="Path to flashinfer/ source root for static fitrace checks.",
    )


def _add_diagnose_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "diagnose-run",
        help="Convert reports/run_report.json into proposal/agent_feedback.md for the next agent pass.",
    )
    parser.add_argument("--run", type=Path, required=True, help="Run path or path relative to runs/.")


def _add_repair_loop_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "repair-loop",
        help="Generate a fixed repair prompt from run diagnostics and optionally invoke an external agent.",
    )
    parser.add_argument("--run", type=Path, required=True, help="Run path or path relative to runs/.")
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        help="Path to flashinfer/ source root for static fitrace checks.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help="Maximum repair/check rounds when --agent-command is provided.",
    )
    parser.add_argument(
        "--agent-command",
        nargs=argparse.REMAINDER,
        help="Optional external agent command. The repair prompt is sent to stdin; pass this after repair-loop options.",
    )


def _add_spawn_agents_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "spawn-agents",
        help="Generate first-pass proposal prompts and optionally run N external agents.",
    )
    parser.add_argument("--model", required=True, help="HF model name, for example microsoft/Phi-4-mini-instruct.")
    parser.add_argument(
        "--run-prefix",
        type=Path,
        help=(
            "Base run path or path relative to runs/. Defaults to <model_slug>/<YYYYMMDD>_firstpass. "
            "With --count 1 this exact run is used; "
            "with --count N, sibling runs ending in _agent_a/_agent_b/... are used."
        ),
    )
    parser.add_argument("--hf-config", type=Path, help="Defaults to agent_inputs/config/<model_slug>.json.")
    parser.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    parser.add_argument("--flashinfer-root", type=Path, default=DEFAULT_FLASHINFER_ROOT)
    parser.add_argument("--cookbook-root", type=Path, default=DEFAULT_COOKBOOK_ROOT)
    parser.add_argument(
        "--sglang-model-hint",
        action="append",
        default=[],
        help="Relative SGLang source hint. Can be passed multiple times.",
    )
    parser.add_argument("--count", type=int, default=1, help="Number of first-pass agents/prompts. Defaults to 1.")
    parser.add_argument(
        "--merge-output-dir",
        type=Path,
        help="Output proposal directory for merge-proposals after agents finish. Defaults to <run_prefix>_merged/proposal when --count > 1.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge multi-agent proposal outputs automatically.",
    )
    parser.add_argument(
        "--agent",
        choices=["codex"],
        help="Shortcut external agent command. Currently supports codex.",
    )
    parser.add_argument(
        "--agent-command",
        nargs=argparse.REMAINDER,
        help="Optional explicit external agent command. Overrides --agent. Each prompt is sent to stdin; pass this after spawn-agents options.",
    )


def _add_merge_proposals_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "merge-proposals",
        help="Union multiple review-only proposal bundles and report conflicts for human review.",
    )
    parser.add_argument(
        "--proposal-dir",
        type=Path,
        action="append",
        required=True,
        help="Proposal directory, or run directory containing proposal/. Pass at least two.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output proposal directory.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline proposal onboarding tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_agent_inputs_parser(subparsers)
    _add_check_parser(subparsers)
    _add_agent_loop_parser(subparsers)
    _add_check_proposal_parser(subparsers)
    _add_diagnose_run_parser(subparsers)
    _add_repair_loop_parser(subparsers)
    _add_spawn_agents_parser(subparsers)
    _add_merge_proposals_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare-agent-inputs":
        report = prepare_agent_inputs(
            models=args.model,
            output_root=args.output_root,
            refresh=args.refresh,
            cookbook_repo=args.cookbook_repo,
            check_sglang_root=args.check_sglang_root,
            check_flashinfer_root=args.check_flashinfer_root,
        )
        report_path = args.output_root / "prepare_report.json"
        _write_json(report_path, report)
        summary = report["summary"]
        print(f"models: {summary['models']}")
        print(f"configs ok: {summary['configs_ok']}")
        print(f"cookbook ok: {summary['cookbook_ok']}")
        print(f"cookbook matches: {summary['cookbook_matches']}")
        print(f"source checks ok: {summary['source_checks_ok']}")
        print(f"report: {report_path}")
        return 0

    if args.command == "check":
        proposal_dir = args.candidates.parent
        report = check_proposal(
            proposal_dir=proposal_dir,
            candidates_path=args.candidates,
            hf_config_path=args.hf_config,
            flashinfer_root=args.flashinfer_root,
        )
        if args.output:
            _write_json(args.output, report)
        summary = report["summary"]
        print(f"entries: {summary['entries']}")
        print(f"collect candidates: {summary['collect_candidates']}")
        print(f"importable targets: {summary['importable_targets']}")
        print(f"fitrace targets: {summary['fitrace_targets']}")
        print(f"definition draft targets: {summary['definition_draft_targets']}")
        print(f"errors: {summary['errors']}")
        print(f"warnings: {summary['warnings']}")
        if args.output:
            print(f"report: {args.output}")
        return 0 if summary["ok"] else 1

    if args.command == "agent-loop":
        result = run_agent_loop(
            proposal_dir=args.proposal_dir,
            candidates_path=args.candidates,
            hf_config_path=args.hf_config,
            flashinfer_root=args.flashinfer_root,
        )
        summary = result["summary"]
        print(f"ready for human review: {summary['ready_for_human_review']}")
        print(f"errors: {summary['errors']}")
        print(f"warnings: {summary['warnings']}")
        print(f"feedback: {result['outputs']['feedback']}")
        print(f"report: {result['outputs']['loop_report']}")
        return 0 if summary["ok"] else 1

    if args.command == "check-proposal":
        result = run_agent_loop(
            proposal_dir=args.proposal_dir,
            candidates_path=args.candidates,
            hf_config_path=args.hf_config,
            flashinfer_root=args.flashinfer_root,
        )
        summary = result["summary"]
        print(f"proposal check ok: {summary['ok']}")
        print(f"ready for human review: {summary['ready_for_human_review']}")
        print(f"errors: {summary['errors']}")
        print(f"warnings: {summary['warnings']}")
        print(f"feedback: {result['outputs']['feedback']}")
        print(f"report: {result['outputs']['loop_report']}")
        return 0 if summary["ok"] else 1

    if args.command == "diagnose-run":
        result = diagnose_run(run=args.run)
        summary = result["summary"]
        print(f"run diagnostics ok: {summary['ok']}")
        print(f"errors: {summary['errors']}")
        print(f"warnings: {summary['warnings']}")
        print(f"action_required: {summary.get('action_required', 0)}")
        print(f"feedback: {result['outputs']['feedback']}")
        print(f"report: {result['outputs']['diagnostics']}")
        return 0 if summary["ok"] else 1

    if args.command == "repair-loop":
        result = repair_loop(
            run=args.run,
            hf_config_path=args.hf_config,
            flashinfer_root=args.flashinfer_root,
            agent_command=args.agent_command,
            max_rounds=args.max_rounds,
        )
        summary = result["summary"]
        print(f"diagnostics ok: {summary['diagnostics_ok']}")
        print(f"diagnostics action_required: {summary.get('diagnostics_action_required', 0)}")
        print(f"ready for human review: {summary['ready_for_human_review']}")
        print(f"errors: {summary['errors']}")
        print(f"warnings: {summary['warnings']}")
        print(f"rounds: {summary['rounds']}/{summary['max_rounds']}")
        print(f"agent ran: {summary['agent_ran']}")
        print(f"repair prompt: {result['outputs']['repair_prompt']}")
        print(f"feedback: {result['outputs']['feedback']}")
        print(f"report: {result['outputs']['agent_loop']}")
        return 0 if summary["ready_for_human_review"] else 1

    if args.command == "spawn-agents":
        run_prefix = args.run_prefix or _default_run_prefix(args.model)
        hf_config_path = args.hf_config or _default_hf_config_path(args.model)
        agent_command = args.agent_command
        agent_env = None
        if agent_command is None:
            agent_command, agent_env = _agent_command_from_shortcut(args.agent)
        merge_output_dir = args.merge_output_dir
        if merge_output_dir is None and args.count > 1 and agent_command is not None and not args.no_merge:
            merge_output_dir = _default_merge_output_dir(run_prefix)
        result = spawn_agents(
            model_name=args.model,
            run_prefix=run_prefix,
            hf_config_path=hf_config_path,
            sglang_root=args.sglang_root,
            flashinfer_root=args.flashinfer_root,
            cookbook_root=args.cookbook_root,
            sglang_model_hints=args.sglang_model_hint,
            count=args.count,
            agent_command=agent_command,
            agent_env=agent_env,
            merge_output_dir=merge_output_dir,
            progress=True,
        )
        summary = result["summary"]
        print(f"model: {summary['model']}")
        print(f"prompts: {summary['count']}")
        print(f"agents started: {summary['agents_started']}")
        print(f"agent failures: {summary['agent_failures']}")
        for item in result["prompts"]:
            print(f"prompt: {item['prompt']}")
        if result["merge_report"]:
            print(f"merge report: {result['merge_report']}")
        print(f"report: {result['report']}")
        ok = summary["agent_failures"] == 0
        if summary["merge_ok"] is not None:
            ok = ok and bool(summary["merge_ok"])
        return 0 if ok else 1

    if args.command == "merge-proposals":
        result = merge_proposals(
            proposal_dirs=args.proposal_dir,
            output_dir=args.output_dir,
        )
        summary = result["summary"]
        print(f"merged candidates: {summary['merged_candidates']}")
        print(f"conflicts: {summary['conflicts']}")
        print(f"candidate conflicts: {summary['candidate_conflicts']}")
        print(f"draft file conflicts: {summary['draft_file_conflicts']}")
        print(f"output: {args.output_dir}")
        print(f"review: {args.output_dir / 'merge_review.md'}")
        print(f"report: {args.output_dir / 'merge_report.json'}")
        return 0 if summary["ok"] else 1

    raise SystemExit(f"ERROR: unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
