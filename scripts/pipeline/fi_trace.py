"""Stage official FlashInfer fi_trace definitions into the dataset layout."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path


def _canonical_json(value: object) -> str:
    """Canonical JSON string used for same/different checks."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _axis_const_value(defn: dict, axis_name: str) -> int | None:
    axes = defn.get("axes", {})
    if not isinstance(axes, dict):
        return None
    axis = axes.get(axis_name, {})
    if not isinstance(axis, dict) or axis.get("type") != "const":
        return None
    value = axis.get("value")
    return value if isinstance(value, int) else None


def _reject_fi_trace_definition(defn: dict) -> str | None:
    """Return a rejection reason for known-bad official fi_trace definitions."""
    if defn.get("op_type") != "gqa_paged":
        return None

    q_heads = _axis_const_value(defn, "num_qo_heads")
    kv_heads = _axis_const_value(defn, "num_kv_heads")
    if q_heads is None or kv_heads is None:
        return None
    if kv_heads > q_heads:
        return (
            "gqa_paged num_kv_heads is larger than num_qo_heads; "
            f"likely old-runtime KV-cache shape mismatch ({kv_heads} > {q_heads})"
        )
    if q_heads % kv_heads != 0:
        return (
            "gqa_paged num_qo_heads must be divisible by num_kv_heads "
            f"({q_heads} % {kv_heads} != 0)"
        )
    return None


def _normalize_definition_reference(defn: dict) -> None:
    """Normalize official fi_trace reference snippets to dataset schema rules."""
    reference = defn.get("reference")
    if not isinstance(reference, str):
        return
    if "def run(" not in reference:
        reference = re.sub(
            r"^(\s*)def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
            r"\1def run(",
            reference,
            count=1,
            flags=re.MULTILINE,
        )
    imports: list[str] = []
    if "torch" in reference and "import torch" not in reference:
        imports.append("import torch")
    if "math." in reference and "import math" not in reference:
        imports.append("import math")
    if imports:
        reference = "\n".join(imports) + "\n\n" + reference
    defn["reference"] = reference


def _set_description(container: dict, key: str, description: str) -> None:
    value = container.get(key)
    if isinstance(value, dict) and not value.get("description"):
        value["description"] = description


def _normalize_fi_trace_definition(defn: dict) -> None:
    """Normalize official fi_trace JSONs to the current dataset schema."""
    _normalize_definition_reference(defn)

    tags = defn.get("tags")
    tag_set = set(tags) if isinstance(tags, list) else set()
    fi_api = next(
        (
            tag.removeprefix("fi_api:")
            for tag in tag_set
            if isinstance(tag, str) and tag.startswith("fi_api:")
        ),
        "",
    )
    op_type = defn.get("op_type")
    if op_type == "gqa_ragged":
        axes = defn.get("axes")
        inputs = defn.get("inputs")
        if isinstance(axes, dict):
            _set_description(axes, "num_qo_heads", "Number of query/output attention heads.")
            _set_description(axes, "num_kv_heads", "Number of key/value attention heads.")
            _set_description(axes, "head_dim", "Per-head hidden dimension.")
        if isinstance(inputs, dict):
            _set_description(inputs, "q", "Ragged query tensor.")
            _set_description(inputs, "k", "Ragged key tensor.")
            _set_description(inputs, "v", "Ragged value tensor.")
            for name in ("qo_indptr", "kv_indptr"):
                spec = inputs.get(name)
                if isinstance(spec, dict) and spec.get("dtype") == "unknown":
                    spec["dtype"] = "int32"
        return

    if op_type == "cascade_merge":
        axes = defn.get("axes")
        outputs = defn.get("outputs")
        if isinstance(axes, dict):
            _set_description(axes, "num_heads", "Number of attention heads.")
            _set_description(axes, "head_dim", "Per-head hidden dimension.")
        if isinstance(outputs, dict):
            _set_description(outputs, "v_merged", "Merged attention output state.")
            _set_description(outputs, "s_merged", "Merged logsumexp state.")
        return

    if op_type != "rmsnorm":
        return

    axes = defn.get("axes")
    inputs = defn.get("inputs")
    outputs = defn.get("outputs")
    if not isinstance(axes, dict) or not isinstance(inputs, dict) or not isinstance(outputs, dict):
        return

    _set_description(axes, "batch_size", "Number of rows/tokens to normalize.")
    _set_description(axes, "hidden_size", "Hidden dimension normalized per row.")
    _set_description(inputs, "hidden_states", "Input activations to normalize.")
    _set_description(inputs, "input", "Input activations to normalize.")
    _set_description(inputs, "residual", "Residual tensor added before normalization.")
    _set_description(inputs, "weight", "Per-channel RMSNorm scale.")
    _set_description(outputs, "output", "Normalized output activations.")

    if fi_api == "flashinfer.norm.fused_add_rmsnorm":
        # Current dataset schema rejects input/output name overlap. The official
        # template exposes residual as an in-place updated tensor, but the
        # reference returns only the normalized output, so keep the explicit
        # output and drop the overlapping residual field.
        outputs.pop("residual", None)


def stage_fi_trace_definitions(
    fi_trace_out_dir: Path,
    definitions_dir: Path,
    *,
    dry_run: bool,
    project_root: Path | None = None,
    log: Callable[[str], None] = print,
    log_step: Callable[[str], None] | None = None,
) -> list[str]:
    """Stage official FlashInfer fi_trace JSONs into definitions/{op_type}/.

    FlashInfer writes all JSON files into one flat dump directory. The dataset
    layout groups definitions by op_type, so this function performs only that
    filesystem staging step and refuses to overwrite different existing files.
    """
    if not fi_trace_out_dir.is_dir():
        log(f"ℹ️  No official fi_trace output found: {fi_trace_out_dir}")
        return []

    if log_step is not None:
        log_step("Stage Official fi_trace Definitions")
    staged: list[str] = []
    skipped_same: list[str] = []
    conflicts: list[str] = []
    invalid: list[str] = []
    root = project_root.resolve() if project_root is not None else None

    for src in sorted(fi_trace_out_dir.glob("*.json")):
        try:
            defn = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            invalid.append(f"{src.name}: invalid JSON ({exc})")
            continue

        name = defn.get("name")
        op_type = defn.get("op_type")
        if not isinstance(name, str) or not name or not isinstance(op_type, str) or not op_type:
            invalid.append(f"{src.name}: missing string name/op_type")
            continue
        issue = _reject_fi_trace_definition(defn)
        if issue:
            invalid.append(f"{src.name}: {issue}")
            continue
        _normalize_fi_trace_definition(defn)

        dst = definitions_dir / op_type / f"{name}.json"
        if root is not None:
            try:
                rel = dst.resolve().relative_to(root).as_posix()
            except ValueError:
                rel = str(dst)
        else:
            rel = str(dst)
        if dst.exists():
            try:
                existing = json.loads(dst.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                conflicts.append(f"{rel}: existing JSON is invalid ({exc})")
                continue
            if _canonical_json(existing) == _canonical_json(defn):
                skipped_same.append(name)
                continue
            conflicts.append(f"{rel}: existing definition differs from official fi_trace output")
            continue

        staged.append(name)
        if dry_run:
            log(f"(dry-run) Would stage {src.name} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(defn, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log(f"✅ staged {src.name} -> {dst}")

    if skipped_same:
        log(f"⏭️  Already present with same content: {len(skipped_same)}")
    if invalid:
        for item in invalid:
            log(f"⚠️  Invalid fi_trace definition: {item}")
    if conflicts:
        for item in conflicts:
            log(f"❌ fi_trace definition conflict: {item}")
        raise RuntimeError(f"{len(conflicts)} fi_trace definition conflict(s)")

    manifest = {
        "fi_trace_out_dir": str(fi_trace_out_dir),
        "definitions_dir": str(definitions_dir),
        "staged": staged,
        "skipped_same": skipped_same,
        "invalid": invalid,
    }
    manifest_path = fi_trace_out_dir.parent / "fi_trace_staged_definitions.json"
    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"fi_trace staged: {len(staged)} new, {len(skipped_same)} existing, manifest={manifest_path}")
    return staged


def stage_runtime_fi_trace_if_needed(
    *,
    runtime_requested: bool,
    probe_output: Path | None,
    definitions_dir: Path,
    dry_run: bool,
    project_root: Path,
    log: Callable[[str], None],
    log_step: Callable[[str], None],
) -> Path | None:
    """Stage official fi_trace definitions for a runtime probe output."""
    if not runtime_requested or probe_output is None:
        return None

    fi_trace_out_dir = probe_output.parent / "fi_trace_out"
    manifest_path = fi_trace_out_dir.parent / "fi_trace_staged_definitions.json"
    stage_fi_trace_definitions(
        fi_trace_out_dir=fi_trace_out_dir,
        definitions_dir=definitions_dir,
        dry_run=dry_run,
        project_root=project_root,
        log=log,
        log_step=log_step,
    )
    return manifest_path
