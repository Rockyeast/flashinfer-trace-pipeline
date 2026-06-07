#!/usr/bin/env python3
"""Generate definition JSONs from kernel_inventory.json.

This is Step 3 of the v2 pipeline. It reads the kernel inventory produced by
parse_probe.py and generates definition JSON files for kernels that don't
already exist.

Adapter-owned op_types render definitions through scripts/adapters/<op_type>.py.

Usage:
    python scripts/generate_definitions.py \
        tmp/run/Qwen_.../kernel_inventory.json \
        --model-tag qwen3.5-35b-a3b \
        --tp 2

    # Dry run (show what would be generated)
    python scripts/generate_definitions.py \
        tmp/run/Qwen_.../kernel_inventory.json \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from adapters import generate_definition as generate_adapter_definition  # noqa: E402
from adapters import ignored_definition_inputs_for_observed_params  # noqa: E402
from adapters import ignored_observed_kwargs  # noqa: E402
from parse.inventory_helpers import is_existing_kernel  # noqa: E402
from artifact_schemas import validate_definition  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation: observed_kwargs vs definition inputs
# ──────────────────────────────────────────────────────────────────────────────


def _warn_kwargs_mismatch(kernel: dict, definition: dict) -> None:
    """Compare probe-observed kwargs AND param_names with adapter-produced input names.

    两层校验：
    1) observed_kwargs — kwargs key 名（如 sm_scale、k_scale），只含真正的关键字参数
    2) observed_param_names — 函数签名的完整参数名列表（含位置参数和 kwargs），
       来自 inspect.signature()，可以将匿名的 args[0] 映射回真实参数名（如 qo_indptr）

    只对有实际数据的 kernel 做校验。纯位置参数的 kernel（如 rmsnorm）通常
    observed_kwargs 为空，但 observed_param_names 可能非空。
    """
    def_inputs = set(definition.get("inputs", {}).keys())
    if not def_inputs:
        return  # definition 没有 inputs 字段（异常情况），跳过

    def_name = kernel.get("definition_name", "?")

    # 校验 1：kwargs key 名 vs definition inputs
    obs_kw = kernel.get("observed_kwargs", [])
    if obs_kw:
        obs_set = set(obs_kw)
        ignored = ignored_observed_kwargs(kernel, definition)
        # 只检查 obs → def 方向：probe 看到但 definition 没定义的 kwargs
        # （definition inputs 可能包含纯位置参数如 q、k、v，不在 kwargs 里，不算 mismatch）
        # 一些 runtime kwargs 是官方 TraceTemplate 明确不建模的控制参数，
        # 例如 sampling 的 filter_apply_order/check_nan，交给 adapter 声明后忽略。
        probe_only = obs_set - def_inputs - ignored
        if probe_only:
            print(
                f"  ⚠️  kwargs mismatch [{def_name}]: "
                f"probe observed kwargs not in definition inputs: {sorted(probe_only)}",
                file=sys.stderr,
            )

    # 校验 2：param_names（函数签名参数名）vs definition inputs
    obs_pn = kernel.get("observed_param_names", [])
    if obs_pn:
        param_set = set(obs_pn) | set(kernel.get("observed_kwargs", []))
        ignored_inputs = ignored_definition_inputs_for_observed_params(
            kernel,
            definition,
        )
        # definition inputs 里有但函数签名里没有的参数名 → 可能是模板用了错误的名字
        def_not_in_sig = def_inputs - param_set - ignored_inputs
        if def_not_in_sig:
            print(
                f"  ⚠️  param_names mismatch [{def_name}]: "
                f"definition inputs not in function signature: {sorted(def_not_in_sig)}",
                file=sys.stderr,
            )


DECISION_GENERATE = "generate_with_adapter"
DECISION_SKIP_EXISTING = "skip_existing"
DECISION_SKIP_NOT_FLASHINFER_API = "skip_not_flashinfer_api"
DECISION_SKIP_NEEDS_CONFIG = "skip_needs_config"


@dataclass(frozen=True)
class DefinitionDecision:
    action: str
    reason: str
    output_path: Path


def load_staged_fi_trace_names(manifest_path: Path | None) -> set[str]:
    """Load definition names staged from official fi_trace output."""
    if manifest_path is None or not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    names: set[str] = set()
    for key in ("staged", "skipped_same"):
        values = manifest.get(key, [])
        if isinstance(values, list):
            names.update(v for v in values if isinstance(v, str) and v)
    return names


def resolve_definition_decision(
    kernel: dict,
    definitions_dir: Path,
    *,
    include_existing: bool,
    replace: bool,
    staged_fi_trace_names: set[str] | None = None,
) -> DefinitionDecision:
    """Decide whether one inventory entry should enter adapter generation."""
    def_name = kernel["definition_name"]
    op_type = kernel["op_type"]
    out_path = definitions_dir / op_type / f"{def_name}.json"
    staged_fi_trace_names = staged_fi_trace_names or set()

    if def_name in staged_fi_trace_names and out_path.exists() and not replace:
        return DefinitionDecision(
            action=DECISION_SKIP_EXISTING,
            reason="official fi_trace definition already staged",
            output_path=out_path,
        )

    if (is_existing_kernel(kernel) or out_path.exists()) and not include_existing and not replace:
        return DefinitionDecision(
            action=DECISION_SKIP_EXISTING,
            reason="definition already exists",
            output_path=out_path,
        )

    if out_path.exists() and not replace:
        return DefinitionDecision(
            action=DECISION_SKIP_EXISTING,
            reason="definition already exists; use --replace to overwrite",
            output_path=out_path,
        )

    fi_api = kernel.get("fi_api")
    if not isinstance(fi_api, str) or not fi_api.startswith("flashinfer."):
        return DefinitionDecision(
            action=DECISION_SKIP_NOT_FLASHINFER_API,
            reason="no observed flashinfer.* API evidence",
            output_path=out_path,
        )

    if "NEEDS_CONFIG" in def_name:
        return DefinitionDecision(
            action=DECISION_SKIP_NEEDS_CONFIG,
            reason="kernel still needs HF config resolution",
            output_path=out_path,
        )

    return DefinitionDecision(
        action=DECISION_GENERATE,
        reason="ready for adapter generation",
        output_path=out_path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────


def generate_definition(
    kernel: dict,
    model_tag: str,
    tp: int,
) -> dict | None:
    """Generate a definition dict for a single kernel entry.

    Definitions are rendered only through scripts/adapters/. Unknown op_types
    must get a reviewed adapter first so classification, inventory extraction,
    definition naming, and definition rendering stay together.
    """
    return generate_adapter_definition(kernel, model_tag, tp)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate definition JSONs from kernel inventory",
    )
    parser.add_argument(
        "inventory_path",
        type=Path,
        help="Path to kernel_inventory.json (from parse_probe.py)",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Model tag for definitions (e.g. qwen3.5-35b-a3b). Auto-detected from inventory if not set.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="Tensor parallelism degree (default: 2)",
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=Path("definitions"),
        help="Output root directory for definitions",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing definitions",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Also regenerate definitions that already exist",
    )
    parser.add_argument(
        "--include-deferred-kernels",
        action="store_true",
        help="Also generate definitions listed in inventory.deferred_kernels for manual experiments.",
    )
    parser.add_argument(
        "--fi-trace-manifest",
        type=Path,
        default=None,
        help="Optional fi_trace_staged_definitions.json manifest for source-priority reporting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )
    args = parser.parse_args()

    if not args.inventory_path.exists():
        print(f"ERROR: {args.inventory_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.inventory_path, encoding="utf-8") as f:
        inventory = json.load(f)

    model_tag = args.model_tag
    if not model_tag:
        # Auto-detect from model name: "Qwen/Qwen3.5-35B-A3B" → "qwen3.5-35b-a3b"
        model_name = inventory.get("model", "")
        model_tag = model_name.split("/")[-1].lower() if "/" in model_name else model_name.lower()
        if not model_tag:
            model_tag = "unknown"

    kernels = list(inventory.get("kernels", []))
    deferred_kernels = list(inventory.get("deferred_kernels", []))
    if args.include_deferred_kernels:
        kernels.extend(deferred_kernels)

    generated = 0
    skipped_existing = 0
    skipped_not_flashinfer_api = 0
    skipped_needs_config = 0
    skipped_deferred = len(deferred_kernels) if not args.include_deferred_kernels else 0
    not_generated = 0
    staged_fi_trace_names = load_staged_fi_trace_names(args.fi_trace_manifest)

    for kernel in kernels:
        def_name = kernel["definition_name"]
        decision = resolve_definition_decision(
            kernel,
            args.definitions_dir,
            include_existing=args.include_existing,
            replace=args.replace,
            staged_fi_trace_names=staged_fi_trace_names,
        )

        if decision.action != DECISION_GENERATE:
            if decision.action == DECISION_SKIP_EXISTING:
                skipped_existing += 1
            elif decision.action == DECISION_SKIP_NOT_FLASHINFER_API:
                skipped_not_flashinfer_api += 1
            elif decision.action == DECISION_SKIP_NEEDS_CONFIG:
                skipped_needs_config += 1

            if args.dry_run:
                print(f"  ⏭️  {def_name}: {decision.reason}")
            elif decision.action != DECISION_SKIP_EXISTING:
                print(f"  ⚠️  {def_name}: {decision.reason}, skipping")
            continue

        # Generate definition through the owning adapter only.
        definition = generate_definition(
            kernel,
            model_tag,
            args.tp,
        )
        if definition is None:
            not_generated += 1
            print(
                f"  ❌ {def_name}: no adapter produced a definition",
                file=sys.stderr,
            )
            continue

        # Cross-validate observed kwargs vs generated definition inputs
        _warn_kwargs_mismatch(kernel, definition)
        validate_definition(definition)

        if args.dry_run:
            print(f"  🆕 {def_name} → {decision.output_path}")
            continue

        out_path = decision.output_path
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(definition, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        generated += 1
        print(f"  ✅ {def_name} → {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Generate Definitions Summary:")
    print(f"  ✅ Generated:             {generated}")
    print(f"  ⏭️  Skipped existing:       {skipped_existing}")
    print(f"  ⏸️  Skipped not FlashInfer: {skipped_not_flashinfer_api}")
    print(f"  ⚠️  Skipped needs config:   {skipped_needs_config}")
    print(f"  ⏸️  Deferred:              {skipped_deferred}")
    print(f"  ❌ Not generated:          {not_generated}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
