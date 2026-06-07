#!/usr/bin/env python3
"""Generate a static candidate kernel inventory.

This is the static entrypoint. It does not run inference. It builds candidate
kernels from HF config, SGLang source signals, and adapter-owned metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adapters._param_utils import extract_hf_dims  # noqa: E402
from parse.inventory_helpers import (  # noqa: E402
    is_existing_kernel,
    is_new_kernel,
    needs_config_kernel,
)
from static.sglang_analyzer import analyze_sglang_model  # noqa: E402
from static.static_kernel_candidates import build_static_kernels  # noqa: E402


def _normalize_page_sizes(page_sizes: list[int] | None) -> list[int]:
    """Return positive page_size values with stable de-duplication."""
    normalized: list[int] = []
    for page_size in page_sizes or [1]:
        if page_size <= 0:
            continue
        if page_size not in normalized:
            normalized.append(page_size)
    return normalized or [1]


def build_static_inventory(
    *,
    model_name: str,
    hf_config_path: Path,
    definitions_dir: Path | None,
    sglang_root: Path | None,
    tp: int,
    page_sizes: list[int],
    page_size_source: str,
) -> dict:
    """Return a static candidate inventory dict."""
    config = json.loads(hf_config_path.read_text(encoding="utf-8"))
    dims = extract_hf_dims(config, tp)
    analysis = analyze_sglang_model(sglang_root, config)
    normalized_page_sizes = _normalize_page_sizes(page_sizes)
    kernels_by_name: dict[str, dict] = {}
    for page_size in normalized_page_sizes:
        for kernel in build_static_kernels(
            dims=dims,
            analysis=analysis,
            model_name=model_name,
            tp=tp,
            definitions_dir=definitions_dir,
            page_size=page_size,
            page_size_source=page_size_source,
        ):
            kernels_by_name.setdefault(kernel["definition_name"], kernel)
    kernels = list(kernels_by_name.values())
    return {
        "model": model_name,
        "discovery_source": "static",
        "hf_config": str(hf_config_path),
        "sglang_model_file": analysis.get("model_file"),
        "tp": tp,
        "page_sizes": normalized_page_sizes,
        "page_size_source": page_size_source,
        "kernels": kernels,
        "deferred_kernels": [],
        "summary": {
            "total": len(kernels),
            "existing": sum(1 for k in kernels if is_existing_kernel(k)),
            "new": sum(1 for k in kernels if is_new_kernel(k)),
            "needs_config": sum(1 for k in kernels if needs_config_kernel(k)),
            "static_candidates": len(kernels),
            "runtime_observed": 0,
        },
        "static_analysis": {
            "dims": dims,
            "sglang": analysis,
            "page_sizes": normalized_page_sizes,
            "page_size_source": page_size_source,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Static kernel candidates into kernel_inventory.json")
    parser.add_argument("--model-name", required=True, help="HuggingFace model name")
    parser.add_argument("--hf-config", type=Path, required=True, help="Path to HF config.json")
    parser.add_argument("--definitions-dir", type=Path, default=Path("definitions"))
    parser.add_argument("--sglang-root", type=Path, default=Path("sglang"))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=None, help="Legacy single explicit page_size.")
    parser.add_argument(
        "--page-sizes",
        nargs="+",
        type=int,
        default=None,
        help="One or more page_size values for paged static candidates.",
    )
    parser.add_argument("--output", type=Path, default=Path("kernel_inventory.json"))
    args = parser.parse_args()

    if not args.hf_config.exists():
        print(f"ERROR: HF config not found: {args.hf_config}", file=sys.stderr)
        sys.exit(1)

    sglang_root = args.sglang_root if args.sglang_root.exists() else None
    if args.page_sizes is not None:
        page_sizes = args.page_sizes
        page_size_source = "explicit_probe_page_sizes"
    elif args.page_size is not None:
        page_sizes = [args.page_size]
        page_size_source = "explicit_page_size"
    else:
        page_sizes = [1]
        page_size_source = "sglang_default"
    inventory = build_static_inventory(
        model_name=args.model_name,
        hf_config_path=args.hf_config,
        definitions_dir=args.definitions_dir,
        sglang_root=sglang_root,
        tp=args.tp,
        page_sizes=page_sizes,
        page_size_source=page_size_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nStatic Kernel Inventory ({inventory['summary']['total']} candidates):")
    for kernel in inventory["kernels"]:
        print(
            f"  {kernel['definition_name']:50s} "
            f"op={kernel['op_type']:12s} variant={kernel['variant']:12s} "
            f"fi_api={kernel.get('fi_api') or '-'}"
        )
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
