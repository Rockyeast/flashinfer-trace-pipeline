#!/usr/bin/env python3
"""Generate review-only parse-rule proposals for LLM-classified trace IDs.

This tool reads a pipeline run's kernel_inventory.json, gathers
llm_classified_trace_ids (and optionally unmatched_trace_ids), enriches them with
signature samples from aggregated_summary.json, and asks an LLM for parse-rule
proposals.

The output is intentionally review-only. It does not edit parse/parse_probe.py and
does not turn candidates into inventory entries.

Usage:
  python tools/propose_parse_rules.py --run-dir tmp/run/<run>

  python tools/propose_parse_rules.py \
    --inventory tmp/run/google_gemma-3-27b-it_20260503_161022/kernel_inventory.json \
    --summary tmp/run/google_gemma-3-27b-it_20260503_161022/probe/aggregated_summary.json

Outputs by default:
  <run-dir>/llm_diagnostics/parse_rules/
    proposal_input.json
    prompt.md
    response.md
    parse_rule_proposals.json
    parse_rule_proposals.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any


DEFAULT_MODEL = "claude-sonnet-4-6"


def _json_default(obj: Any) -> str:
    return str(obj)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _infer_default_out_dir(inventory: Path) -> Path:
    # For runs, inventory lives directly under <run-dir>/kernel_inventory.json.
    # For other layouts, parent still gives a useful run-local report location.
    return inventory.parent / "llm_diagnostics" / "parse_rules"


def _resolve_run_paths(
    run_dir: Path | None,
    inventory: Path | None,
    summary: Path | None,
) -> tuple[Path, Path | None]:
    if run_dir is None:
        if inventory is None:
            raise ValueError("pass either --run-dir or --inventory")
        return inventory, summary

    if inventory is None:
        candidate = run_dir / "kernel_inventory.json"
        if not candidate.exists():
            raise ValueError(f"cannot find inventory at {candidate}")
        inventory = candidate

    if summary is None:
        candidates = [
            run_dir / "probe" / "aggregated_summary.json",
            run_dir / "aggregated_summary.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                summary = candidate
                break

    return inventory, summary


def _compact_signature(sig: dict) -> dict:
    signature = sig.get("signature") or {}
    stack = sig.get("stack") or []
    return {
        "trace_id": sig.get("trace_id"),
        "category": sig.get("category"),
        "count": sig.get("count"),
        "signature": signature,
        "stack_sample": stack[:4],
    }


def _build_signatures_by_trace(summary: dict, max_sigs_per_trace: int) -> dict[str, list[dict]]:
    by_trace: dict[str, list[dict]] = {}
    for sig in summary.get("top_signatures") or []:
        trace_id = sig.get("trace_id")
        if not trace_id:
            continue
        bucket = by_trace.setdefault(trace_id, [])
        if len(bucket) < max_sigs_per_trace:
            bucket.append(_compact_signature(sig))
    return by_trace


def _collect_items(
    inventory: dict,
    signatures_by_trace: dict[str, list[dict]],
    *,
    include_unmatched: bool,
    max_items: int,
) -> list[dict]:
    items: list[dict] = []

    for item in inventory.get("llm_classified_trace_ids") or []:
        trace_id = item.get("trace_id")
        if not trace_id:
            continue
        items.append({
            "kind": "llm_classified_trace_id",
            "trace_id": trace_id,
            "count": item.get("count"),
            "suggested_op_type": item.get("suggested_op_type"),
            "suggested_variant": item.get("suggested_variant"),
            "suggested_fi_api": item.get("suggested_fi_api"),
            "suggested_target_api": item.get("suggested_target_api"),
            "source": item.get("source"),
            "reason": item.get("reason"),
            "reasoning": item.get("reasoning"),
            "signatures": signatures_by_trace.get(trace_id, []),
        })

    if include_unmatched:
        for item in inventory.get("unmatched_trace_ids") or []:
            trace_id = item.get("trace_id")
            if not trace_id:
                continue
            items.append({
                "kind": "unmatched_trace_id",
                "trace_id": trace_id,
                "count": item.get("count"),
                "suggested_op_type": None,
                "suggested_variant": None,
                "suggested_fi_api": None,
                "source": "unmatched",
                "reason": "Parse did not classify this trace_id.",
                "signatures": signatures_by_trace.get(trace_id, []),
            })

    items.sort(key=lambda x: int(x.get("count") or 0), reverse=True)
    return items[:max_items]


def _build_prompt(model_name: str | None, items: list[dict]) -> str:
    items_json = json.dumps(items, indent=2, ensure_ascii=False)
    return dedent(f"""
        You are helping review a GPU-kernel tracing pipeline for FlashInfer-Bench.

        The parser produced llm_classified_trace_ids/unmatched_trace_ids. These
        are NOT formal inventory entries yet. Your task is to propose parse-rule
        follow-ups for human review. Do not assume the proposal is correct unless
        the signatures support it.

        Model/run: {model_name or "unknown"}

        For each item, output one JSON object with:
        - trace_id
        - review_decision: one of ["promote", "ignore", "defer"]
        - proposed_op_type: snake_case or null
        - proposed_variant: snake_case/string or null
        - proposed_fi_api: official FlashInfer API string beginning with "flashinfer.", or null
        - proposed_target_api: non-FlashInfer API string or null
        - definition_name_template: e.g. "gelu_tanh_and_mul_i{{intermediate_size}}" or null
        - params_to_extract: object mapping param name to extraction strategy
        - collector_notes: how workload collection should hook/capture this, or null
        - test_generator_notes: how a test generator would verify it, or null
        - risks: list of assumptions or hazards
        - confidence: "low", "medium", or "high"
        - reasoning: concise explanation

        Use "ignore" for helper functions, registration functions, logging,
        shape checks, or top-level modules that are not benchmark kernels.
        Use "defer" when the trace looks meaningful but needs source-code review.
        Use "promote" only when it likely corresponds to a benchmarkable kernel
        and enough parameters can be inferred from signatures.
        Prefer concrete benchmark kernel families over umbrella categories. For
        example, propose `gelu_tanh_and_mul` with variant `default`, not
        `activation` with variant `gelu_tanh_and_mul`.

        Important: this is a proposal only. Do not produce Python source edits.

        Input items:
        ```json
        {items_json}
        ```

        Respond with ONLY JSON:
        ```json
        {{
          "proposals": [
            {{
              "trace_id": "...",
              "review_decision": "promote|ignore|defer",
              "proposed_op_type": null,
              "proposed_variant": null,
              "proposed_fi_api": null,
              "proposed_target_api": null,
              "definition_name_template": null,
              "params_to_extract": {{}},
              "collector_notes": null,
              "test_generator_notes": null,
              "risks": [],
              "confidence": "low|medium|high",
              "reasoning": "..."
            }}
          ]
        }}
        ```
    """).strip()


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_json_response(text: str) -> dict:
    match = _JSON_BLOCK_RE.search(text)
    raw = match.group(1).strip() if match else text.strip()
    return json.loads(raw)


def _call_anthropic(
    prompt: str,
    *,
    model: str,
    base_url: str | None,
    max_tokens: int,
) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is not installed") from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    effective_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
    if effective_base_url:
        client = anthropic.Anthropic(api_key=api_key, base_url=effective_base_url)
    else:
        client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _chunk_items(items: list[dict], batch_size: int) -> list[list[dict]]:
    """Split proposal items into bounded LLM requests."""
    batch_size = max(1, batch_size)
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _dry_run_proposals(items: list[dict]) -> dict:
    proposals = []
    for item in items:
        proposals.append({
            "trace_id": item["trace_id"],
            "review_decision": "defer",
            "proposed_op_type": item.get("suggested_op_type"),
            "proposed_variant": item.get("suggested_variant"),
            "proposed_fi_api": item.get("suggested_fi_api"),
            "proposed_target_api": item.get("suggested_target_api"),
            "definition_name_template": None,
            "params_to_extract": {},
            "collector_notes": None,
            "test_generator_notes": None,
            "risks": ["Dry-run placeholder; no LLM was called."],
            "confidence": "low",
            "reasoning": item.get("reason") or "Review this trace_id manually.",
        })
    return {"proposals": proposals}


def _write_markdown(path: Path, inventory_path: Path, summary_path: Path | None, proposals: dict) -> None:
    lines = [
        "# Parse Rule Proposals",
        "",
        f"- inventory: `{inventory_path}`",
        f"- summary: `{summary_path}`" if summary_path else "- summary: not provided",
        "",
        "These are review-only proposals. They do not modify parser rules.",
        "",
    ]

    for prop in proposals.get("proposals", []):
        trace_id = prop.get("trace_id", "<unknown>")
        decision = prop.get("review_decision", "defer")
        confidence = prop.get("confidence", "low")
        lines.extend([
            f"## `{trace_id}`",
            "",
            f"- decision: `{decision}`",
            f"- confidence: `{confidence}`",
            f"- op_type: `{prop.get('proposed_op_type')}`",
            f"- variant: `{prop.get('proposed_variant')}`",
            f"- fi_api: `{prop.get('proposed_fi_api')}`",
            f"- target_api: `{prop.get('proposed_target_api')}`",
            f"- definition_name_template: `{prop.get('definition_name_template')}`",
            "",
            "params_to_extract:",
            "```json",
            json.dumps(prop.get("params_to_extract") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
        ])
        if prop.get("collector_notes"):
            lines.extend(["collector_notes:", "", str(prop["collector_notes"]), ""])
        if prop.get("test_generator_notes"):
            lines.extend(["test_generator_notes:", "", str(prop["test_generator_notes"]), ""])
        risks = prop.get("risks") or []
        if risks:
            lines.append("risks:")
            for risk in risks:
                lines.append(f"- {risk}")
            lines.append("")
        if prop.get("reasoning"):
            lines.extend(["reasoning:", "", str(prop["reasoning"]), ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LLM parse-rule proposals for review.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing kernel_inventory.json")
    parser.add_argument("--inventory", type=Path, default=None, help="Path to kernel_inventory.json")
    parser.add_argument("--summary", type=Path, default=None, help="Path to probe/aggregated_summary.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for proposal files")
    parser.add_argument("--include-unmatched", action="store_true", help="Also include unmatched_trace_ids")
    parser.add_argument(
        "--max-items",
        "--max-candidates",
        dest="max_items",
        type=int,
        default=20,
        help="Maximum LLM-classified/unmatched items to propose on",
    )
    parser.add_argument("--max-signatures-per-trace", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of trace_ids per LLM request (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum response tokens per LLM request (default: 4096)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM; write placeholder proposals")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Anthropic model (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=None, help="Anthropic-compatible proxy base URL")
    args = parser.parse_args()

    inventory_path, summary_path = _resolve_run_paths(args.run_dir, args.inventory, args.summary)
    inventory = _load_json(inventory_path)
    summary = _load_json(summary_path) if summary_path else {}
    out_dir = args.out_dir or _infer_default_out_dir(inventory_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    signatures_by_trace = _build_signatures_by_trace(summary, args.max_signatures_per_trace)
    items = _collect_items(
        inventory,
        signatures_by_trace,
        include_unmatched=args.include_unmatched,
        max_items=args.max_items,
    )

    model_name = inventory.get("model") or summary.get("model_name")
    proposal_input = {
        "model": model_name,
        "inventory": str(inventory_path),
        "summary": str(summary_path) if summary_path else None,
        "item_count": len(items),
        "items": items,
    }
    _write_json(out_dir / "proposal_input.json", proposal_input)

    if args.dry_run:
        prompt = _build_prompt(model_name, items)
        (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        response_text = ""
        proposals = _dry_run_proposals(items)
    else:
        batches = _chunk_items(items, args.batch_size)
        all_proposals: list[dict] = []
        prompt_parts: list[str] = []
        response_parts: list[str] = []
        for index, batch in enumerate(batches, start=1):
            prompt = _build_prompt(model_name, batch)
            prompt_parts.extend([
                f"# Batch {index}/{len(batches)}",
                "",
                prompt,
                "",
            ])
            response_text = _call_anthropic(
                prompt,
                model=args.model,
                base_url=args.base_url,
                max_tokens=args.max_tokens,
            )
            response_parts.extend([
                f"# Batch {index}/{len(batches)}",
                "",
                response_text,
                "",
            ])
            batch_proposals = _parse_json_response(response_text)
            all_proposals.extend(batch_proposals.get("proposals", []))
        (out_dir / "prompt.md").write_text("\n".join(prompt_parts), encoding="utf-8")
        (out_dir / "response.md").write_text("\n".join(response_parts), encoding="utf-8")
        proposals = {"proposals": all_proposals}

    _write_json(out_dir / "parse_rule_proposals.json", proposals)
    _write_markdown(out_dir / "parse_rule_proposals.md", inventory_path, summary_path, proposals)
    apply_manifest = {
        "schema_version": 1,
        "source_inventory": str(inventory_path),
        "source_summary": str(summary_path) if summary_path else None,
        "proposals": proposals.get("proposals", []),
    }
    _write_json(out_dir / "apply_manifest.json", apply_manifest)

    print(f"Wrote proposal input: {out_dir / 'proposal_input.json'}")
    print(f"Wrote proposals:      {out_dir / 'parse_rule_proposals.json'}")
    print(f"Wrote markdown:       {out_dir / 'parse_rule_proposals.md'}")
    print(f"Wrote apply manifest: {out_dir / 'apply_manifest.json'}")
    print(f"Items: {len(items)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
